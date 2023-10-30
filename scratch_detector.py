from ultralytics import YOLO
from flask import request, Response, Flask
from waitress import serve
from PIL import Image
import json
import numpy as np
import cv2
from plantcv import plantcv as pcv

app = Flask(__name__)

"""
Root handler returns contents of html file
"""
@app.route("/")
def root():
    with open("index.html") as file:
        return file.read()


"""
Receives uploaded file and passes it into object detection model
Coordinates and lengths of detected scratches is returned as response
"""
@app.route("/detect", methods=["POST"])
def detect():
    buf = request.files["image_file"]
    boxes = detect_objects_on_image(Image.open(buf.stream), True)
    return Response(
      json.dumps(boxes),
      mimetype='application/json'
    )


"""
Receives image and fits trained model weights
For each detected object, calculate the scratch length
Return list of scratches with coordinates and length
"""
def detect_objects_on_image(buf, from_file):
    # Apply best weights from training to test image
    model = YOLO("best.pt")
    results = model.predict(buf)
    result = results[0]
    output = []

    # Convert image into opencv format
    buf.save("asset.jpg")
    pil_img = Image.open("asset.jpg")
    opencv_img = np.array(pil_img)
    if opencv_img.shape[2] == 4:
        opencv_img = cv2.cvtColor((opencv_img, cv2.COLOR_RGBA2BGR))

    # Iterate through detected scratches to calculate lengths
    for box in result.boxes:
        x1, y1, x2, y2 = [
          round(x) for x in box.xyxy[0].tolist()
        ]
        length = scratch_len(opencv_img, [x1,y1,x2,y2])
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        output.append([
          x1, y1, x2, y2, result.names[class_id], prob, round(length,2)
        ])
    return output


"""
Apply canny edge detection to test image to highlight scratch outline
Apply dilation and erosion transformations to fill scratch outline
"""
def detected_edges(img_cv):
    # Convert to grayscale
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

    # Apply canny edge detection
    edges = cv2.Canny(image=img_blur, threshold1=50, threshold2=100)

    # Apply dilation and erosion transformations
    edges = cv2.dilate(edges, None, iterations=1)
    edges = cv2.erode(edges, None, iterations=1)

    return edges


"""
Passes in test image and returns scratch length measurement
"""
def scratch_len(img_cv, location, plot=True):

    x1, y1, x2, y2 = location
    asset_img = img_cv[y1:y2, x1:x2]

    edges = detected_edges(asset_img)

    # threshold image
    _, edges_bin = cv2.threshold(edges, 40, 255, cv2.THRESH_BINARY)

    # Convert black and white pixel values into binary format
    height, width = edges_bin.shape
    for i in range(height):
        for j in range(width):
            edges_bin[i][j] = 1 if edges_bin[i][j] == 255 else 0

    # Skeletonise image
    edges_skel = pcv.morphology.skeletonize(edges_bin)  # pcv skeletonize returns 0 and 1 img / skimage skel returns True and False values

    # Get image contours
    contours, hierarchy = cv2.findContours(edges_skel, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    edge_contours = [c for c in contours if cv2.arcLength(c, False) > 100]

    # Get contour perimeter, divide by two and multiply by calibration factor
    measurement = 0
    for i, cnt in enumerate(edge_contours):
        current_measurement = float(cv2.arcLength(edge_contours[i], False) / 2) * 0.052
        if current_measurement > measurement:
            measurement = current_measurement

    return measurement


serve(app, host='0.0.0.0', port=8080)