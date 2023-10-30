# Scratch Detection
Using YOLOv8 object detection, this webapp uses a trained object detection model to classify scratches on laptop images. 

## Requirements
* Ultralytics
* Flask
* Waitress
* PIL
* OpenCV
* PlantCV
* JSON
* Numpy

## Training
* Training images have been labelled in https://app.roboflow.com/
* Run each line of scratch_detection_training.ipynb to complete training of model and view validation results
* Best weights from training captured in best.pt

## Testing
* Ensure best.pt is in the same directory as scratch_detector.py
* Run scratch_detector.py and access webpage via http://localhost:8080/
* Choose file from 'Demo Test Images' folder to run object detection on test image


