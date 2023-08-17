# Screen-crack-detection-using-Yolov8
## Purpose
---
This mini project aim to test the availability of using Yolo V8 as model for phone screen crack detection. The procedure includes data collection from public,
data annotation, model selection and performance evaluation. The overall development period of this project is 1 week, and thus it only focus on model functionality instead of accuracy.

## content
---
<li>Result preview</li>
<li>Overview</li>
<li>Dataset</li>
<li>Model</li>
<li>Training</li>
<li>Evaluation</li>

---
## Result preview


## overview


## Dataset


## Model
### Yolo V8
official github link: https://github.com/ultralytics/ultralytics <br>
Yolo (You only look once) is a real-time detection system. It is featured for its accuracy and fast-training conpared to other detection models. 
Its high effciency come from the operation: <br>
1. The image wil be segmented into m*n grid-cell. Each grid-cell is responsible for localizing and predicting whether the 
target object is within the cell, giving a score of probability.
2. For all the cell that probailibity > 0, it create a bounding box of the prediction on the target object.
3. After confirming the possible objects in each cell, the Intersection of Union (IOU) will come in place.

Since this is a testing model, I eventually use V8-nano for faster training and less memory occupied.


## Training


## Evaluation
 
