# Screen-crack-detection-using-Yolov8
# Purpose
---
This mini project aim to test the availability of using Yolo V8 as model for phone screen crack detection. The procedure includes data collection from public,
data annotation, model selection and performance evaluation. The overall development period of this project is 1 week, and thus it only focus on model functionality instead of accuracy.

---
# Result preview
<img height="270" alt="result-2" src="https://github.com/Jarvis2030/Screen-crack-detection-using-Yolov8/assets/77675271/f0afed7b-d28f-4c83-9722-ba01055a5e39"> 
<img height="270" alt="result-v1" src="https://github.com/Jarvis2030/Screen-crack-detection-using-Yolov8/assets/77675271/77db917a-a257-42f6-8c4a-98adaabb3429"> 
<img height="270" alt="server detection image" src="https://github.com/Jarvis2030/Screen-crack-detection-using-Yolov8/assets/77675271/4a440bce-7808-431e-939f-46387f6203a6"> 

# overview
---
The widespread use of mobile phones has led to increased demand and rising prices in the market. As a result, more people are turning to the used phone market, which is projected to reach a value of 99 billion by 2026. Accurate price estimation in this market is crucial, with the condition of the mobile screen playing a significant role. Damage or cracks on the screen impact sensitivity and appearance, affecting customer willingness to purchase. Automating the inspection process is essential to streamline operations, reduce dependence on human resources, and minimize price variability. <br>
This project aims to create a mobile platform for internal price estimation during clients' used phone transition. It will incorporate an AI detection model, historical price data, and an operational portal to enhance efficiency and standardize the trading process. By leveraging cutting-edge technology, the platform aims to make buying and selling used phones more accessible, efficient, and profitable for all parties involved.

# Dataset
---
To train precisely, each categories required 1000-2000 bounding box to reach the optimum accuracy.

<h3>Open-source Dataset:</h3>

[SurfaceDefectDetectionDataset](https://blog.csdn.net/qq_27871973/article/details/84974231) <br>
[Cracked Mobile Screen Dataset](https://www.kaggle.com/datasets/dataclusterlabs/cracked-screen-dataset?resourcedownload) <br>
[Mobile Phone Screen Surface Defect Segmentation Dataset](https://github.com/jianzhang96/MSD) <br>


# Model
## Yolo V8
official github link: https://github.com/ultralytics/ultralytics <br>
Yolo (You only look once) is a real-time detection system. It is featured for its accuracy and fast-training conpared to other detection models. 
Its high effciency come from the operation: <br>
1. The image wil be segmented into m*n grid-cell. Each grid-cell is responsible for localizing and predicting whether the 
target object is within the cell, giving a score of probability.
2. For all the cell that probailibity > 0, it create a bounding box of the prediction on the target object.
3. After confirming the possible objects in each cell, the Intersection of Union (IOU) will come in place.

Since this is a testing model, I eventually use V8-nano for faster training and less memory occupied.


# Performance iImprovement
---
## Preliminary Analysis
<li> Check the performance graph and find the direction for the improve.
<li>Criteria: Recall, precision, confusion matrix, train loss, validation loss

|Category|	Observation	|Possible issue|
|:------:|:-----------:|:------------:|
|Monitor Scratch	| High precision low recall|
|Monitor Crack|	High precision, low recall	|
|Burn-in	|High precision, low recall|
|Dead pixel	|Balanced precision & recall|	N/A|


### Approach:
1.	Adjust threshold.
2.	Hyperparameter tuning.
3.	Transfer learning

## Data processing
<li>Data augmentation: increase training data amount (goal: >= 500 per category)
<li>Test time augmentation (TTA): increase training data amount (goal: >= 500 per category) </li>
<li>Data feature enhancement & noise cleaning: remove background & noise, apply threshold, etc. </li>

### Approach:
<li>TTA can be done by YOLO utils function, Data augmentation will be hand-crafted.</li>
<li>Data feature enhancement: Considering apply ResNet50 backbone for feature pyramid hierarchy and input the result to YOLO for detection.</li>



# Evaluation
[comment]: <> (
<img height="400" alt="server detection image" src="https://github.com/Jarvis2030/Screen-crack-detection-using-Yolov8/assets/77675271/6896a97e-3a79-453c-9e9e-f9abd9d2209f"> 
<img height="400" alt="server detection image" src="https://github.com/Jarvis2030/Screen-crack-detection-using-Yolov8/assets/77675271/39942e39-05ef-4398-a750-c9cf361d1409"> 
<img height="400" alt="server detection image" src="https://github.com/Jarvis2030/Screen-crack-detection-using-Yolov8/assets/77675271/118058bf-2acc-41ab-9f5e-2a36879b8b5c"> 
<img height="400" alt="server detection image" src="https://github.com/Jarvis2030/Screen-crack-detection-using-Yolov8/assets/77675271/05933568-58f8-48c1-981b-849b4d63ced0"> )


|Category	| Defect Type	|Achievability|Best Approach|Recall|Precision|F1 score|
|:-------:|:-----------:|:-----------:|:-----------:|:----:|:-------:|:------:|
External	|Monitor Scratch	|Yes|	YOLOv8|	63.6%	|89.4%|	74.3%|
External |Monitor Crack|	Yes|	YOLOv8|55.7%	|75.1%|63.9%|
External	|Case Scratch	|N/A|	N/A|	N/A	|N/A	|N/A|
Internal	|Burn-in	|Yes|	YOLOv8|	74.6%|	96.5%|	84.1%|
Internal	|Dead pixel	|Yes	|YOLOv8 |53.4%	|53.8%	|53.6%|
Internal	|Partly not functioning display	|N/A|	N/A|	N/A|	N/A|	N/A|


 
