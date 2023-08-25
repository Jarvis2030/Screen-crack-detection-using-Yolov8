from ultralytics import YOLO
import torch

class yolomodel():
    def __init__(self):
        self.model = YOLO('yolov5n.yaml')
        self.service = torch.device("mps")
        print("Using service:", self.service)

    def trainmodel(self):
        self.model.to(self.service)
        self.model.train(data = 'config.yaml',epochs = 100)
    
    def __call__(self):
        self.trainmodel()

Yolov1 = yolomodel()
Yolov1()