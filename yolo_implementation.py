
import torch
from ultralytics import YOLO
import numpy as np
import cv2
from time import time

class defectDetaction():

    def __init__(self, model = None,cap_index = 0) -> None:
        self.cap_index = cap_index
        self.model = self.load_model(model)
        self.classes = self.model.names
        self.service = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using service:", self.service)


    def load_model(self, model_name):
        if model_name: 
            print("we got a model!")
            model = torch.hub.load('.', 'custom', path = model_name, source='local') 
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained = True)
        
        return model
    
    def score_frame(self, frame):
        self.model.to(self.service)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:,-1], results.xyxyn[0][:, :-1]
        return labels, cord
    
    def class_to_label(self, label_class):

        return self.classes[int(label_class)]
    
    def plot_boxes(self, result, frame):
        label, cord = result
        n = len(label)
        x,y = frame.shape[1],frame.shape[0]
        for i in range(n):
            row = cord[i]
            print(row)
            if row[4] >= 0.2: # Thereshold for the confidence score
                x1, y1, x2, y2 = int(row[0]*x), int(row[1]*y), int(row[2]*x), int(row[3]*y)
                outline_bg = (0,255,0)
                cv2.rectangle(frame, (x1,y1),(x2,y2),outline_bg,2)
                cv2.putText(frame, self.class_to_label(label[i]),(x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, outline_bg, 2)
            
            return frame
    
    def __call__(self):

            cam = cv2.VideoCapture(1)
            while cam.isOpened():

                ret,frame = cam.read()

            
                starttime = time()
                results = self.score_frame(frame)
                frame = self.plot_boxes(results, frame)
                endtime = time()
                
                fps = 1/np.round(endtime-starttime, 2)
                cv2.putText(frame, f"FPS: {fps}", (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

                cv2.imshow("Yolo detection", frame)

                k = cv2.waitKey(1) & 0xFF

                if k == 27:
                    break


detector = defectDetaction(cap_index=0, model='best.pt')
detector()
