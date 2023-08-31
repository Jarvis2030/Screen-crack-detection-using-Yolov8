
import torch
import numpy as np
import pandas as pd
import cv2
from time import time
import yaml

from yolov5.utils.metrics import bbox_iou
from yaml.loader import SafeLoader
from argparse import ArgumentParser

class defectDetection:

    def __init__(self, 
                 model = None, 
                 model_type ="YOLO", 
                 op_type = "Prediction", 
                 cap_index = 0,
                 data_path = None,
                 ) -> None:
        
        self.cap_index = cap_index
        self.model_type = model_type
        self.model = self.load_model(model)
        self.op_type = op_type # either "Real-time" or "Image"
        self.classes = self.model.names
        self.data_path = data_path
        self.service = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using service:", self.service)



    def load_model(self, model_name):
        if self.model_type == 'YOLO':
            if model_name: 
                model = torch.hub.load('.', 'custom', path = model_name, source='local') 
            else:
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained = True)
        
        return model
    

    def class_to_label(self, label_class):
        return self.classes[int(label_class)]
    
    
    def score_frame(self, frame):

        self.model.to(self.service)
        results = self.model(frame)

        if self.op_type == 'Testing':
            # find the ground truth from the data
            truth = torch.tensor([list(map(float, x.split())) for x in open("./labels/test/.txt").readlines()])
            # find all the iou of the image
            all_iou = []
            for r in results.xywh[0]:
                iou = float(max(bbox_iou(r[None, :4], truth[:, 1:])))
                all_iou.append(iou)
                *xywh, p, c = r
                frame = self.plot_boxes(frame, xywh, p, c, iou)
            self.save_result(all_iou)
        
        
        if self.op_type == "Prediction":
            print(self.classes)
            for r in results.xywh[0]:
                *xywh, p, c = r
                print(f"The result is: {xywh,p,c}")
                frame = self.plot_boxes(frame, xywh, p, c) # plot the result on the frame
        
        return frame
    
    
    def plot_boxes(self, frame, xywh, confidence, label, iou = None):
        x = int(xywh[0])
        y = int(xywh[1])
        w = int(xywh[2] / 2)
        h = int(xywh[3] / 2)
        print(x,y,w,h)

        pt1 = (x - w, y - h)
        pt2 = (x + w, y + h)
        cv2.rectangle(frame, pt1, pt2, (0,255,0), 2)

        text =self.class_to_label(label) + "  " + str(round(float(confidence), 2))
        if self.op_type == 'Testing': text += "  " + str(round(iou, 2))

        (w2, h2), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
        pt3 = (x - w + w2, y - h - h2)
        cv2.rectangle(frame, pt1, pt3, (0,255,0), -1)
        cv2.putText(frame, text, pt1, cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
            
        return frame
    
    def save_result (self, all_iou):
        avg = all_iou.mean()
        df = pd.DataFrame(all_iou)
        df['average'] = avg
        df.to_csv('iou_score.csv')
        

    
    def __call__(self):

        # img = cv2.imread('/Users/linyuchun/Desktop/Crowd-of-Diverse-People_800x528-768x512.jpg')
        # frame = self.score_frame(img)

        # cv2.imshow("Yolo detection", frame)
        # cv2.waitKey(5000)


        cam = cv2.VideoCapture(1)
        while cam.isOpened():

            _,frame = cam.read()

            
            starttime = time()
            results = self.score_frame(frame)
            endtime = time()
                
            fps = 1/np.round(endtime-starttime, 2)
            cv2.putText(results, f"FPS: {fps}", (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

            cv2.imshow("Yolo detection", results)

            k = cv2.waitKey(1) & 0xFF

            if k == 27:
                break


detector = defectDetection(cap_index=0, model_type= "YOLO", op_type="Prediction")
detector()
