#!/usr/bin/env/python3

"""Script to numerically evaluatte the performance of a YOLO model"""

import os
import glob
import cv2 as cv
import numpy as np
import math
import machinevisiontoolbox as mvt
from machinevisiontoolbox.Image import Image as MvtImage
import matplotlib.pyplot as plt

weights_file = '/home/'
img_dir = '/home/'
img_save_dir = '/home/'
max_no = 10
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = YOLO(weights_file)

img_list = sorted(os.path.join(img_dir, '*.jpg'))

def save_image_predictions(self, predictions, img, imgname, imgsavedir, BGR=True):
      """
      save predictions/detections (assuming predictions in yolo format) on image
      """
      # img = cv.imread(imgname)
      # assuming input image is rgb, need to convert back to bgr:
      
      imgw, imgh = img.shape[1], img.shape[0]
      for p in predictions:
          x1, y1, x2, y2 = p[0:4].tolist()
          conf = p[4]
          cls = int(p[5])
          #extract back into cv lengths
          x1 = x1*imgw
          x2 = x2*imgw
          y1 = y1*imgh
          y2 = y2*imgh        
          cv.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), self.class_colours[self.classes[cls]], 2)
          cv.putText(img, f"{self.classes[cls]}: {conf:.2f}", (int(x1), int(y1 - 5)), cv.FONT_HERSHEY_SIMPLEX, 0.5, self.class_colours[self.classes[cls]], 2)

      imgsavename = os.path.basename(imgname)
      imgsave_path = os.path.join(imgsavedir, imgsavename[:-4] + self.SUFFIX_IMG)
      if BGR:
          img = cv.cvtColor(img, cv.COLOR_RGB2BGR)        
      cv.imwrite(imgsave_path, img)
      return True

def count_instances(predictions):
  if len(predicitions)==0 or predictions is None:
    return 0;
  else:
    boxes: Boxes = pred[0].boxes
    print(boxes)
  

for i, img_name in enumerate(img_list):
  if i > max_no:
    break
  image = cv.imread(img_name)
  image_rgb = cv.cvtColor(image, cv.Color_BGR2RGB)

  pred = model.predict(source=image,
                                  save=False,
                                  save_txt=False,
                                  save_conf=True,
                                  verbose=False,
                                  imgsz=self.img_size,
                                  conf=self.conf)
  boxes: Boxes = pred[0].boxes 
  pred = []
  for b in boxes:
            if torch.cuda.is_available():
                xyxyn = b.xyxyn[0]
                pred.append([xyxyn[0], xyxyn[1], xyxyn[2], xyxyn[3], b.conf, b.cls])
            else:
                cls = int(b.cls)
                conf = float(b.conf)
                xyxyn = b.xyxyn.cpu().numpy()[0]
                x1n = xyxyn[0]
                y1n = xyxyn[1]
                x2n = xyxyn[2]
                y2n = xyxyn[3]  
                pred.append([x1n, y1n, x2n, y2n, conf, cls])
        
  # after iterating over boxes, make sure pred is on GPU if available (and a single tensor)
  if torch.cuda.is_available():
            pred = torch.tensor(pred, device="cuda:0")
  else:
            pred = torch.tensor(pred)
  if len(pred) > 0:
            predictions = self.nms(pred, self.conf, self.iou)
  else:
            predictions = [] # empty/0 case
  save_image_predictions(predictions, image_rgb, img_name, imgsave_dir)
  count_instances(predictions)

