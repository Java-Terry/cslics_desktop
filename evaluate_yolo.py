#!/usr/bin/env/python3

"""Script to numerically evaluate the performance of a YOLO model"""

import os
import glob
import cv2 as cv
import numpy as np
import math
from ultralytics import YOLO

#weights_file = '/home/java/Java/ultralytics/runs/detect/train - alor_atem_1000/weights/best.pt'
weights_file = '/home/java/Java/ultralytics/runs/detect/cslics_dektop_1000_cslics/weights/best.pt'
img_dir = '/home/java/Java/data/cslics_desktop_data/202311_Nov_cslics_desktop_sample_images/images'
imgsave_dir = '/home/java/Java/data/cslics_desktop_data/202311_Nov_cslics_desktop_sample_images/detect'
label_dir = '/home/java/Java/data/cslics_desktop_data/202311_Nov_cslics_desktop_sample_images/labels'
meta_dir = '/home/java/Java/cslics'
max_no = 10
model = YOLO(weights_file)

img_list = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))

with open(os.path.join(meta_dir, 'metadata','obj.names'), 'r') as f:
    classes = [line.strip() for line in f.readlines()]
    classes.append('Larvae')

orange = [255, 128, 0] # four-eight cell stage
blue = [0, 212, 255] # first cleavage
purple = [170, 0, 255] # two-cell stage
yellow = [255, 255, 0] # advanced
brown = [144, 65, 2] # damaged
green = [0, 255, 00] # egg
class_colours = {classes[0]: orange,
                classes[1]: blue,
                classes[2]: purple,
                classes[3]: yellow,
                classes[4]: brown,
                classes[5]: green,
                classes[6]: [0, 255, 0]}

def draw_dotted_rect(img, x1, y1, x2, y2, color, thicknes, gap=30):
    pts = [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]
    start=pts[0]
    end=pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        start=end
        end=p
        # draw dashed line
        dist = ((start[0]-end[0])**2 + (start[1]-end[1])**2)**.5
        parts = []
        for i in np.arange(0,dist,gap):
            r = i/dist
            x = int((start[0]*(1-r)+end[0]*r)+.5)
            y = int((start[1]*(1-r)+end[1]*r)+.5)
            p = (x,y)
            parts.append(p)
        for p in parts:
            cv.circle(img,p,thicknes,color,-1)

def ground_truth_compare_predict(img_rgb, txt_dir, imgname, predictions, imgsavedir, BGR=True):
    """Shows an image with ground truth annotations and predictions to help compare the differences"""
    # ground truth section
    imgw, imgh = img_rgb.shape[1], img_rgb.shape[0]
    basename = os.path.basename(imgname)
    ground_truth_txt = os.path.join(txt_dir, basename[:-4] + '.txt')
    if os.path.exists(ground_truth_txt):
        with open(ground_truth_txt, 'r') as f:
            lines = f.readlines() # <object-class> <x> <y> <width> <height>
        for part in lines:
            parts = part.rsplit()
            class_idx = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            ow = float(parts[3])
            oh = float(parts[4])
            x1 = (x - ow/2)*imgw
            x2 = (x + ow/2)*imgw
            y1 = (y - oh/2)*imgh
            y2 = (y + oh/2)*imgh
            cv.rectangle(img_rgb, (int(x1), int(y1)), (int(x2), int(y2)), class_colours[classes[class_idx]], 2)
    # predictions section
    for p in predictions:
        x1, y1, x2, y2 = p[0:4]
        conf = p[4]
        cls = int(p[5])
        #extract back into cv lengths
        x1 = x1*imgw
        x2 = x2*imgw
        y1 = y1*imgh
        y2 = y2*imgh       
        draw_dotted_rect(img_rgb, int(x1), int(y1), int(x2), int(y2), class_colours[classes[cls]], 7)
        #cv.rectangle(img_rgb, (int(x1), int(y1)), (int(x2), int(y2)), self.class_colours[self.classes[cls]], 2)
        cv.putText(img_rgb, f"{classes[cls]}: {conf:.2f}", (int(x1), int(y1 - 5)), cv.FONT_HERSHEY_SIMPLEX, 1, class_colours[classes[cls]], 2)
    # save image
    imgsavename = os.path.basename(imgname)
    imgsave_path = os.path.join(imgsavedir, imgsavename[:-4] + '.jpg')    
    img_bgr = cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR) # RGB
    cv.imwrite(imgsave_path, img_bgr)

def save_image_predictions(predictions, img, imgname, imgsavedir, BGR=True):
    """
    save predictions/detections (assuming predictions in yolo format) on image
    """
    # img = cv.imread(imgname)
    # assuming input image is rgb, need to convert back to bgr:
    
    imgw, imgh = img.shape[1], img.shape[0]
    for p in predictions:
        x1, y1, x2, y2 = p[0:4]
        conf = p[4]
        cls = int(p[5])
        #extract back into cv lengths
        x1 = x1*imgw
        x2 = x2*imgw
        y1 = y1*imgh
        y2 = y2*imgh        
        try:
            cv.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), class_colours[classes[cls]], 2)
            cv.putText(img, f"{classes[cls]}: {conf:.2f}", (int(x1), int(y1 - 5)), cv.FONT_HERSHEY_SIMPLEX, 0.5, class_colours[classes[cls]], 2)
        except:
            import code
            code.interact(local=dict(globals(), **locals()))
    imgsavename = os.path.basename(imgname)
    imgsave_path = os.path.join(imgsavedir, imgsavename[:-4] + '.jpg')
    if BGR:
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)        
    cv.imwrite(imgsave_path, img)
    return True

def count_by_prediction(predictions):
    count = 0
    if len(predictions)==0 or predictions is None:
        return 0
    else:
        for p in predictions:
            count += 1
    #class counts
        count_dir = {}
        for p in predictions:
            cls = int(p[5])
            count_dir[cls] = count_dir.get(cls, 0) + 1
        print("Preidiction Class counts:")
        for class_label, num in count_dir.items():
            print("Class:", class_label, "| Count:", num)
    return count
      
def get_label_counts(img_base_name, label_dir, show_counts=False):
    """From a label file, get the yolo labels from label_dir and count the instances."""
    larva_counts = {}
    label_file_path = os.path.join(label_dir, img_base_name + '.txt')
    if not os.path.isfile(label_file_path):
        print(f'no label file found for {img_base_name}')
        return None
    with open(label_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            class_number = line.split()[0]
            larva_counts[class_number] = larva_counts.get(class_number, 0) + 1
    total_counts = sum(larva_counts.values())
    if show_counts:
        print("Larvae counts for", img_base_name + ":")
        for class_label, count in larva_counts.items():
            print("Class:", class_label, "| Count:", count)
        print("Total number of larvae:", total_counts)
    return total_counts  

count_list = []
for i, img_name in enumerate(img_list):
    if i > max_no:
        break
    print(f"processing {i}/{len(img_list)}")
    image = cv.imread(img_name)
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    img_name_base = os.path.basename(img_name).split('.')[0]

    pred = model.predict(source=image,
                                  save=False,
                                  save_txt=False,
                                  save_conf=True,
                                  verbose=False,
                                  imgsz=640,
                                  conf=0.5)
    boxes = pred[0].boxes 
    pred = []
    for b in boxes:
        cls = int(b.cls)
        conf = float(b.conf)
        xyxyn = b.xyxyn.cpu().numpy()[0]
        x1n = xyxyn[0]
        y1n = xyxyn[1]
        x2n = xyxyn[2]
        y2n = xyxyn[3]  
        pred.append([x1n, y1n, x2n, y2n, conf, cls])
    predictions = pred
  
    ground_truth_compare_predict(image_rgb, label_dir, img_name, predictions, imgsave_dir)
    #save_image_predictions(predictions, image_rgb, img_name, imgsave_dir)
    larvae_count = get_label_counts(img_name_base, label_dir, show_counts=True)
    blob_count = count_by_prediction(predictions)
    print(f'Count instances: {blob_count} | Ground_truth count: {larvae_count}')
    count_list.append((blob_count, larvae_count))

count_acc_list = []
for counts in count_list:
    count_acc_list.append(abs(counts[0] - counts[1])/counts[1])

print('------------   Evaluation ----------------------')
print(f'average accuracy: {1-(sum(count_acc_list)/len(count_acc_list))}')
   
import code
code.interact(local=dict(globals(), **locals()))
print('Done')

# ### Use yolo validation
# metrics = model.val(data='cslics_desktop.yml')  # no arguments needed, dataset and settings remembered
# metrics.box.map    # map50-95
# metrics.box.map50  # map50
# metrics.box.map75  # map75
# metrics.box.maps   # a list contains map50-95 of each category
