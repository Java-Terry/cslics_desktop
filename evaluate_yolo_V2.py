#!/usr/bin/env/python3

"""Script to numerically evaluate the performance of a YOLO model using bonding boxes"""

import os
import glob
import cv2 as cv
import numpy as np
import math
from ultralytics import YOLO

class YOLOEvaluator:
    def __init__(self, weights_file, img_dir, imgsave_dir, label_dir, meta_dir, results_file_name=None, results_save_loc=None, max_no=1000):
        self.weights_file = weights_file
        self.img_dir = img_dir
        self.imgsave_dir = imgsave_dir
        self.label_dir = label_dir
        self.meta_dir = meta_dir
        self.max_no = max_no
        self.results_file_name = results_file_name if results_file_name is not None else 'results'
        self.results_save_loc = results_save_loc if results_save_loc is not None else imgsave_dir
        self.classes, self.class_colours = self.load_classes_and_colours()
        self.model = YOLO(weights_file)
        self.img_list = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
        os.makedirs(imgsave_dir, exist_ok=True)

    def load_classes_and_colours(self):
        with open(os.path.join(self.meta_dir, 'metadata','obj.names'), 'r') as f:
            classes = [line.strip() for line in f.readlines()]
            classes.append('Larvae')

        orange = [255, 128, 0]  # four-eight cell stage
        blue = [0, 212, 255]    # first cleavage
        purple = [170, 0, 255]  # two-cell stage
        yellow = [255, 255, 0]  # advanced
        brown = [144, 65, 2]    # damaged
        green = [0, 255, 0]     # egg
        
        class_colours = {classes[0]: orange,
                         classes[1]: blue,
                         classes[2]: purple,
                         classes[3]: yellow,
                         classes[4]: brown,
                         classes[5]: green,
                         classes[6]: [0, 255, 0]}

        return classes, class_colours

    def draw_dotted_rect(self, img, x1, y1, x2, y2, color, thicknes, gap=30):
        """Draws a dotted rectangle on an image"""
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

    def ground_truth_compare_predict(self, img_rgb, txt_dir, imgname, predictions, imgsavedir, BGR=True):
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
                cv.rectangle(img_rgb, (int(x1), int(y1)), (int(x2), int(y2)), self.class_colours[self.classes[class_idx]], 2)
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
            self.draw_dotted_rect(img_rgb, int(x1), int(y1), int(x2), int(y2), self.class_colours[self.classes[cls]], 7)
            #cv.rectangle(img_rgb, (int(x1), int(y1)), (int(x2), int(y2)), self.class_colours[self.classes[cls]], 2)
            cv.putText(img_rgb, f"{self.classes[cls]}: {conf:.2f}", (int(x1), int(y1 - 5)), cv.FONT_HERSHEY_SIMPLEX, 1, self.class_colours[self.classes[cls]], 2)
        # save image
        imgsavename = os.path.basename(imgname)
        imgsave_path = os.path.join(imgsavedir, imgsavename[:-4] + '.jpg')    
        img_bgr = cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR) # RGB
        cv.imwrite(imgsave_path, img_bgr)

    def save_image_predictions(self, predictions, img, imgname, imgsavedir, BGR=True):
        """save predictions/detections (assuming predictions in yolo format) on image"""
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
            cv.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), self.class_colours[self.classes[cls]], 2)
            cv.putText(img, f"{self.classes[cls]}: {conf:.2f}", (int(x1), int(y1 - 5)), cv.FONT_HERSHEY_SIMPLEX, 0.5, self.class_colours[self.classes[cls]], 2)
        imgsavename = os.path.basename(imgname)
        imgsave_path = os.path.join(imgsavedir, imgsavename[:-4] + '.jpg')
        if BGR:
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)        
        cv.imwrite(imgsave_path, img)
        return True

    def bb_get_larvae_count(self, img_base_name, label_dir):
        """From a label file, get the yolo labels from label_dir and extract the bounding box coordinates (normalised). """
        bboxes = []
        label_file_path = os.path.join(label_dir, img_base_name + '.txt')
        if not os.path.isfile(label_file_path):
            print(f'No label file found for {img_base_name}')
            return None
        with open(label_file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.split()
                class_number = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                # Calculate bounding box coordinates
                x1 = (x_center - width / 2)
                y1 = (y_center - height / 2)
                x2 = (x_center + width / 2)
                y2 = (y_center + height / 2)
                bboxes.append([x1, y1, x2, y2, class_number])
        return bboxes

    def calculate_iou(self, box1, box2):
        """Calculate the Intersection over Union (IoU) of two bounding boxes."""
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        inter_x1 = max(x1, x3)
        inter_y1 = max(y1, y3)
        inter_x2 = min(x2, x4)
        inter_y2 = min(y2, y4)
        inter_area = max(0, inter_x2 - inter_x1 + 1) * max(0, inter_y2 - inter_y1 + 1)
        box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
        box2_area = (x4 - x3 + 1) * (y4 - y3 + 1)
        union_area = box1_area + box2_area - inter_area
        iou = inter_area / union_area
        return iou

    ##TODO Remove once debugging is done
    def get_label_counts(self, img_base_name, label_dir, show_counts=False):
        """Add for debugging purposes."""
        """From a label file, get the yolo labels from label_dir and count the instances."""
        larva_counts = {}
        label_file_path = os.path.join(label_dir, img_base_name + '.txt')
        if not os.path.isfile(label_file_path):
            print(f'no label file found for {img_base_name}')
            return None
        with open(label_file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                class_number = int(line.split()[0])
                larva_counts[class_number] = larva_counts.get(class_number, 0) + 1
        if show_counts:
            for key in larva_counts:
                print(f'Label: Class {key} | {larva_counts[key]}')
        return larva_counts  
    
    ##TODO Remove once debugging is done
    def count_by_prediction(self, predictions):
        """Add for debugging purposes."""
        count_dir = {}
        if len(predictions)==0 or predictions is None:
            return count_dir
        else:
            for p in predictions:
                cls = int(p[5])
                count_dir[cls] = count_dir.get(cls, 0) + 1
        for key in count_dir:
            print(f'Predict: Class {key} | {count_dir[key]}')
        return count_dir

    def compare_predictions_with_labels(self, predictions, labels, num_classes, IOU_THRESHOLD=0.5):
        """Compare predicted results with labeled results for each class."""
        class_TP = {class_label: 0 for class_label in range(0, num_classes)}
        class_FP = {class_label: 0 for class_label in range(0, num_classes)}
        class_FN = {class_label: 0 for class_label in range(0, num_classes)}
        unmatched_predictions = predictions[:] 
        for label in labels:
            x1_label, y1_label, x2_label, y2_label, class_number_label = label
            label_box = [x1_label, y1_label, x2_label, y2_label] 
            # Find the best matching predicted box (if any)
            best_iou = 0
            best_prediction = None
            for prediction in unmatched_predictions:
                x1_pred, y1_pred, x2_pred, y2_pred, conf, class_number_pred = prediction
                pred_box = [x1_pred, y1_pred, x2_pred, y2_pred] 
                if class_number_label == class_number_pred:
                    iou = self.calculate_iou(label_box, pred_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_prediction = prediction
            if best_iou >= IOU_THRESHOLD:
                class_TP[class_number_label] += 1
                unmatched_predictions.remove(best_prediction)  # Remove the matched prediction
            else:
                class_FN[class_number_label] += 1
        for prediction in unmatched_predictions:
            _, _, _, _, _, class_number_pred = prediction
            class_FP[class_number_pred] += 1
        return class_TP, class_FP, class_FN

    def format_predictions(self, pred):
        """From model.predict() output, format the predictions into a list of lists."""
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
        return predictions

    def print_final_stats(self, total_TP, total_FP, total_FN, classstat):
        """Print the final stats."""
        total_precision = total_TP/(total_TP + total_FP)
        total_recall = total_TP/(total_TP + total_FN)
        print(f"Total Stats: TP: {total_TP}, FP: {total_FP}, FN: {total_FN}, Precision: {total_precision:.2f}, Recall: {total_recall:.2f}")
        print('Class Stats:')
        for class_label in classstat.keys():
            class_TP = classstat[class_label]['TP']
            class_FP = classstat[class_label]['FP']
            class_FN = classstat[class_label]['FN']
            if class_TP + class_FP == 0:
                class_precision = 0
            else:
                class_precision = class_TP / (class_TP + class_FP)
            if class_TP + class_FN == 0:
                class_recall = 0
            else:
                class_recall = class_TP / (class_TP + class_FN)
            print(f"{self.classes[class_label]}: TP: {class_TP}, FP: {class_FP}, FN: {class_FN}, Precision: {class_precision:.2f}, Recall: {class_recall:.2f}")
        
    def update_counts(self, total_TP, total_FP, total_FN, class_TP, class_FP, class_FN, classstat, show=False):
        """Update the counts and if show=true, shows img stats."""
        total_TP += sum(class_TP.values())
        total_FP += sum(class_FP.values())
        total_FN += sum(class_FN.values())
        img_TP = int(sum(class_TP.values()))
        img_FP = int(sum(class_FP.values()))
        img_FN = int(sum(class_FN.values()))
        if img_TP + img_FP == 0:
            img_precision = 0
        else:
            img_precision = img_TP / (img_TP + img_FP)
        if img_TP + img_FN == 0:
            img_recall = 0
        else:
            img_recall = img_TP / (img_TP + img_FN)
        for class_label in classstat.keys():
            classstat[class_label]['TP'] += class_TP.get(class_label, 0)
            classstat[class_label]['FP'] += class_FP.get(class_label, 0)
            classstat[class_label]['FN'] += class_FN.get(class_label, 0)
        if show:
            print(f"Img Stats: TP: {img_TP}, FP: {img_FP}, FN: {img_FN}, Precision: {img_precision:.2f}, Recall: {img_recall:.2f}")
        return total_TP, total_FP, total_FN, classstat

    def save_results_to_file(self, file_name, total_TP, total_FP, total_FN, classstat):
        """Save the results to a file."""
        file_name = os.path.join(self.results_save_loc, file_name + '.txt')
        with open(file_name, 'w') as f:
            total_precision = total_TP / (total_TP + total_FP)
            total_recall = total_TP / (total_TP + total_FN)
            f.write(f"{total_precision:.2f} {total_recall:.2f}\n")
            for class_label in classstat.keys():
                class_TP = classstat[class_label]['TP']
                class_FP = classstat[class_label]['FP']
                class_FN = classstat[class_label]['FN']
                if class_TP + class_FP == 0:
                    class_precision = 0
                else:
                    class_precision = class_TP / (class_TP + class_FP)
                if class_TP + class_FN == 0:
                    class_recall = 0
                else:
                    class_recall = class_TP / (class_TP + class_FN)
                f.write(f"{class_label} {class_precision:.2f} {class_recall:.2f}\n")

    def run(self):
        total_FP = total_TP = total_FN = total_precision = total_recall = 0
        num_classes = len(self.classes)
        classstat = {class_label: {'TP': 0, 'FP': 0, 'FN': 0} for class_label in range(0, num_classes)}
        for i, img_name in enumerate(self.img_list):
            if i >= self.max_no:
                break
            print(f"processing {i}/{len(self.img_list)}")
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
            predictions = self.format_predictions(pred)

            self.ground_truth_compare_predict(image_rgb, label_dir, img_name, predictions, imgsave_dir)
            #save_image_predictions(predictions, image_rgb, img_name, imgsave_dir)
            bb_larvae_count = self.bb_get_larvae_count(img_name_base, label_dir)
            a = self.count_by_prediction(predictions)
            b = self.get_label_counts(img_name_base, label_dir, show_counts=True)
            class_TP, class_FP, class_FN = self.compare_predictions_with_labels(predictions, bb_larvae_count, num_classes)
            total_TP, total_FP, total_FN, classstat = self.update_counts(total_TP, total_FP, total_FN, class_TP, class_FP, class_FN, classstat, )
            
        self.print_final_stats(total_TP, total_FP, total_FN, classstat)
        self.save_results_to_file(self.results_file_name, total_TP, total_FP, total_FN, classstat)        
        # import code
        # code.interact(local=dict(globals(), **locals()))
        print('Done')

if __name__ == '__main__':
    #weights_file = '/home/java/Java/ultralytics/runs/detect/train - alor_atem_1000/weights/best.pt'
    weights_file = '/home/java/Java/ultralytics/runs/detect/cslics_desktop_2000_cslics/weights/best.pt'
    label_dir = '/home/java/Java/data/cslics_desktop_data/labels/test'
    img_dir = '/home/java/Java/data/cslics_desktop_data/images/test'
    imgsave_dir = '/home/java/Java/data/cslics_desktop_data/test/detections/cslics_desktop_2000_cslics'
    results_file_name = "cslics_desktop_2000_cslics"
    meta_dir = '/home/java/Java/cslics'
    max_no = 60
    model = YOLO(weights_file)
    evaluator = YOLOEvaluator(weights_file, img_dir, imgsave_dir, label_dir, meta_dir, results_file_name, None, max_no)
    evaluator.run()


# ### Use yolo validation
# metrics = model.val(data='cslics_desktop.yml')  # no arguments needed, dataset and settings remembered
# metrics.box.map    # map50-95
# metrics.box.map50  # map50
# metrics.box.map75  # map75
# metrics.box.maps   # a list contains map50-95 of each category
