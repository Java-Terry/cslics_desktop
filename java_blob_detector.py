#!/usr/bin/env/python3

"""
script to use clasical computervison to do blob detection (minimal viable product)

Day 1: 2024-02-17
    poossible to combine adaptive and otsu thresholding to get a mask that is more accurate?
        No, not really
Day 2: 20224-02-18
    more and different thresholding methods, combined with blob detection
    does seem to get a semi reasonable result
Day 3: 2024-02-19
    colour thresholding
Day 4: 2024-02-27
    combine the best of the previous days, and try to get a good result
Day 5: 2024-03-05
    functionize the code, and count detectios for evaluation comparision
"""

import os
import glob
import cv2 as cv
import numpy as np
import math
import machinevisiontoolbox as mvt
from machinevisiontoolbox.Image import Image as MvtImage
import matplotlib.pyplot as plt


MAX_COUNT = 90
img_dir = '/home/java/Java/data/cslics_desktop_data/202311_Nov_cslics_desktop_sample_images'
label_dir = '/home/java/Java/data/cslics_desktop_data/202311_Nov_cslics_desktop_sample_images/labels'
save_dir = '/home/java/Java/data/cslics_desktop_data/20311_nov_output'
os.makedirs(save_dir, exist_ok=True)

img_list = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
# felt images not possible to use for this (too much noise)
img_list_filtered = [img_path for img_path in img_list if 'fert' not in os.path.basename(img_path)]


img_size = 4056
blur = 5
# morphological operation parameters
kernel_size_small = 3
number_iterations = 7
kernal_size_large = 11
num_larg_iter = 6
## blob rejection criteria
area_min = 500 #or 100 to keep outlines
area_max = 4000
circ_min = 0.5
circ_max = 1.1
## contour drawing parameters
contour_colour = [0, 255, 0]
contour_thickness = 2
# adaptive thresholding parameters
local_area = 25 #7
mean_sub = 6 #2
#sobel
sobel_thr = 255
# Colour thresholds
lab_thresh_lower = np.array([180, 100, 100]) 
lab_thresh_upper = np.array([210, 130, 150]) 
ycrcb_thresh_lower = np.array([160, 120, 100]) 
ycrcb_thresh_upper = np.array([230, 150, 200])
hsv_thresh_lower = np.array([0, 50, 170]) 
hsv_thresh_upper = np.array([80, 255, 200]) 

###################### image processing and thresholding ######################

def prep_img(img_rgb, img_scale_factor):
    """From an RGB image, resize, blur, smooth, and increase contrast."""
    img_resized: np.ndarray = cv.resize(img_rgb, None, fx=img_scale_factor, fy=img_scale_factor)
        # blur and smooth
    img_blur = cv.GaussianBlur(img_resized, (blur, blur),0)
    bilateral_filter = cv.bilateralFilter(src=img_blur, d=10, sigmaColor=10, sigmaSpace=10)
        # increase contrast
    r, g, b = cv.split(bilateral_filter)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    b_clahe = clahe.apply(b)
    g_clahe = clahe.apply(g)
    r_clahe = clahe.apply(r)
    img_clahe = cv.merge((b_clahe, g_clahe, r_clahe))
    return img_clahe

def binary_sobel(img_rgb, blur, sobel_thr):
    """From an RGB image, apply sobel filter and threshold."""
    img_blur = cv.GaussianBlur(img_rgb, (blur, blur),0)
    sobel_x = cv.Sobel(src=img_blur, ddepth=cv.CV_64F, dx=1, dy=0, ksize=5) 
    sobel_y = cv.Sobel(src=img_blur, ddepth=cv.CV_64F, dx=0, dy=1, ksize=5)
    g_m_x = cv.magnitude(sobel_x, np.zeros_like(sobel_x))
    g_m_y = cv.magnitude(np.zeros_like(sobel_y), sobel_y)
    _, SxBI = cv.threshold(g_m_x, sobel_thr, 255, cv.THRESH_BINARY)
    _, SyBI = cv.threshold(g_m_y, sobel_thr, 255, cv.THRESH_BINARY)
    SyBI = SyBI.astype('uint8')[:, :, 0]
    SxBI = SxBI.astype('uint8')[:, :, 0]
    binary_sobel = cv.bitwise_or(SxBI, SyBI)
    return binary_sobel

def binary_canny(img_rgb, minVal, maxVal, kernal_size=3):
    """From an RGB image, apply canny filter and threshold."""
    edges = cv.Canny(img_rgb, minVal, maxVal, kernal_size)
    _, edges = cv.threshold(edges, 1, 255, cv.THRESH_BINARY)
    return edges

def adaptive_thres(img_gray, local_area, mean_sub):
    """From a grayscale image, apply adaptive thresholding."""
    img_thresha = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, local_area, mean_sub)
    img_thresha = np.invert(img_thresha)
    return img_thresha

def global_thres(img_gray):
    """From a grayscale image, apply global thresholding."""
    _, img_thresh = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    img_thresh = np.invert(img_thresh)
    return img_thresh

def rm_salt_pepper(img_gray, number_iterations, kernel_size):
    """From a grayscale image, remove salt and pepper noise."""
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(kernel_size, kernel_size))
    img_m = img_gray.copy()
    for i in range(number_iterations):
        img_m = cv.morphologyEx(img_m, cv.MORPH_OPEN, kernel) # get rid of salt & pepper nois
    return img_m

def fill_in(img_gray, number_iterations, kernel_size):
    """From a grayscale image, fill in holes."""
    kernal = cv.getStructuringElement(cv.MORPH_ELLIPSE,(kernel_size, kernel_size))
    img_m = cv.dilate(img_gray, kernal, iterations=number_iterations)
    for i in range(number_iterations):
        img_m = cv.morphologyEx(img_m, cv.MORPH_OPEN, kernal)
    img_m = cv.erode(img_m, kernal, iterations=number_iterations-1)
    return img_m

def seperate_blobs(img_binary, kernel_size_small, kernal_size_large, number_iterations):
    """From a binary image, seperate the blobs using a small kernal to erode and a large kernal to open."""
    kernel_small = cv.getStructuringElement(cv.MORPH_ELLIPSE,(kernel_size_small, kernel_size_small))
    img_m= cv.erode(img_binary, kernel_small, iterations=number_iterations-1)
    kernel_large = cv.getStructuringElement(cv.MORPH_ELLIPSE,(kernal_size_large, kernal_size_large))
    img_m2 = cv.morphologyEx(img_m, cv.MORPH_OPEN, kernel_large)
    img_m2 = np.invert(img_m2)
    return img_m2

def LAB_Tresholds(img_rgb, lab_thresh_lower, lab_thresh_upper):
    """From an RGB image, apply LAB thresholding."""
    img_lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2LAB)
    mask_lab = cv.inRange(img_lab, lab_thresh_lower, lab_thresh_upper)
    mask_lab = np.invert(mask_lab)
    return mask_lab

def YCrCb_Tresholds(img_rgb, ycrcb_thresh_lower, ycrcb_thresh_upper):
    """From an RGB image, apply YCrCb thresholding."""
    img_ycrcb = cv.cvtColor(img_rgb, cv.COLOR_RGB2YCrCb)
    mask_ycrcb = cv.inRange(img_ycrcb, ycrcb_thresh_lower, ycrcb_thresh_upper)
    mask_ycrcb = np.invert(mask_ycrcb)
    return mask_ycrcb

def HSV_Tresholds(img_rgb, hsv_thresh_lower, hsv_thresh_upper):
    """From an RGB image, apply HSV thresholding."""
    img_hsv = cv.cvtColor(img_rgb, cv.COLOR_RGB2HSV)
    mask_hsv = cv.inRange(img_hsv, hsv_thresh_lower, hsv_thresh_upper)
    return mask_hsv

def get_blobs(img_binary, circ_min, area_min, area_max, circ_max, save=False, save_dir=None, save_name=None, img_name_base=None):
    """From a binary image, get the blobs."""
    image_mvt = MvtImage(img_binary)
    try: 
        blobs = mvt.Blob(image_mvt)
        imblobs = blobs.drawBlobs(image_mvt, None, None, None, contourthickness=-1)
        #blob filtering
        b0 = [b for b in blobs if ((b.circularity > circ_min) and 
                                b.area > area_min and b.area < area_max and
                                b.circularity < circ_max)] #add to circ_min 'or not b.parent==-1' to keep outlines
        b0_area = [b.area for b in b0]
        b0_circ = [b.circularity for b in b0]
        # get index of blobbs that passed thresholds
        icont = [i for i, b in enumerate(blobs) if (blobs[i].area in b0_area and
                                                        blobs[i].circularity in b0_circ)] 
        if save:
            imblobs = blobs.drawBlobs(image_mvt, None, icont, None, contourthickness=-1)
            imblobs.write(os.path.join(save_dir, img_name_base + save_name))
        return icont, blobs
    except:
        return None, None

def draw_blob_contors(icont, blobs, img_rgb, contour_colour, contour_thickness, save_dir, save_name, img_name_base):
    img_contours = cv.cvtColor(img_rgb.copy(), cv.COLOR_RGB2BGR)
    if icont is not None:
        for i in icont:
                cv.drawContours(img_contours,
                                blobs._contours,
                                i,
                                contour_colour,
                                thickness=contour_thickness,
                                lineType=cv.LINE_8)
        img_contours = MvtImage(img_contours)
        img_contours.write(os.path.join(save_dir, img_name_base + '_' + save_name + '.jpg'))

def get_larve_counts(img_base_name, label_dir, show_counts=False):
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

def get_blob_counts(icont):
    """From a list of blobs, get the count of the blobs."""
    try:
        return len(icont)
    except:
        return 0
    
def dorian(img_name):
    """Dorians code for detection"""
    img_name_base =  os.path.basename(img_name).split('.')[0]
    img = cv.imread(img_name)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_height, img_width,_ = img.shape    
    img_scale_factor: float = img_size / img_width
    img_resized: np.ndarray = cv.resize(img, None, fx=img_scale_factor, fy=img_scale_factor)
    img_h_resized, img_w_resized, _ = img_resized.shape
    img_blur = cv.GaussianBlur(img_resized, (blur, blur),0)
    # image thresholding
    img_gray = cv.cvtColor(img_blur, cv.COLOR_RGB2GRAY)
    thresh, img_thresh = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    img_thresha = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    img_thresh = np.invert(img_thresh)
    img_thresha = np.invert(img_thresha)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(kernel_size_small, kernel_size_small))
    img_m = img_thresh.copy()
    for i in range(number_iterations):
        img_m = cv.morphologyEx(img_m, cv.MORPH_OPEN, kernel) # get rid of salt & pepper noise
    # to really try and split the connected components:
    if number_iterations-1 > 0:
        n_iter_erosion = number_iterations - 1
    else:
        n_iter_erosion = 1
    img_m = cv.erode(img_m, kernel, iterations=n_iter_erosion)
    kernel_large = cv.getStructuringElement(cv.MORPH_ELLIPSE,(kernal_size_large, kernal_size_large))
    img_m = cv.morphologyEx(img_m, cv.MORPH_OPEN, kernel_large)

    ### blob analysis time
    image_mvt = MvtImage(img_m)
    blobs = mvt.Blob(image_mvt)
    b0 = [b for b in blobs if ((b.area < area_max and b.area > area_min) and (b.circularity > circ_min and b.circularity < circ_max))]
    b0_area = [b.area for b in b0]
    b0_circ = [b.circularity for b in b0]
    b0_cent = [b.centroid for b in b0]
    # get index of blobbs that passed thresholds
    icont = [i for i, b in enumerate(blobs) if (blobs[i].centroid in b0_cent and 
                                                    blobs[i].area in b0_area and 
                                                    blobs[i].circularity in b0_circ)] 
    count = get_blob_counts(icont)
    return count


###################### actual code loops ######################

def main():
    count_list = []
    for i, img_name in enumerate(img_list):
        if i>=MAX_COUNT:
            break  
        img = cv.imread(img_name)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img_h, img_w, _ = img.shape   
        img_name_base = os.path.basename(img_name).split('.')[0]
        print(f'processing image {i}/{len(img_list)}, name: {img_name_base}')

        img_scale_factor: float = img_size / img_w
        img_prep = prep_img(img, img_scale_factor)
        img_gray = cv.cvtColor(img_prep, cv.COLOR_RGB2GRAY)
        img_h, img_w, _ = img_prep.shape   

        img_sobel = binary_sobel(img_prep, blur, sobel_thr)
        img_canny = binary_canny(img_prep, 100, 200)
        img_thresha = adaptive_thres(img_gray, local_area, mean_sub)
        img_m = rm_salt_pepper(img_thresha, number_iterations, kernel_size_small)
        img_m2 = fill_in(img_m, num_larg_iter, kernal_size_large)

        titles = ['Original', 'prepared', 'Sobel', 'Canny', "Athres_denoised", "Athres_filled"]
        images = [img, cv.cvtColor(img_prep, cv.COLOR_RGB2BGR), img_sobel, img_canny, img_m, img_m2]
        fig = plt.figure(figsize=(20, 10))
        for j in range(len(titles)):
            plt.subplot(2,math.ceil(len(titles)/2),j+1)
            plt.imshow(images[j])
            plt.title(titles[j])
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, img_name_base + '_01_steps.jpg'))

        ## Colours (day three)
        mask_lab = LAB_Tresholds(img, lab_thresh_lower, lab_thresh_upper)
        map_ycrcb = YCrCb_Tresholds(img, ycrcb_thresh_lower, ycrcb_thresh_upper)
        mask_hsv = HSV_Tresholds(img, hsv_thresh_lower, hsv_thresh_upper)
        combined_colours = cv.bitwise_or(cv.bitwise_and(mask_lab, map_ycrcb), mask_hsv)
        combined_colours_denoised = rm_salt_pepper(combined_colours, number_iterations, kernel_size_small)
        icont, blobs = get_blobs(combined_colours_denoised, circ_min, area_min, area_max, 
                                circ_max, save=True, save_dir=save_dir,  save_name='_02_color.jpg', img_name_base=img_name_base)
        blob_count1 = get_blob_counts(icont)
        draw_blob_contors(icont, blobs, img, contour_colour, contour_thickness, save_dir, '02_colour_contor', img_name_base)

        ## More thresholds (day two)
        img_combined = cv.bitwise_or(img_m, cv.bitwise_or(img_canny, img_sobel))
        img_dilute = fill_in(img_combined, number_iterations, kernel_size_small)
        #cv.imwrite(os.path.join(save_dir, img_name_base + '_03_combined.jpg'), img_dilute)
        icont, blobs = get_blobs(img_dilute, circ_min, area_min, area_max, 
                                circ_max, save=True, save_dir=save_dir,  save_name='_03_adv.jpg', img_name_base=img_name_base)
        blob_count2 = get_blob_counts(icont)
        draw_blob_contors(icont, blobs, img, contour_colour, contour_thickness, save_dir, '03_adv_contor', img_name_base)

        ## Adaptive and global thresholding (day one)
        img_global_thres = global_thres(img_gray)
        mask_original = cv.bitwise_and(img_prep, img_prep, mask=img_m2)
        mask_original = cv.bitwise_and(mask_original, mask_original, mask=img_global_thres)
        img_combined = cv.cvtColor(mask_original, cv.COLOR_RGB2GRAY)
        img_global_thres2 = global_thres(img_combined)
        img_m2 = seperate_blobs(img_global_thres2, kernel_size_small, kernal_size_large, number_iterations)
        icont, blobs = get_blobs(img_m2, circ_min, area_min, area_max, 
                                circ_max, save=True, save_dir=save_dir, save_name='_04_adv_glo.jpg', img_name_base=img_name_base)
        blob_count3 = get_blob_counts(icont)
        draw_blob_contors(icont, blobs, img, contour_colour, contour_thickness, save_dir, '04_global_contor', img_name_base)

        titles = ['Colours', 'adaptivr', 'global']
        images = [combined_colours_denoised, img_dilute, img_m2]
        fig = plt.figure(figsize=(20, 10))
        for j in range(len(titles)):
            plt.subplot(2,math.ceil(len(titles)/2),j+1)
            plt.imshow(images[j])
            plt.title(titles[j])
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, img_name_base + '_05_steps.jpg'))

        blob_count4 = dorian(img_name)

        larvae_count = get_larve_counts(img_name_base, label_dir)
        count_list.append((img_name_base, blob_count1, blob_count2, blob_count3, blob_count4, larvae_count))

        # import code
        # code.interact(local=dict(globals(), **locals()))
    print("------------   TOTAL COUNTS:   ---------------")
    colour_acc_list = []
    adv_acc_list = []
    global_acc_list = []
    dorian_acc_list = []
    for counts in count_list:
        print(f'img {counts[0]}: blob counts: {counts[1]}, {counts[2]}, {counts[3]}, {counts[4]}, larvae count: {counts[5]}')
        colour_acc_list.append(abs(counts[1]-counts[5])/counts[5])
        adv_acc_list.append(abs(counts[2]-counts[5])/counts[5])
        global_acc_list.append(abs(counts[3]-counts[5])/counts[5])
        dorian_acc_list.append(abs(counts[4]-counts[5])/counts[5])
    print('------------   Evaluation ----------------------')
    print(f'average colour accuracy: {1-(sum(colour_acc_list)/len(colour_acc_list))}')
    print(f'average advanced accuracy: {1-(sum(adv_acc_list)/len(adv_acc_list))}')
    print(f'average global accuracy: {1-(sum(global_acc_list)/len(global_acc_list))}')
    print(f'average dorian accuracy: {1-(sum(dorian_acc_list)/len(dorian_acc_list))}')

if __name__ == "__main__":
    main()
    
