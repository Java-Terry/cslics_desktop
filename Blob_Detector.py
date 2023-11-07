#!/usr/bin/env/python3

"""
quick script to do counting for cslics desktop (white background, clean images)
"""


# input image
# resize image
# colour thresholding in LAB space
# morphological operations
# input into blob mvt


import os
import glob
import cv2 as cv
import numpy as np
import machinevisiontoolbox as mvt
from machinevisiontoolbox.Image import Image as MvtImage
import matplotlib.pyplot as plt

img_dir = '/home/cslics/Data/20231103_cslics_desktop_sample_images'
save_dir = '/home/cslics/Data/20231103_cslics_desktop_sample_output2'
os.makedirs(save_dir, exist_ok=True)

img_list = sorted(glob.glob(os.path.join(img_dir, '*atenuis*.jpg')))
# print(img_list)

img_size = 4056
blur = 5

# morphological operation parameters
kernel_size_small = 3
number_iterations = 5
kernal_size_large = 11
## blob rejection criteria
area_max = 4000
area_min = 500
circ_min = 0.5
circ_max = 1.1

## contour drawing parameters
contour_colour = [0, 255, 0]
contour_thickness = 2
# TODO try thresholding out the background, and then removing it
# find contours... using mvt
MAX_COUNT = 20

# TODO histogram equalization to RGB image:
# not normally meant  to apply directly to colour (change colours)
# convert to YCbCr image
# perform HE on the intensity plane Y, then convert back to RGB



# LAB colour space:
# L - lightness, which is luminance
# a - position between maagenta and green
# b - position between yellow and blue
# red threshold: [20, 150, 150], [190, 255, 255]
# Possible yellow threshold: [20, 110, 170][255, 140, 215]
# Possible blue threshold: [20, 115, 70][255, 145, 120]
# lab_thresh_lower = np.array([100, 130, 150])
# lab_thresh_upper = np.array([200, 250, 200])


# for i, img_name in enumerate(img_list[0]):



print('reading in image, resize')

count = 0
for i, img_name in enumerate(img_list):
    if count >= MAX_COUNT:
        break
    # img_name = img_list[7]
    img_name_base =  os.path.basename(img_name).split('.')[0]

    img = cv.imread(img_name)
    print(img_name)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_height, img_width,_ = img.shape    
    img_scale_factor: float = img_size / img_width
    img_resized: np.ndarray = cv.resize(img, None, fx=img_scale_factor, fy=img_scale_factor)
    img_h_resized, img_w_resized, _ = img_resized.shape


    img_blur = cv.GaussianBlur(img_resized, (blur, blur),0)

    # # try simple blob detector
    # NOTE the simple blob detector sucks because it seems like there's very little control over what/how blobs are found, and then accessing blob info like perimeter coordinates seems not easily possible
    # params = cv.SimpleBlobDetector_Params()
    # params.filterByColor=False
    # params.filterByArea=False
    # params.filterByInertia=False
    # params.filterByCircularity=False
    # params.filterByConvexity=False
    # params.minArea=100
    # params.minCircularity=0.5
    # params.minConvexity=0.2
    # detector = cv.SimpleBlobDetector_create(params)
    # keypoints=detector.detect(img_blur)
    # blank = np.zeros((1,1))
    # blobs = cv.drawKeypoints(img_resized, keypoints, blank, (255,0,0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # nblobs = len(keypoints)
    # txt = "Num blobs: " + str(nblobs)
    # cv.putText(blobs, txt, (20, 550), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
    # plt.imshow(blobs)
    # plt.show()

    # image thresholding
    img_gray = cv.cvtColor(img_blur, cv.COLOR_RGB2GRAY)

    # NOTE: this is not used, was just performed out of interest
    # image histogram equalisation
    # img_hist = cv.equalizeHist(img_gray)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_hist = clahe.apply(img_gray)

    # img_thresh = cv.Canny(img_blur, 10, 25, L2gradient=True)

    # otsu's thresholding
    thresh, img_thresh = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

    # adaptive thresholding
    img_thresha = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

    # invert the thresholded image
    img_thresh = np.invert(img_thresh)
    img_thresha = np.invert(img_thresha)

    titles = ['Original', 'Hist', 'Global', 'Adaptive']
    images = [img_gray, img_hist, img_thresh, img_thresha]

    fig = plt.figure()
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
    plt.savefig(os.path.join(save_dir, img_name_base + 'img_steps.jpg'))

    cv.imwrite(os.path.join(save_dir, img_name_base + 'img_thresh.jpg'), img_thresh)
    cv.imwrite(os.path.join(save_dir, img_name_base + 'img_thresh_adaptive.jpg'), img_thresha)

    # # try edges
    # canny = cv.Canny(img_blur, 10, 25, L2gradient=True)
    # cv.imwrite(os.path.join(save_dir, 'edges.jpg'), canny)

    # # colour space threshold to red/yellow
    # print('color thresholding')
    # # equalize histogram
    # # convert image to CV_8UC1

    ### morphological ops
    # print('morphological operations')

    ### for otsu's global threshold:

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

    cv.imwrite(os.path.join(save_dir, img_name_base + 'morph_blobs.jpg'), img_m)

    #### for adaative thresholds:
    # ksize_morph1 = 7
    # iter = 3
    # ksize_morph2 = 5
    # kernel_small = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3, 3))

    # try inverting image to dilate the background/separate the small cells that are closely connected
    # img_thresh = np.invert(img_thresh)
    # img_m = cv.dilate(img_thresh, kernel_small, iterations=5)

    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(2, 2))
    # img_m = cv.morphologyEx(img_thresha, cv.MORPH_OPEN, kernel) # get rid of salt & pepper noise
    # img_m = cv.dilate(img_m, kernel, iterations=2)
    # img_m = cv.morphologyEx(img_m, cv.MORPH_CLOSE, kernel)
    # img_m = cv.erode(img_m, kernel, iterations=2)
    # img_thresh = np.invert(img_thresh)


    ### blob analysis time
    image_mvt = MvtImage(img_m)
    blobs = mvt.Blob(image_mvt)

    # eliminate blobs based on relevant criteria:
    # reject too-small and too weird (non-circular) blobs

    b0 = [b for b in blobs if ((b.area < area_max and b.area > area_min) and (b.circularity > circ_min and b.circularity < circ_max))]
    b0_area = [b.area for b in b0]
    b0_circ = [b.circularity for b in b0]
    b0_cent = [b.centroid for b in b0]
    # get index of blobbs that passed thresholds
    icont = [i for i, b in enumerate(blobs) if (blobs[i].centroid in b0_cent and 
                                                    blobs[i].area in b0_area and 
                                                    blobs[i].circularity in b0_circ)] 

    imblobs = blobs.drawBlobs(image_mvt, None, icont, None, contourthickness=-1)
    imblobs.write(os.path.join(save_dir, img_name_base + 'blobs.jpg'))

    # draw contours

    img_contours = cv.cvtColor(img_resized.copy(), cv.COLOR_BGR2RGB)
    for i in icont:
        cv.drawContours(img_contours,
                        blobs._contours,
                        i,
                        contour_colour,
                        thickness=contour_thickness,
                        lineType=cv.LINE_8)
    img_contours = MvtImage(img_contours)
    img_contours.write(os.path.join(save_dir, img_name_base + 'contours.jpg'))

    count += 1

print('done')
import code
code.interact(local=dict(globals(), **locals()))