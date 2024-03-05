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

weights = 'l'

