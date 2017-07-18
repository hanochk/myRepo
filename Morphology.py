# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 08:17:59 2015

@author: hkremer
"""

import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn import preprocessing
import skimage



from skimage.morphology import square
broken_line = np.array([[0, 0, 0, 0, 0],
...                         [0, 0, 0, 0, 0],
...                         [1, 1, 0, 1, 1],
...                         [0, 0, 0, 0, 0],
...                         [0, 0, 0, 0, 0]], dtype=np.uint8)
skimage.morphology.closing(broken_line, square(3))