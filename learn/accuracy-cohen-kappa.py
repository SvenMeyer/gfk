#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 00:19:50 2017

@author: sumeyer
"""
import numpy as np
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix, accuracy_score
import copy


from sklearn.metrics import cohen_kappa_score
labeler2 = [1, 1, 1, 1, 1, 0, 0, 0, 0, 1]
labeler1 = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
print(cohen_kappa_score(labeler1, labeler2))

import numpy as np
a = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int)
b = copy.deepcopy(a)
print(a)
for i in range(11):
    print("cohen_kappa_score = ", cohen_kappa_score(a,b),end="")
    print(" - accuracy_score    = ", accuracy_score(a,b))
    b[i]=0