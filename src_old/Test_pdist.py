#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 12:10:26 2024

@author: fbrient
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform
import pylab as plt
from scipy.stats.mstats import gmean


x = np.array([[0,10],[10,10],[20,20]])
squareform(pdist(x))

y = np.array([[0,10],[10,10],[20,20],[10,0]])
squareform(pdist(y)) 
aa = pdist(y)

print(aa)
print(gmean(aa))
print(np.mean(aa))
plt.plot(y[:,0],y[:,1],'o')

