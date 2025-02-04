#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 14:02:30 2025

The goal of the script to analysis the saved data and plot figures

Input:
    - Netcdf data (ex: ../data/spectra/Spectra_FIR1k_V0010_001.nc)


@author: fbrient
"""

import numpy as np
import xarray as xr
from glob import glob
import pylab as plt

def read_netcdfs(files, dim):
    # glob expands paths with * to a list of files, like the unix shell
    paths = sorted(glob(files))
    datasets = [xr.open_dataset(p) for p in paths]
    combined = xr.concat(datasets, dim)
    return combined


# Open all netcdf files
pathin   = "../data/spectra/"
prefix   = "FIR1k"
filein0  = pathin+'*'+prefix+'*.nc'
data     = read_netcdfs(filein0, dim='time')

# wavelength, altitude, time and PBL height
kv       = data.kv
z        = data.z
time     = data.time
for idxt,tt in enumerate(time):
    PBL  = data.PBL[idxt]
    
    
    weights= [dz[int(fraczi*idxzi)] for fraczi in fracziall]
    PI_E_sum=np.average(PI_E, axis=0, weights=weights)
    Erad_sum=np.average(E1drad_all, axis=0, weights=weights)
    Euvrad_sum=np.average(Euv1drad_all, axis=0, weights=weights)
    print(PI_E_sum.shape)
    
    plot_flux(kv2,Erad_sum,PI_E_sum,kPBL=kPBL,Euv=Euvrad_sum,\
              y1lab='E(k)',y2lab='PiE'+Anom+'_Sum',
              plotlines=True,namefig=namefig)
