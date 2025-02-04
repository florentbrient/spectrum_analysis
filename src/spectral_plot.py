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
import tools as tl

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

# Compute altitude differences
dz = np.diff(z)  # Differences between consecutive altitudes
dz = np.insert(dz, 0, dz[0])  # Assume first weight equals first diff

# Initializing
nt,nkv,nz=np.shape(data.PI_E)
PI_E_sum = np.zeros((nt,nkv))
Erad_sum = np.zeros((nt,nkv))
#PI_E_sum2 = np.zeros((nt,nkv))


for idxt,tt in enumerate(time):
    PBL  = data.PBL[idxt]
    idxpbl = tl.near(z,PBL).data
    
    idxall=np.arange(0,idxpbl+30)
    weights= dz[idxall] #+1 for all (small) overshoot
    
    PI_E_sum[idxt,:]=np.average(data.PI_E[idxt,:,idxall], axis=-1, weights=weights)
    Erad_sum[idxt,:]=np.average(data.E1dr[idxt,:,idxall], axis=-1, weights=weights)
    
    # test
    #fracziall = np.arange(0,1.2,0.1)
    #idxall2   = [int(ij*idxpbl) for ij in fracziall]
    #weights   = dz[idxall2]
    #PI_E_sum2[idxt,:]=np.average(data.PI_E[idxt,:,idxall2], axis=-1, weights=weights)
    
    
    
#    plot_flux(kv2,Erad_sum,PI_E_sum,kPBL=kPBL,Euv=Euvrad_sum,\
#              y1lab='E(k)',y2lab='PiE'+Anom+'_Sum',
#              plotlines=True,namefig=namefig)
