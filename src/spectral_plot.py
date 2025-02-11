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
import tools as tl
import pylab as plt


# Open all netcdf files
pathin   = "../data/spectra/"
prefix   = "FIR1k"
filein0  = pathin+'*'+prefix+'*.nc'
data     = tl.read_netcdfs(filein0, dim='time')

# Dir for figures
pathout = "../figures/"
pathout+= prefix+'/'
tl.mkdir(pathout)

# wavelength, altitude, time and PBL height
kv       = data.kv
z        = data.z
time     = data.time

# Compute altitude differences
dz = np.diff(z)  # Differences between consecutive altitudes
dz = np.insert(dz, 0, dz[0])  # Assume first weight equals first diff

# Initializing
nt,nkv,nz =np.shape(data.PI_E)
PI_E_sum = np.zeros((nt,nkv))
Erad_sum = np.zeros((nt,nkv))
var_sum  = np.zeros(nt)

for idxt,tt in enumerate(time):
    # Find PBL height,
    # Remove first layer, and add one layer above the PBL (overshoot)
    PBL     = data.PBL[idxt]
    kPBL    = tl.z2k(PBL) # rad/m
    idxpbl  = tl.near(z,PBL).data
    idxall  = np.arange(1,idxpbl+1) 
    weights = dz[idxall] 
    
    # Averaging between the surface and PBL
    PI_E_sum[idxt,:]=np.average(data.PI_E[idxt,:,idxall], axis=-1, weights=weights)
    Erad_sum[idxt,:]=np.average(data.E1dr[idxt,:,idxall], axis=-1, weights=weights)
    
    # Index of TKE
    var_sum[idxt]  = np.average(data.variance[idxt,idxall], axis=-1, weights=weights)
    
    # Plot Figures for each time
    namefig=pathout+'E_PI_'+prefix+'_'+"{:02}".format(tt)
    Er = np.tile(Erad_sum[idxt,:], (1, 1))
    PI = np.tile(PI_E_sum[idxt,:], (1, 1))
    kPBL = np.tile(kPBL.values, (1, 1))
    tl.plot_flux(kv,Er,PI=PI,
              kPBL=kPBL,\
              y1lab='E',y2lab='PiE',
              plotlines=True,namefig=namefig)


# Plot Figures for all time
namefig=pathout+'E_PI_'+prefix+'_All'
tl.plot_flux(kv,Erad_sum,PI=PI_E_sum,
          kPBL=tl.z2k(data.PBL).data,
          y1lab='E',y2lab='PiE',
          plotlines=True,namefig=namefig)




