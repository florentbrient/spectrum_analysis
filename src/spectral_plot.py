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
#from scipy import integrate



# Open all netcdf files
pathin   = "../data/spectra/"
prefix   = "FIR1k" #"IHOP" #"FIR1k"
filein0  = pathin+'*'+prefix+'*.nc'
data     = tl.read_netcdfs(filein0, dim='time')

# Name of var 2D: LWP or PRW
nvar = "LWP"
if prefix=="IHOP":
    nvar = "PRW"

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
PI_E_sum,PI_Z_sum,Erad_sum = [np.zeros((nt,nkv)) for ij in range(3)]
var_sum,kvmax,kvmaxLWP,kin = [np.zeros(nt) for ij in range(4)]
Ers,ErLWPs = [np.zeros((nt,nkv)) for ij in range(2)]
varLWP = "E1dr_"+nvar

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
    PI_Z_sum[idxt,:]=np.average(data.PI_Z[idxt,:,idxall], axis=-1, weights=weights)

    
    # Index of TKE
    var_sum[idxt]  = np.average(data.variance[idxt,idxall], axis=-1, weights=weights)
    
    # Data to plot
    Er = np.tile(Erad_sum[idxt,:], (1, 1))
    PI = np.tile(PI_E_sum[idxt,:], (1, 1))
    kPBL = np.tile(kPBL.values, (1, 1))
    ErLWP = np.tile(data[varLWP][idxt,:], (1, 1))
    
    # Smooth and calculate max
    Ers[idxt,:]    = tl.smooth(Er,sigma=1)
    ErLWPs[idxt,:] = tl.smooth(ErLWP,sigma=1)
    kvmax[idxt]    = kv[Ers[idxt,:].argmax()]
    kvmaxLWP[idxt] = kv[ErLWPs[idxt,:].argmax()]
    
    # Find scale of energy injection
    # Based on the gradient of PI_E
    
    #Ecum  = np.cumsum(PI[0,:])/np.sum(PI[07,:])
    #Ecum = integrate.cumulative_trapezoid(PI[0,:],kv,initial=0)/(kv-kv[0])
    #print(kv,Ecum)
    # Wrong: change by np.trapz??? np.trapzcum?
    
    #eps   = 0.02 #1%
    #kin[idxt]   = kv[np.argmax(Ecum>eps)] # First time higher than eps
    
    grad = np.gradient(PI_E_sum[idxt,:],kv)
    grad = tl.smooth(grad,sigma=2)
    # Need a condition here -> not find a good one !
    #if np.mean(grad)>0:
    kin[idxt]  = kv[np.argmax(grad)] # First time higher than eps
    print('kin ',kin[idxt], tl.z2k(kin[idxt]), idxt)
    #else:
    #    kin[idxt]  = np.nan 
    
    
    # Plot Figures for each time
    tmp = np.tile(Ers[idxt,:], (1, 1))
    namefig=pathout+'E_PI_'+prefix+'_'+"{:02}".format(tt)
    tl.plot_flux(kv,Er,PI=PI,
              kPBL=kPBL,kin=kin[idxt],smooth=tmp,\
              y1lab='E',y2lab='PiE',
              plotlines=True,namefig=namefig)
    
    # Plot LWP spectra
    tmp = np.tile(ErLWPs[idxt,:], (1, 1))
    namefig=pathout+'Er'+nvar+'_'+prefix+'_'+"{:02}".format(tt)
    tl.plot_flux(kv,ErLWP,kPBL=kPBL,smooth=tmp,\
              y1lab=r'$E_${nvar}}}$',plotlines=True,
              namefig=namefig)
        
ErLWP  = data[varLWP].data

# Comparing variance and integral spectra
#var_sp = np.trapz(ErLWP,kv)
#print(var_sp/data.var_LWP)
#plt.scatter(var_sp,data.var_LWP);plt.show()

# increasing variance (spectra (t-1) - spectra (t))
diff_ErLWP=[]
diff_ErLWP+= [ErLWPs[tt,:]-ErLWPs[tt-1,:] for tt in time[1::]]
diff_ErLWP = np.array(diff_ErLWP)



# Plot TKE Figures for all time
kPBLall = tl.z2k(data.PBL).data
namefig=pathout+'E_PI_'+prefix+'_All'
tl.plot_flux(kv,Erad_sum,PI=PI_E_sum,
          kPBL=kPBLall,
          y1lab='E',y2lab='PiE',
          plotlines=True,namefig=namefig)

namefig=pathout+'Er_'+prefix+'_All'
tl.plot_flux(kv,Erad_sum,kPBL=kPBLall,
          y1lab='E',plotlines=True,namefig=namefig)

namefig=pathout+'PI_'+prefix+'_All'
namey=r'$\PI_E$'
tl.plot_flux(kv,PI_E_sum,kPBL=kPBLall,
          y1lab='PI_E',y2lab='PiE',
          plotlines=True,namefig=namefig, logx=False)

# LWP
namefig=pathout+'Er'+nvar+'_'+prefix+'_All'
tl.plot_flux(kv,ErLWP,kPBL=kPBLall,\
          y1lab='E',plotlines=True,namefig=namefig)
    
    
# Plot increasing variance
namefig=pathout+'Er'+nvar+'_'+prefix+'_DIffTime'
tl.plot_flux(kv,diff_ErLWP,kPBL=kPBLall[1::],\
          y1lab='E(t) - E(t-1)',plotlines=True,
          namefig=namefig, logx=False)

# Plot temporal evolution of aspect ratio
namefig=pathout+'aspect_ratio_'+prefix
Gamma = {}
Gamma['TKE'] =kPBLall/kvmax #(2pi/kvmax)/(2pi/kPBL) 
Gamma[nvar]= kPBLall/kvmaxLWP #(2pi/kvmax)/(2pi/kPBL) 
LambdaEpsIn = kPBLall/kin
lambdaIn = {}
lambdaIn['LambdaEpsIn']=LambdaEpsIn
tl.plot_time(time,Gamma,
             lambdaIn=LambdaEpsIn,
             namex='Aspect Ratio (-)',
             namefig=namefig)

# Plot temporal evolution of length scale of epsilon_in
namefig=pathout+'lambdaIn_'+prefix
title=r'Relative scale of energy injection $\epsilon_{in}$'
tl.plot_time(time,lambdaIn,
             namex=r'$\lambda_{in}$ /$\lambda_{PBL}$  (-)',
             title=title,
             namefig=namefig)



# Plot Variance
namefig=pathout+'variance_'+prefix
Gamma = {}
Gamma['Var TKE'] =  var_sum
Gamma['Var '+nvar]= data['var_'+nvar]*10
lambdaIn['LambdaEpsIn']=LambdaEpsIn
tl.plot_time(time,Gamma,
             namex='Variance',
             namefig=namefig)



# Test Emma figure
var = "spectral_slope_binned_"+nvar
x = data[var]
y = np.max(ErLWPs,axis=1)

# Create scatter plot
plt.scatter(x, y, color='blue')

# Annotate each point with its corresponding number
for i, (xi, yi) in enumerate(zip(x, y), start=1):
    plt.text(xi+0.1, yi+0.1, str(i), fontsize=8, ha='center', va='center', color='red')
#             bbox=dict(facecolor='red', edgecolor='black', boxstyle='circle,pad=0.3'))

# Labels and title
plt.xlabel("SLope")
plt.ylabel("Max")
plt.title("Scatter Plot with Data Numbers")

plt.show()








