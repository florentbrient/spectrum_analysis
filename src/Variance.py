#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 17:42:14 2023

@author: fbrient
"""
import netCDF4 as nc
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as st
import os
import spectra

def mkdir(path):
   try:
     os.mkdir(path)
   except:
     pass

def ajax(axe,of=1):
    if len(axe.shape) == 1:
     axe = axe[1:-1]*of
    elif len(axe.shape) == 2:
     axe = axe[1:-1,1:-1]*of
    else:
     print('Problem ajustaxes')
    return axe

def variance(tmp):
    return np.var(tmp) #ddof =0 by default (maximum likelihood estimate of the variance )

def skewness(tmp):
    return st.skew(tmp)

def findlevels(var1c):
    levels = {}; tmp=None
    levels['SVT004']=[0,42,0.5]
    levels['SVT006']=[0,900,10]
    if var1c in levels.keys():
        tmp    = levels[var1c]
    return tmp

def plot2D(data,x,y,filesave
           ,labelx='Undef',labely='Undef'
           ,cmap='Blues_r',fts=18,size=[16.0,8.0]
           ,levels=None,RCT=None):
    
    fig   = plt.figure()
    ax    = fig.add_subplot(111)
    CS    = ax.contourf(x,y,data,cmap=cmap,levels=levels)
    cbar  = plt.colorbar(CS)
    cbar.ax.tick_params(labelsize=fts)
    #fig.set_size_inches(15.0, 6.5) #width, height
    
    if RCT is not None:
        ax.contour(x,y,RCT,levels=[0,0.1],colors='w',linewidths=1.0,linestyles='--')

    ax.set_xlabel(labelx,fontsize=fts)
    ax.set_ylabel(labely,fontsize=fts)
    plt.xticks(size=fts)
    plt.yticks(size=fts)
    fig.set_size_inches(size[0], size[1])
    fig.savefig(filesave + '.png')
    fig.savefig(filesave + '.pdf')
    plt.close()
    return fig,ax

    
    
    

#####
# This code aims to calculate/plot variance, skewness of 2D fields
# from outputs of the MNH model (v5.5.1)


path0 = '/home/fbrient/MNH/'
model = 'FIRE2Dreal'
file0 = 'EXP.1.VTIME.OUT.TTIME.nc'

EXP   = 'FIR2D'
#EXP   = 'F2DNW'
file0 = file0.replace('EXP',EXP)
VTIME = 'V0002'
file0 = file0.replace('VTIME',VTIME)
#TTIME = 720
#file  = file.replace('TTIME',str(TTIME))
#print(file)
file0 = path0+model+'/'+file0

# Variable to study
var1c  = 'RCT'
levels = findlevels(var1c)
if levels is not None:
    levels  = np.arange(levels[0],levels[1],levels[2])

# Time
hours = np.arange(1,721,1)

# Choices
makefigures=0
pltcloud=True
pltspectra=True
smoothspectra=20
i0=0

# Save figures
pathsave0="../figures/"+EXP+"/"
mkdir(pathsave0)
filesave0=pathsave0+'_'.join([var1c,EXP,VTIME])+'_TTIME'
filesptr0=pathsave0+'spectra_'+'_'.join([var1c,EXP,VTIME])+'_zALT'+'_TTIME'

for ih,hour in enumerate(hours):
    file      = file0.replace('TTIME','{0:03}'.format(hour))
    filesptr1 = filesptr0.replace('TTIME','{0:03}'.format(hour))
    print(file)
    fl_dia = nc.Dataset(file, 'r' )

    if ih==0:
        # Open dims
        ofdim = 1e-3
        ni=ajax(np.array(fl_dia['ni'][:]),of=ofdim)
        nj=ajax(np.array(fl_dia['nj'][:]),of=ofdim)
        nj_u=ajax(np.array(fl_dia['nj_u'][:]),of=ofdim)
        ni_u=ajax(np.array(fl_dia['ni_u'][:]),of=ofdim)
        ni_v=ajax(np.array(fl_dia['ni_v'][:]),of=ofdim)
        nj_u=ajax(np.array(fl_dia['nj_u'][:]),of=ofdim)
        nj_v=ajax(np.array(fl_dia['nj_v'][:]),of=ofdim)
        level=ajax(fl_dia['level'][:],of=ofdim)
        level_w=ajax(fl_dia['level_w'][:],of=ofdim)
        
        delta_x = ni[1]-ni[0]
        delta_z = level[1]-level[0]
        
        zi,xi = np.meshgrid(level,ni)
        
        sig2,skew,lambda_max = \
            [np.zeros((len(hours),len(level)))*np.nan for ij in range(3)]

    # Open variables
    var = fl_dia[var1c][:].squeeze() #level, ni
    var = ajax(var)
    
    if makefigures:
        filesave = filesave0.replace('TTIME','{0:03}'.format(hour))
        RCT=None
        if pltcloud:
            RCT = ajax(fl_dia['RCT'][:].squeeze().T,of=1000.)
        fig,ax   = plot2D(var.T,xi,zi,filesave
           ,labelx='x (km)',labely='z (km)',levels=levels,RCT=RCT)
        #plt.contourf(xi,zi,var.T)
        #plt.show()
        
    # Temporal iteration
    i0+=1
    log_spec=(i0 == smoothspectra)
    
    for ik,zz in enumerate(level):
        var1D = var[ik,:]
        # Variance
        sig2[ih,ik]  = variance(var1D)
        # Skewness
        skew[ih,ik]  = skewness(var1D)
        
        # Spectra
        k_v,SPECTRE_V,VAR_V = spectra.spectra(var1D,delta_x)
        spec_log            = k_v*SPECTRE_V/VAR_V
        if ik==0 and ih==0:
            spec_log_avg=np.zeros((len(level),len(spec_log)))
        
        # Saving spectra
        spec_log_avg[ik,:]+=spec_log
        
        # Smoothing spectra
        if log_spec and ik==50.:
            spec_plot = spec_log_avg[ik,:] / smoothspectra
            if not np.isnan(spec_plot).all():
                lambda_max[ih,ik] = spectra.lamb2k(k_v[np.argmax(spec_plot)])
            #if ik==30:
                #plt.plot(k_v,spec_plot)
                #plt.show()
                if pltspectra:
                    filesptr  = filesptr1.replace('ALT',str(zz))
                    spectra.plot_spectra(k_v,spec_plot,filesptr) #,fitpoly=fitpoly,ystd=ystd)
    
    # Reinit
    if log_spec:
        i0=0
        spec_log_avg=np.zeros((len(level),len(spec_log)))
        
    #plt.plot(lambda_max[ih,:],level)
    #plt.show()    
    del var,fl_dia


zi,hoursi = np.meshgrid(level,hours)
plt.contourf(hoursi,zi,skew)
plt.show()

