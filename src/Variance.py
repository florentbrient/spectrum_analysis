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
           ,cmap='Blues_r',fts=15,size=[16.0,8.0]
           ,levels=None,RCT=None):
    
    fig   = plt.figure()
    ax    = fig.add_subplot(111)
    CS    = ax.contourf(x,y,data,cmap=cmap,levels=levels)
    cbar  = plt.colorbar(CS)
    cbar.ax.tick_params(labelsize=fts)
    #fig.set_size_inches(15.0, 6.5) #width, height
    
    if RCT is not None:
        ax.contour(x,y,RCT,levels=[0,0.0001],colors='w',linewidths=1.0,linestyles='--')

    ax.set_xlabel(labelx,fontsize=fts)
    ax.set_ylabel(labely,fontsize=fts)
    #plt.xticks(size=fts)
    #plt.yticks(size=fts)
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
EXP   = 'F2DNW'
file0 = file0.replace('EXP',EXP)
VTIME = 'V0001'
file0 = file0.replace('VTIME',VTIME)
#TTIME = 720
#file  = file.replace('TTIME',str(TTIME))
#print(file)
file0 = path0+model+'/'+file0

# Variable to study
var1c  = 'SVT006'
levels = findlevels(var1c)
if levels is not None:
    levels  = np.arange(levels[0],levels[1],levels[2])

# Time
hours = np.arange(1,721,1)

# Choices
makefigures=1
pltcloud=True
filesave0="../figures/"+'_'.join([var1c,EXP,VTIME])+'_TTIME'

for ih,hour in enumerate(hours):
    file  = file0.replace('TTIME','{0:03}'.format(hour))
    print(file)
    fl_dia = nc.Dataset(file, 'r' )

    if ih==0:
        # Open dims
        ni=np.array(fl_dia['ni'][:])
        nj=np.array(fl_dia['nj'][:])
        nj_u=np.array(fl_dia['nj_u'][:])
        ni_u=np.array(fl_dia['ni_u'][:])
        ni_v=np.array(fl_dia['ni_v'][:])
        nj_u=np.array(fl_dia['nj_u'][:])
        nj_v=np.array(fl_dia['nj_v'][:])
        level=fl_dia['level'][:]
        level_w=fl_dia['level_w'][:]
        zi,xi = np.meshgrid(level,ni)
        
        sig2 = np.zeros((len(hours),len(level)))*np.nan
        skew = np.zeros((len(hours),len(level)))*np.nan


    # Open variables
    var = fl_dia[var1c][:].squeeze() #level, ni
    
    if makefigures:
        filesave = filesave0.replace('TTIME','{0:03}'.format(hour))
        RCT=None
        if pltcloud:
            RCT = fl_dia['RCT'][:].squeeze().T
        fig,ax   = plot2D(var.T,xi,zi,filesave
           ,labelx='x (km)',labely='z (km)',levels=levels,RCT=RCT)
        #plt.contourf(xi,zi,var.T)
        #plt.show()
    
    for ik,zz in enumerate(level):
        var1D = var[ik,:]
        # Variance
        sig2[ih,ik]  = variance(var1D)
        # Skewness
        skew[ih,ik]  = skewness(var1D)
    del var,fl_dia


zi,hoursi = np.meshgrid(level,hours)
plt.contourf(hoursi,zi,skew)
plt.show()

