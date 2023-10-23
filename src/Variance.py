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
import tools0 as tl
import time

def mkdir(path):
   try:
     os.mkdir(path)
   except:
     pass

def find_offset(var):
    offset = 1.
    if var in ('RVT','RNPM','RCT'):
        offset = 1000.
    return offset

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
    levels['SVT004']=[0,22,0.2]
    levels['SVT006']=[0,300,5]
    levels['RVT']   =[4,12,0.1]
    levels['WT']    =[-2.5,2.5,0.1]
    if var1c in levels.keys():
        tmp0  = levels[var1c]
        tmp   = np.arange(tmp0[0],tmp0[1],tmp0[2])
        if len(tmp0)>3:
            tmp = np.append(tmp0[3],tmp)
    print('levels ',tmp)
    return tmp

def plot2D(data,x,y,filesave
           ,labelx='Undef',labely='Undef'
           ,cmap='Blues_r',fts=18,size=[16.0,8.0]
           ,levels=None,RCT=None):
    
    fig   = plt.figure()
    ax    = fig.add_subplot(111)
    CS    = ax.contourf(x,y,data,cmap=cmap,levels=levels,extend='both')
    cbar  = plt.colorbar(CS)
    cbar.ax.tick_params(labelsize=fts)
    #fig.set_size_inches(15.0, 6.5) #width, height
    
    if RCT is not None:
        ax.contour(x,y,RCT,levels=[0,0.1],colors='w',linewidths=1.0,linestyles='--')

    # FIll Background for blank values
    #background_color = CS.cmap(-1)   # the last color in cmap
    #ax.set_fc(background_color)
    # FIll Background for blank values
    #background_color = CS.cmap(0)   # the last color in cmap
    #ax.set_fc(background_color)
    
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
# This code aims to calculate and draw 
# -variance
# -skewness
# -spectrum analysis
# -length scale
# from of 2D-field outputs of the MNH model (v5.5.1)


path0 = '/home/fbrient/MNH/'
#model = 'FIRE2Dreal'
model = 'IHOP2D'
file0 = 'EXP.1.VTIME.OUT.TTIME.nc'

if model == 'FIRE2Dreal':
    EXP   = 'FIR2D'
    #EXP   = 'F2DNW'
elif model == 'IHOP2D':
    EXP   = 'IHOP0'

file0 = file0.replace('EXP',EXP)
VTIME = 'V0001'
file0 = file0.replace('VTIME',VTIME)
#TTIME = 720
#file  = file.replace('TTIME',str(TTIME))
#print(file)
file0 = path0+model+'/'+file0

# Variable to study
var1c  = 'WT'
levels = findlevels(var1c)

# Time
hours = np.arange(1,840,30)

# Choices
makefigures=1
pltcloud=True
pltspectra=False
smoothspectra=1
i0=0;ilog=0

# How to find PBL top    
inv       = 'THLM'
offset    = 0.25
    
# Save figures
pathsave0="../figures/"+EXP+"/"
mkdir(pathsave0)
filesave0=pathsave0+'_'.join([var1c,EXP,VTIME])+'_TTIME'
filesptr0=pathsave0+'spectra_'+'_'.join([var1c,EXP,VTIME]) \
    +'_zALT'+'_TTIME_s'+'{0:02}'.format(smoothspectra)

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
        
        var1D=['level','ni']
        
        delta_x = ni[1]-ni[0]
        delta_z = level[1]-level[0]
        
        zi,xi = np.meshgrid(level,ni)
        
        NL      = len(level)
        NH      = len(hours)
        NHsp    = int(len(hours)/smoothspectra)
        sig2,skew  = [np.zeros((NH,NL))*np.nan for ij in range(2)]
        lambda_max = np.zeros((NHsp,NL))*np.nan
    

    time1     = time.time()
    idxzi     = tl.findpbltop(inv,fl_dia,var1D,offset=offset)
    time2     = time.time()
    print('%s function took %0.3f ms' % ("Find PBL", (time2-time1)*1000.0))
    print(hour,idxzi,level[idxzi])

    # Open variables
    var   = fl_dia[var1c][:].squeeze() #level, ni
    ofdim = find_offset(var1c)
    var   = ajax(var,of=ofdim)
    
    if makefigures:
        filesave = filesave0.replace('TTIME','{0:03}'.format(hour))
        RCT=None
        if pltcloud and (RCT in fl_dia.variables):
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
            spec_log_avg=np.zeros((NL,len(spec_log)))
        
        # Saving spectra
        spec_log_avg[ik,:]+=spec_log
        
        # Smoothing spectra
        if log_spec: # and ik==50.:
            spec_plot = spec_log_avg[ik,:] / smoothspectra
            if not np.isnan(spec_plot).all():
                #print(spec_plot)
                #print(np.argmax(spec_plot))
                #print(ilog,ik)
                lambda_max[ilog,ik] = spectra.lamb2k(k_v[np.argmax(spec_plot)])
            #if ik==30:
                #plt.plot(k_v,spec_plot)
                #plt.show()
                if pltspectra:
                    filesptr  = filesptr1.replace('ALT','{:0.3}'.format(zz))
                    spectra.plot_spectra(k_v,spec_plot,filesptr) #,fitpoly=fitpoly,ystd=ystd)
    
    # Reinit
    if log_spec:
        i0=0; ilog+=1
        spec_log_avg=np.zeros((NL,len(spec_log)))
        
    #plt.plot(lambda_max[ih,:],level)
    #plt.show()    
    del var,fl_dia


# Plot spectra
hourspectra     = hours[smoothspectra::smoothspectra]
zi,hourspectrai = np.meshgrid(level,hourspectra)
CS    = plt.contourf(hourspectrai,zi,lambda_max)
cbar  = plt.colorbar(CS)
plt.show()

zi,hoursi = np.meshgrid(level,hours)
plt.contourf(hoursi,zi,skew)
plt.show()

