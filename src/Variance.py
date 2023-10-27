#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 17:42:14 2023

@author: fbrient
"""
import netCDF4 as nc
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
import scipy.stats as st
import os
import spectra
import tools0 as tl
import time
import plot_tools as pltl

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
    levels['SVT004']=[0,9,0.1]
    levels['SVT005']=[0,400,10]
    levels['SVT006']=[0,200,5]
    levels['RVT']   =[4,12,0.1]
    levels['WT']    =[-2.5,2.5,0.1]
    levels['DIVUV'] =[-0.04,0.04,0.005]
    if var1c in levels.keys():
        tmp0  = levels[var1c]
        tmp   = np.arange(tmp0[0],tmp0[1],tmp0[2])
        if len(tmp0)>3:
            tmp = np.append(tmp0[3],tmp)
    return tmp

def findextrema(model):
    zminmax = None
    zmax  = {'FIRE2Dreal':1.0, 'BOMEX2D':2, 'ARMCU2D':2}
    if model in zmax.keys():
        zminmax=[0,zmax[model]]
    return zminmax

def findcmap(var):
    cmap    = 'Blues_r'
    cmapall = {}
    cmapall['WT']='RdBu_r'
    cmapall['SVT004']='Reds'
    cmapall['SVT005']='Greens'
    cmapall['SVT006']='Greens'
    cmapall['DIVUV']='RdBu_r'
    if var in cmapall.keys():
        cmap=cmapall[var]
    return cmap

def plot2D(data,x,y,filesave
           ,var1c=None,title='Title'
           ,labelx='Undef',labely='Undef',zminmax=None
           ,cmap='Blues_r',fts=18,size=[18.0,12.0]
           ,levels=None,RCT=None,idx_zi=None
           ,data2=None,var2c=None,title2='Title 2'
           ,timech=None,joingraph=False):
    
    #fig   = plt.figure()
    if var2c is not None and not joingraph:
        fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
    else:
        fig, ax = plt.subplots(1)
        ax      = [ax]
        size    = [18.0,7.0]
        #ax    = fig.add_subplot(111)
    #print(ax)
    levels = findlevels(var1c)
    cmap   = findcmap(var1c)
    for ij in np.arange(len(ax)):
        if ij==1: # Caution - remove data, level
            data=data2;levels=findlevels(var2c);cmap=findcmap(var2c)
        CS    = ax[ij].contourf(x,y,data,cmap=cmap,levels=levels,extend='both')
        cbar  = plt.colorbar(CS)
        cbar.ax.tick_params(labelsize=fts)
        
        if var2 is not None and joingraph:
            # Add contour
            levels=findlevels(var2c)[::10]
            cmap_r = pltl.reverse_colourmap(cm.get_cmap(findcmap(var2c)))
            ax[ij].contour(x,y,data2,\
                           cmap=cmap_r,\
                           levels=levels,linestyles='--')
            
        #if var2 is not None:
        #    levels2=findlevels(var2c)
        #    CS2   = ax[1].contourf(x,y,data2,cmap=cmap,levels=levels2,extend='both')
        #    cbar2 = plt.colorbar(CS2)
        #    cbar2.ax.tick_params(labelsize=fts)
        #fig.set_size_inches(15.0, 6.5) #width, height
        
        if RCT is not None:
            ax[ij].contour(x,y,RCT,levels=[0.01],colors='k',linewidths=2.0,linestyles='dotted')
    
        if idx_zi is not None:
            ax[ij].axhline(y=zi[0,idx_zi],color='k',linewidth=1.0,linestyle='--')
        
        if zminmax is not None:
            ax[ij].set_ylim(zminmax)
        ax[ij].set_xlabel(labelx,fontsize=fts)
        ax[ij].set_ylabel(labely,fontsize=fts)
        #ax[ij].tick_params(axis='x', labelsize=fts)
        ax[ij].tick_params(axis='both', labelsize=fts)
        

        #plt.xticks(size=fts)
        #plt.yticks(size=fts)
    
    if timech is not None:
        plt.text(0.02,0.9,timech, fontsize=22, transform=plt.gcf().transFigure)
        
    fig.set_size_inches(size[0], size[1])
    fig.savefig(filesave + '.png')
    #fig.savefig(filesave + '.pdf')
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
model = 'FIRE2Dreal'
#model = 'IHOP2D'
#model = 'BOMEX2D'
#model = 'IHOP2DNW'
file0 = 'EXP.1.VTIME.OUT.TTIME.nc'
VTIME = 'V0001'

if model == 'FIRE2Dreal':
    EXP   = 'FIR2D'
    #EXP   = 'F2DNW'
elif model == 'IHOP2D':
    EXP   = 'IHOP0'
elif model == 'IHOP2DNW':
    EXP   = 'IHOP2'
elif model == 'BOMEX2D':
    #EXP   = 'Ru0x0'
    EXP   = 'B2DNW'
    VTIME = 'V0001'

file0 = file0.replace('EXP',EXP)
file0 = file0.replace('VTIME',VTIME)
#TTIME = 720
#file  = file.replace('TTIME',str(TTIME))
#print(file)
file0 = path0+model+'/'+file0

# Variable to study
var1c  = 'SVT004'

var2   = None
var2c  = None
var2c  = 'SVT006'

varall  = ''.join([var1c,var2c]) if var2c is not None else var1c
varall2 = varall

# Time
#hours = np.arange(1,840,1)
hours = np.arange(400,440,1)
#hours = np.arange(1,718,1)


# Choices
makefigures=1
pltcloud=True
pltspectra=False
joingraph=True
if joingraph:
    varall2=varall+'_join'

# Infos figures
zminmax = findextrema(model)

# Smoothing SpectraS
smoothspectra=60
i0=0;ilog=0

# Polytfit Spectra
fitpoly = True
xpoly   = 40
yfit    = None

# How to find PBL top    
inv       = 'THLM'
offset    = 0.25
    
# Save figures
pathsave0="../figures/"+EXP+"/"
mkdir(pathsave0)
filesave0=pathsave0+'_'.join([varall2,EXP,VTIME])+'_TTIME'
filesptr0=pathsave0+'spectra_'+'_'.join([varall,EXP,VTIME]) \
    +'_TTIME'+'_zALT_s'+'{0:02}'.format(smoothspectra)

time_start     = time.time()
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
        
        # Init
        sig2,skew  = [np.zeros((NH,NL))*np.nan for ij in range(2)]
        lambda_max = np.zeros((NHsp,NL))*np.nan
        lambda_fit = np.zeros((NHsp,NL))*np.nan
        idxzi      = np.zeros(NH,dtype=int)
    

    timech = fl_dia.variables['time']
    timech = nc.num2date(timech[:],timech.units,only_use_python_datetimes=False)
    timech = timech[0].strftime('%Y-%m-%d (%H:%M:%S)')

    time1     = time.time()
    idxzi[ih] = tl.findpbltop(inv,fl_dia,var1D,offset=offset)
    time2     = time.time()
    print('%s function took %0.3f ms' % ("Find PBL", (time2-time1)*1000.0))
    #print(hour,idxzi[ih])
    #print(level[idxzi[ih]])

    # Open variables
    var   = tl.createnew(var1c,fl_dia,var1D)
    #print(var1c,var)
    #var   = fl_dia[var1c][:].squeeze() #level, ni
    #var   = var[:].squeeze()
    ofdim = find_offset(var1c)
    var   = ajax(var,of=ofdim)
    
    # Second variable (optionnal)
    if var2c is not None:
        var2   = tl.createnew(var2c,fl_dia,var1D)
        #var2  = fl_dia[var2c][:].squeeze()
        var2  = ajax(var2,find_offset(var2c)) 
    
    if makefigures:
        filesave = filesave0.replace('TTIME','{0:03}'.format(hour))
        RCT=None
        if pltcloud and ('RCT' in fl_dia.variables):
            RCT = ajax(fl_dia['RCT'][:].squeeze().T,of=1000.)        
        fig,ax   = plot2D(var.T,xi,zi,filesave,
                          var1c=var1c,title=var1c,zminmax=zminmax,
                          labelx='x (km)',labely='z (km)',
                          RCT=RCT,idx_zi=idxzi[ih],
                          data2=var2.T,var2c=var2c,title2=var2c,
                          timech=timech,joingraph=joingraph)
        #plt.contourf(xi,zi,var.T)
        #plt.show()
        
    # Temporal iteration
    i0+=1
    log_spec=(i0 == smoothspectra)
    
    for ik,zz in enumerate(level):
        var0       = var[ik,:]
        # Variance
        sig2[ih,ik] = variance(var0)
        # Skewness
        skew[ih,ik] = skewness(var0)
        
        # Spectra
        k_v,SPECTRE_V,VAR_V = spectra.spectra(var0,delta_x)
        spec_log            = k_v*SPECTRE_V/VAR_V
        if ik==0 and ih==0:
            spec_log_avg=np.zeros((NL,len(spec_log)))
        
        # Saving spectra
        spec_log_avg[ik,:]+=spec_log
        
        # Smoothing spectra
        if log_spec:
            spec_plot = spec_log_avg[ik,:] / smoothspectra
            if not np.isnan(spec_plot).all():
                lambda_max[ilog,ik] = spectra.lamb2k(k_v[np.argmax(spec_plot)])

                if fitpoly:
                    xp   = k_v
                    p    = np.poly1d(np.polyfit(xp, spec_plot, xpoly))
                    yfit = p(xp)
                    lambda_fit[ilog,ik] = spectra.lamb2k(k_v[np.argmax(yfit)])

                if pltspectra and ik==40.:
                    filesptr  = filesptr1.replace('ALT','{:0.3}'.format(zz))
                    spectra.plot_spectra(k_v,spec_plot,filesptr,\
                                         yfit=yfit) #,fitpoly=fitpoly,ystd=ystd)
    
    # Reinit
    if log_spec:
        i0=0; ilog+=1
        spec_log_avg=np.zeros((NL,len(spec_log)))
        
    #plt.plot(lambda_max[ih,:],level)
    #plt.show()    
    del var,fl_dia

# end time
time_end     = time.time()
print('%s tooks %0.3f s' % ("*** The program", (time_end-time_start)))

### Spectra analysis
# Relative to zi
relzi           = True

# does not work for smoothspectra=1
idxsp           = np.arange(smoothspectra,NH,smoothspectra) 
hourspectra     = hours[idxsp]
PBLspectra      = level[idxzi][idxsp]

# Plot spectra
#zi,hourspectrai = np.meshgrid(level,hourspectra)
#CS    = plt.contourf(hourspectrai,zi,lambda_max)
#cbar  = plt.colorbar(CS)
#plt.show()

x,y   = hourspectra,level
data  = lambda_max.T
if fitpoly:
    data=lambda_fit.T
PBL   = PBLspectra

if relzi:
    data /= PBLspectra

spectra.plot_length(x,y,data,PBLheight=PBL)

# Variance, skewness
zi,hoursi = np.meshgrid(level,hours)
CS    = plt.contourf(hoursi,zi,skew)
cbar  = plt.colorbar(CS)
plt.show()

