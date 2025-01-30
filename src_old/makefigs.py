#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 18:54:59 2024

@author: fbrient
"""

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import tools0 as tl
import time

#@profile
def plot2D(data,x,y,filesave
           ,var1c=None,title='Title'
           ,labelx='Undef',labely='Undef'
           ,zminmax=None, Anom=0
           ,cmap='Blues_r',fts=18,size=[18.0,12.0]
           ,levels=None,RCT=None,idx_zi=None
           ,data2=None,var2c=None
           ,timech=None,joingraph=False
           ,model=None):
    

    #fig   = plt.figure()
    if var2c is not None and not joingraph:
        fig, ax = plt.subplots(2, 1, sharex=True, sharey=True,facecolor="white")
    else:
        fig, ax = plt.subplots(1,facecolor="white")
        ax      = [ax]
        size    = [18.0,8.0]
        fig.tight_layout(rect=[0.05,0.1,1.05,.92])  #def rectangle [0,0,1,1]
        #ax    = fig.add_subplot(111)
            
    #print(ax)
    sign   = True
    signature = '$\it{Florent\ Brient}$'
    
    levels = tl.findlevels(var1c,Anom=Anom,model=model)
    cmap   = tl.findcmap(var1c,Anom=Anom)
    if model is not None:
        dx=(x[1,0]-x[0,0])*1000.
        dz=(y[0,1]-y[0,0])*1000.
        textcas=tl.infocas(model,dx=dx,dz=dz)
    
    for ij in np.arange(len(ax)):
        if ij==1: # Caution - remove data, level
            data=data2
            levels=tl.findlevels(var2c,Anom=Anom,model=model)
            cmap=tl.findcmap(var2c,Anom=Anom)
        #print(x.shape,y.shape,data.shape)
        
        nmin,nmax = np.nanmin(data),np.nanmax(data)
        if levels is not None:
          nmin,nmax = np.nanmin(levels),np.nanmax(levels)
        norm = None
        if (np.sign(nmin)!=np.sign(nmax)) and nmin!=0:
          norm = mpl.colors.Normalize(vmin=nmin, vmax=abs(nmin))
        
        time1 = time.time()
        print(x.shape,y.shape,data.shape)
        CS    = ax[ij].contourf(x,y,data,cmap=cmap,levels=levels,norm=norm,extend='both')
        time2 = time.time()
        print('%s function took %0.3f ms' % ("Contourf ", (time2-time1)*1000.0))
        
        cbar  = plt.colorbar(CS)
        cbar.ax.tick_params(labelsize=fts)
        
        if var2c is not None and joingraph:
            # Add contour
            levels=tl.findlevels(var2c,Anom=Anom,model=model)
            if levels is not None:
                levels=levels[::10]
            #cmap_r = pltl.reverse_colourmap(cm.get_cmap(findcmap(var2c)))
            cmap_r = tl.findcmap(var2c,Anom=Anom)
            #print('cmap_r  ',cmap_r,levels)
            #color = 'k'
            time1 = time.time()
            ax[ij].contour(x,y,data2,\
                           #color=color,\
                           cmap=cmap_r,\
                           levels=levels,\
                           linestyles=np.where(levels >= 0, "-", "--"))
            time2 = time.time()
            print('%s function took %0.3f ms' % ("Contour ", (time2-time1)*1000.0))
                
        
        if RCT is not None:
            ax[ij].contour(x,y,RCT,levels=[0,0.01],colors='k',linewidths=2.0,linestyles='dotted')
    
        if idx_zi is not None:
            #ax[ij].axhline(y=zi[0,idx_zi],color='k',linewidth=1.0,linestyle='--')
            ax[ij].axhline(y=y[0,idx_zi],color='k',linewidth=1.0,linestyle='--')

        
        if zminmax is not None:
            ax[ij].set_ylim(zminmax)
        ax[ij].set_xlabel(labelx,fontsize=fts)
        ax[ij].set_ylabel(labely,fontsize=fts)
        #ax[ij].tick_params(axis='x', labelsize=fts)
        ax[ij].tick_params(axis='both', labelsize=fts)
        
        ax[ij].set_title(title[ij],fontsize=fts)

        #plt.xticks(size=fts)
        #plt.yticks(size=fts)
    
    if timech is not None:
        #timech=r"$\bf{"+timech+"}$"
        plt.text(0.7,0.93,timech, fontsize=22, transform=plt.gcf().transFigure\
                 ,color='red')
    if textcas is not None:
        plt.text(0.02,0.94,textcas, fontsize=18, transform=plt.gcf().transFigure)
    if sign:
        plt.text(0.85,0.05,signature, fontsize=16, transform=plt.gcf().transFigure)
        
    fig.set_size_inches(size[0], size[1])
    
    #time1 = time.time()
    #fig.savefig(filesave + '.png')
    #time2 = time.time()
    #print('%s function took %0.3f ms' % ("Savefig ", (time2-time1)*1000.0))
    
    time1 = time.time()
    tl.savefig2(fig,filesave + '.png')
    time2 = time.time()
    print('%s function took %0.3f ms' % ("Savefig 2 ", (time2-time1)*1000.0))

    #fig.savefig(filesave + '.pdf')
    plt.close()
    #gc.collect()
    return fig,ax


def plot2Dfft(f,dx,filesave='NoName',fts=20):
    
    fig, ax = plt.subplots(1,facecolor="white")
    size    = [18.0,18.0]
    fig.tight_layout(rect=[0.05,0.1,1.05,.92])  #def rectangle [0,0,1,1]
    fig.set_size_inches(size[0], size[1])
    
    # Vmin, vmax
    vmin,vmax = 1e-2,5e6

    # Compute the 2D FFT
    F = np.fft.fft2(f)
    # Shift the zero frequency component to the center
    F_shifted = np.fft.fftshift(F)
    psd_2d = np.abs(F_shifted) ** 2 #/ np.prod(f.shape)  # Energy-preserving 2D PSD

    # Calculate the frequency ranges for the x and y axes
    dy = dx
    freq_x = np.fft.fftfreq(f.shape[1], d=dx)
    freq_y = np.fft.fftfreq(f.shape[0], d=dy)
    freq_x_shifted = np.fft.fftshift(freq_x)
    freq_y_shifted = np.fft.fftshift(freq_y)

    
    #plt.contourf(k_x,k_y,psd_2d,locator=ticker.LogLocator());plt.colorbar();plt.show()
    im= plt.imshow(psd_2d, 
               extent=(freq_x_shifted[0], freq_x_shifted[-1], freq_y_shifted[0], freq_y_shifted[-1]), 
               norm=LogNorm(vmin=vmin, vmax=vmax), 
               cmap='RdBu_r',
               interpolation='none',
               aspect='auto')

    
    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=fts)  # Set colorbar tick labels size


    plt.title('Squared Magnitude of 2D Fourier Transform',fontsize=fts)
    plt.xlabel('Frequency X (k_x)',fontsize=fts)
    plt.ylabel('Frequency Y (k_y)',fontsize=fts)
    
    plt.tick_params(axis='both', which='major', labelsize=fts)  # Major ticks
    plt.tick_params(axis='both', which='minor', labelsize=fts)  # Minor ticks, if they exist

    
    tl.savefig2(fig,filesave + '.png')
    plt.close()
    
    return None

