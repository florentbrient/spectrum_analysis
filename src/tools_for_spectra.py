#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 16:35:18 2024

@author: fbrient
"""

import numpy as np
from scipy.signal import welch
from collections import OrderedDict
import tools0 as tl
import pylab as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm


def pathfig(vtyp,case='',sens='',func='Spectral_flux'):
    pathfig="../figures/"+vtyp+"/3D/"
    pathfig+=case+'/';tl.mkdir(pathfig)
    pathfig+=sens+'/';tl.mkdir(pathfig)
    pathfig+=func+'/';tl.mkdir(pathfig)
    return pathfig

def dimensions(DATA,var1D):
    # Dimensions
    data1D,nzyx,sizezyx = [OrderedDict() for ij in range(3)]
    for ij in var1D:
      data1D[ij]  = DATA[ij][:] #/1000.
      nzyx[ij]    = data1D[ij][1]-data1D[ij][0]
      sizezyx[ij] = len(data1D[ij])

    #xy     = [data1D[var1D[1]],data1D[var1D[2]]] #x-axis and y-axis
    nxny   = nzyx[var1D[1]]*nzyx[var1D[2]] #km^2
    ALT    = data1D[var1D[0]]
    dz     = [0.5*(ALT[ij+1]-ALT[ij-1]) for ij in range(1,len(ALT)-1)]
    dz.insert(0,ALT[1]-ALT[0])
    dz.insert(-1,ALT[-1]-ALT[-2])
    nxnynz = np.array([nxny*ij for ij in dz]) # volume of each level
    nxnynz = np.repeat(np.repeat(nxnynz[:, np.newaxis, np.newaxis]
                                 , sizezyx[var1D[1]], axis=1), sizezyx[var1D[2]], axis=2)
    dx     = nzyx[var1D[1]]
    dy     = nzyx[var1D[2]]

    return nxnynz,data1D,dx,dy,dz


def compute_power_spectrum(u, sampling_rate,nperseg=256):
    """
    Compute the power spectrum using the Welch method.
    """
    freq, power_spectrum = welch(u, fs=sampling_rate,nperseg=nperseg) #, nperseg=len(u)//8)
    return freq, power_spectrum

def compute_structure_function(u, x, r_values):
    """
    Compute the third-order structure function S(r).
    """
    S = np.zeros_like(r_values, dtype=np.float64)
    for i, r in enumerate(r_values):
        diffs = []
        for j in range(len(u) - 1):
            idx = np.searchsorted(x, x[j] + r)  # Find index for x[j] + r
            #print(i,r,j,idx,idx < len(u))
            if idx < len(u):
                diff = u[idx] - u[j]
                diffs.append(diff**3)
        if diffs:
            S[i] = np.mean(diffs)
    return S

def plot_structure_function(freq,power_spectrum,SS,nx,\
                            r_values,PBL=None,norm=True,\
                            namefig='namefig',plotlines=False,\
                            xsize=(18,10),fts=18,lw=2):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=xsize)
    kfreq = 2*np.pi*(freq)
    ax1.loglog(kfreq, np.mean(power_spectrum,axis=0))
    # Plot -5/3 line
    if PBL is not None:
        # Wavenumber of the PBL height
        kPBL      = 2*np.pi/PBL
        
        dkf   = 2*np.pi*(freq.max()*4./nx) #frequency step : 2*fs/N
        k1scale = 3e-1
        kfreq_plot=kfreq[((kfreq>kPBL) & (kfreq<kfreq.max()-4*dkf))]
        ax1.plot(kfreq_plot,k1scale*kfreq_plot**(-5/3.),\
             color='gray',linewidth=3,linestyle='--',label=r'$\mathbf{k^{-5/3}}$')
        ax1.axvline(x=kPBL,color='k',linestyle='--')
    ax1.set_xlabel('Wavenumber [m^-1]')
    ax1.set_ylabel('Power Spectrum')
    ax1.set_title('Wind Velocity')
    
    # Structure Function as S(r)
    #k_values = 2 * np.pi / r_values
    if norm and PBL is not None:
        xplot   = r_values/PBL
        PBLplot = 1
        namex = 'r / h (-)' 
    else:
        xplot   = r_values
        PBLplot = PBL
        namex = 'r distance (m)'
    
    for ik in range(SS.shape[0]):
        ax2.plot(xplot, SS[ik,:], color='grey',alpha=0.2)
    
    ax2.plot(xplot, np.mean(SS,axis=0), marker='o')
    if PBL is not None:
        ax2.axvline(x=PBLplot,color='k',linestyle='--')    
    ax2.axhline(y=0,color='k')
    #plt.xlabel('Wavenumber k [1/m]')
    #plt.xlabel('Radius r [m]')
    ax2.set_xlabel(namex)
    ax2.set_ylabel('S_3')
    ax2.set_title('Third-Order Structure Function')
    
    namefig+='.png'
    tl.savefig2(fig, namefig)
    plt.close()
    
    
    return None