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
#import gc


def mk_pathfig(pathfig,case='',sens='',func='Spectral_flux'):
    #pathfig="../figures/"+vtyp+"/3D/"
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


def compute_gradients(u, v, w, dx, dy, dz):
    # Compute gradients in the x-direction
    du_dx = np.gradient(u, dx, axis=2)
    dv_dx = np.gradient(v, dx, axis=2)
    dw_dx = np.gradient(w, dx, axis=2)

    # Compute gradients in the y-direction
    du_dy = np.gradient(u, dy, axis=1)
    dv_dy = np.gradient(v, dy, axis=1)
    dw_dy = np.gradient(w, dy, axis=1)

    # Compute gradients in the z-direction with varying dz
    du_dz = np.zeros_like(u)
    dv_dz = np.zeros_like(v)
    dw_dz = np.zeros_like(w)
    
    for k in range(1, u.shape[0] - 1):
        du_dz[k, :, :] = (u[k + 1, :, :] - u[k - 1, :, :]) / (2.*dz[k])
        dv_dz[k, :, :] = (v[k + 1, :, :] - v[k - 1, :, :]) / (2.*dz[k])
        dw_dz[k, :, :] = (w[k + 1, :, :] - w[k - 1, :, :]) / (2.*dz[k])
    
    # Handle the boundaries
    #dz_0           = (dz[1] + dz[0])/2.
    dz_0           = dz[0]
    du_dz[0, :, :] = (u[1, :, :] - u[0, :, :]) / dz_0
    dv_dz[0, :, :] = (v[1, :, :] - v[0, :, :]) / dz_0
    dw_dz[0, :, :] = (w[1, :, :] - w[0, :, :]) / dz_0

    #dz_1           = (dz[-1] + dz[-2])/2.
    dz_1           = dz[-1]
    du_dz[-1, :, :] = (u[-1, :, :] - u[-2, :, :]) / dz_1
    dv_dz[-1, :, :] = (v[-1, :, :] - v[-2, :, :]) / dz_1
    dw_dz[-1, :, :] = (w[-1, :, :] - w[-2, :, :]) / dz_1

    return du_dx, dv_dx, dw_dx, du_dy, dv_dy, dw_dy, du_dz, dv_dz, dw_dz


def compute_uBF(U,kk=None,dx=1,dy=1,filter=0):
    # Return low-pass filtered U for different k
    # Two methods
    # filter = 0 => All
    
    
    # Create the frequency grid (2D)
    ny, nx = U.shape
    kx = np.fft.fftfreq(nx,d=1./nx)
    ky = np.fft.fftfreq(ny,d=1./ny)
    # Change to wavenumber
    kx = (2.0*np.pi)*kx/(nx*dx)
    ky = (2.0*np.pi)*ky/(ny*dy)
    
    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    
    # Mean frequency (not useful here -> no radius averaging)
    k  = np.sqrt(kx**2. + ky**2.)
    
    #print('k ',kx.min(),kx.max())
    #print('k ',k.min(),k.max())
    #print('kk ',kk.min(),kk.max())
    #plt.contourf(k);plt.colorbar();plt.show()
    #stop
    
    # Compute low-pass filtered (2D)
    print('** start U_hat **')
    U_hat = np.fft.fft2(U)
    
    # Apply the filter in the frequency domain
    #U_hatf = U_hat * filter_hat
    
    pltcont = False
    levels = np.linspace(U.min(),U.max(),10)
    
    # Create a 1D K for Pi transport
    if kk is None:
        kk = np.linspace(k.min(),k.max(),30)
    
    for idx,k_idx in enumerate(kk):
        
        # Filter 1
        U_hatf = U_hat.copy()
        #plt.imshow(np.abs(U_hatf)**2., norm=LogNorm(), aspect='auto')
        #plt.colorbar()
        #plt.show()
        U_hatf[k>k_idx] = 0.        
        #plt.imshow(np.abs(U_hatf)**2., norm=LogNorm(), aspect='auto')
        #plt.colorbar()
        #plt.show()
        
        Uf = np.fft.ifftn(U_hatf)
        if idx==0: # Save
            Uf_k  = np.array(len(kk)*[np.zeros(Uf.shape)])        
        Uf_k[idx,:,:]=Uf
        
        # Test Wind Averaged (FB!!)
        #Uf  -= np.mean(Uf)
        #Uf2 -= np.mean(Uf2)
        
        Uf2_k = None    
        if filter>0:
            # Filter 2 : Create the Gaussian low-pass filter
            filter_hat = np.exp(-(k**2) / (2 * k_idx**2))
            #plt.plot(filter_hat)
            U_hatf2 = U_hat * filter_hat
            Uf2 = np.fft.ifftn(U_hatf2)
            if idx==0: # Save
                Uf2_k = np.array(len(kk)*[np.zeros(Uf2.shape)])
            Uf2_k[idx,:,:]=Uf2
        
        if pltcont:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            cs1 = ax1.contourf(U,levels=levels)
            plt.colorbar(cs1, ax=ax1)
            cs2 = ax2.contourf(Uf,levels=levels)
            #plt.tight_layout()
            plt.colorbar(cs2, ax=ax2)
            plt.show()
               
            if filter>0:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                cs1 = ax1.contourf(U,levels=levels)
                plt.colorbar(cs1, ax=ax1)
                cs2 = ax2.contourf(Uf2,levels=levels)
                #plt.tight_layout()
                plt.colorbar(cs2, ax=ax2)
                plt.show()
    
    del Uf,U_hat,U_hatf
    #gc.collect()
    return kk,Uf_k,Uf2_k
