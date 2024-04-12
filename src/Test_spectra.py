#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 11:40:43 2024

@author: fbrient
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm
import plot_tools
plt.style.use('classic')
from scipy import stats, ndimage
from scipy.stats import linregress

def _get_rad(data):
    # From https://medium.com/tangibit-studios/2d-spectrum-characterization-e288f255cc59
    h = data.shape[0]
    hc = h // 2  # 256
    w = data.shape[1]
    wc = w // 2

    # create an array of integer radial distances from the center
    Y, X = np.ogrid[0:h, 0:w]
    r = np.hypot(X - wc, Y - hc) #sqrt(x1**2 + x2**2)

    return r

def _get_psd_1d_radial(x, dx):
    
    F = np.fft.fft2(x)  # 2D FFT (no prefactor)
    F = np.fft.fftshift(F)  # Shift so k0 is centred
    psd_2d = np.abs(F) ** 2 / np.prod(x.shape)  # Energy-preserving 2D PSD
    
    # TODO: Assumes even number of points in psd_2d and square domain
    N = np.min(psd_2d.shape)
    L = int(N * dx)

    # Index radii corresponding to horizontal wavenumber 2*pi/L*r
    r = _get_rad(psd_2d)
    r_int = np.round(r).astype(int)
    rp = np.arange(1, N // 2 + 1)
    rp = 0.5 * (rp[1:] + rp[:-1])
    print('r',r_int,rp)

    # SUM all psd_2d pixels with label 'kh' for 0<=kh<=N/2 * 2*pi*L
    # Will miss power contributions in 'corners' kh>N/2 * 2*pi*L
    # This is still a variance quantity.
    print(psd_2d.shape)
    psd_1d = ndimage.sum(psd_2d, r_int, index=rp)

    # Compute prefactor that converts to spectral density and corrects for
    # annulus discreteness
    Ns = ndimage.sum(np.ones(psd_2d.shape), r_int, index=rp)
    print('Ns',Ns) #[8 -> 1542]
    #print(ndimage.sum(r, r_int, index=rp)/Ns)

    kp = 2 * np.pi / L * ndimage.sum(r, r_int, index=rp) / Ns
    print(kp)
    
    psd_1d_v2= psd_1d * (ndimage.sum(r, r_int, index=rp)/Ns**2)* L / (N**2)
    #psd_1d_v2= psd_1d * (ndimage.sum(r, r_int, index=rp)/Ns**2)* dx/N
    
    psd_1d *= L**2 * kp / (2 * np.pi * N**2 * Ns)
    
    model = linregress(psd_1d,psd_1d_v2)
    print('model psd1d ',model)
    
    plt.scatter(psd_1d,psd_1d_v2,color='b')
    plt.show()    
    
    #psd_1d = psd_1d_v2

    return psd_1d


def _get_psd_1d_radial_all(x, dx):
    
    N = np.min(x.shape)
    
    fourier_image = np.fft.fft2(x)  # 2D FFT (no prefactor)
    F = np.fft.fftshift(fourier_image)  # Shift so k0 is centred
    psd_2d = np.abs(F) ** 2 #/N**2 # np.prod(x.shape)  # Energy-preserving 2D PSD
    
    # TODO: Assumes even number of points in psd_2d and square domain
    #N = np.min(psd_2d.shape)
    L = int(N * dx)

    # Index radii corresponding to horizontal wavenumber 2*pi/L*r
    r = _get_rad(psd_2d)
    r_int = np.round(r).astype(int)
    rp = np.arange(1, N // 2 + 1)
    rp = 0.5 * (rp[1:] + rp[:-1])
    print('r',r_int,rp)

    # SUM all psd_2d pixels with label 'kh' for 0<=kh<=N/2 * 2*pi*L
    # Will miss power contributions in 'corners' kh>N/2 * 2*pi*L
    # This is still a variance quantity.
    psd_1d = ndimage.sum(psd_2d, r_int, index=rp)

    # Compute prefactor that converts to spectral density and corrects for
    # annulus discreteness
    Ns = ndimage.sum(np.ones(psd_2d.shape), r_int, index=rp)
    print('Ns',Ns) #[8 -> 1542]

    kp = 2 * np.pi / L * ndimage.sum(r, r_int, index=rp) / Ns    
    #psd_1d_v2= psd_1d * (ndimage.sum(r, r_int, index=rp)/Ns**2)* L / (N**2)
    psd_1d *= L**2 * kp / (2 * np.pi * N**2 * Ns)
    
    #model = linregress(psd_1d,psd_1d_v2)
    #print('model psd1d ',model)
    #plt.scatter(psd_1d,psd_1d_v2,color='b')
    #plt.show()    
    
    #size = x.shape
    #N    = min(size)
    #fourier_image = np.fft.fft2(x)
    fourier_amplitudes = np.abs(fourier_image)**2
    #plt.scatter(fourier_amplitudes,psd_2d,color='g')
    #plt.show()

    fs = 1/N
    kfreq   = np.fft.fftfreq(N) * N
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm    = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2).flatten()
    
    fourier_amplitudes = fourier_amplitudes.flatten()    
    kbins = np.arange(0.5, N//2, 1.) #def
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                         statistic = "mean",
                                         bins = kbins)
    print('knrm ',knrm,kbins)
    print('Abins ',Abins)
        
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    f     = fs*kvals
    k_para              = 2*np.pi*f/dx
    VAR_PARA            = np.sum(Abins[0:-1]*np.diff(k_para)) #def
    spec_log            = k_para*Abins/VAR_PARA #def
    print('kp ',kp,k_para)
    
    model = linregress(psd_1d,spec_log)
    print('**********',model)

    return psd_1d,spec_log

def fpsdnew(x,Fs=1,dx=1):
    size = x.shape
    N    = min(size)
    fourier_image = np.fft.fft2(x)
    fourier_amplitudes = np.abs(fourier_image)**2

    fs = 1/N
    kfreq = np.fft.fftfreq(N) * N
    knrm = kfreq
    if len(size)==2.:
        kfreq2D = np.meshgrid(kfreq, kfreq)
        knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2).flatten()
        print('fourier_amplitudes ',fourier_amplitudes)
        fourier_amplitudes = fourier_amplitudes.flatten()
        
    
    kbins = np.arange(0.5, N//2, 1.) #def
    #kvals = kbins
    kvals = 0.5 * (kbins[1:] + kbins[:-1])

    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                         statistic = "mean",
                                         bins = kbins)
    print('knrm ',knrm,kbins)
    print('Abins ',Abins)
    
    f = fs*kvals
    k_para              = 2*np.pi*f/dx
    VAR_PARA            = np.sum(Abins[0:-1]*np.diff(k_para)) #def
    spec_log            = k_para*Abins/VAR_PARA #def

    return k_para,spec_log


def spectra2D(data,delta):
    # My FFT
    k_para,spec_log = fpsdnew(data,dx=delta)
    
    # FFT from cloudmetrics
    #print('psd_2d ',psd_2d)
    psd_1d_rad = _get_psd_1d_radial(data, delta)
    
    
    psd_1d_rad2,spec_log2 = _get_psd_1d_radial_all(data, delta)
    
    plt.scatter(spec_log,spec_log2,color='k',marker='o')
    plt.show()
    plt.scatter(spec_log2,psd_1d_rad2,color='g',marker='d')
    plt.show()
    
    # Wavenumbers
    #N = np.min(data.shape)
    #L = delta * N
    #k1d = 2 * np.pi / L * np.arange(1, N // 2 + 1)
    
    
    print(psd_1d_rad.max(),spec_log.max(),spec_log.max()/psd_1d_rad.max())

    print(psd_1d_rad.shape,spec_log.shape)
    #plt.plot(k1d,psd_1d_rad,'k')
    #plt.loglog(k_para,spec_log/1000.,'r')
    #plt.plot(k_para,spec_log,'r')
    #plt.scatter(psd_1d_rad[:-1],spec_log,color='r')
    plt.scatter(psd_1d_rad,spec_log,color='r')
    plt.show()
    
    #model = linregress(psd_1d_rad[:-1],spec_log)
    #model = linregress(psd_1d_rad,spec_log)
    #print(model)
        
    return None

delta_x=0.025
Nmax=512
varsave=np.random.random([Nmax,Nmax])*2+10
spectra2D(varsave,delta_x)