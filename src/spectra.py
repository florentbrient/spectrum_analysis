#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 16:24:39 2023

@author: Pierre-Etienne Brilouet 
Modification fbrient
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm
import plot_tools
plt.style.use('classic')
from scipy import stats, ndimage

####
# Tool box to plot spectra
#####

# Write some text on figure
def fmt(x,t='zi'):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    return rf"{s}"+t if plt.rcParams["text.usetex"] else f"{s}"+t


# secondary axis with lambda
def lamb2k(x):
    return 2*np.pi/x

def savefig(fig,pathfig='./'\
            ,namefig='namefig'\
            ,fts=16,xsize=(4,5),zmax=None\
            ,dpi=1000,quality=95,makepdf=False):
    
    plt.xticks(size=fts)
    plt.yticks(size=fts)
    fig.set_size_inches(xsize)
    print(pathfig,namefig)
    fig.savefig(pathfig+namefig+'.png')#,dpi='figure')
    if makepdf:
      fig.savefig(pathfig+namefig+'.pdf')#,quality=quality)#, dpi=dpi)
    plt.close()
    return None

def compute_energy_spectrum(fft_field,size):
    # Shift zero frequency component to the center
    fft_shifted = np.fft.fftshift(fft_field)

    # Compute wave numbers
    size = fft_field.shape[0]
    kx = np.fft.fftfreq(size, d=1.0/size)
    ky = np.fft.fftfreq(size, d=1.0/size)
    kx, ky = np.meshgrid(kx, ky)
    wave_numbers = np.sqrt(kx**2 + ky**2)

    # Compute Energy Spectrum by radial averaging
    max_wave_number = np.max(wave_numbers)
    num_bins = int(size/2)
    bins = np.linspace(0, max_wave_number, num_bins + 1)

    digitized = np.digitize(wave_numbers.flatten(), bins)
    energy_spectrum = np.histogram(wave_numbers.flatten(), bins=bins, weights=np.abs(fft_shifted.flatten())**2)[0]
    energy_spectrum /= np.histogram(wave_numbers.flatten(), bins=bins)[0]

    return bins, energy_spectrum

# Cloud metrics
# def getRad(data):
#     h = data.shape[0]
#     hc = h // 2
#     w = data.shape[1]
#     wc = w // 2

#     # create an array of integer radial distances from the center
#     Y, X = np.ogrid[0:h, 0:w]
#     r = np.hypot(X - wc, Y - hc).astype(np.int)

#     return r
# def getPsd1D(psd2D):
#     r = getRad(psd2D)
#     sh = np.min(psd2D.shape) // 2
#     N = np.ones(psd2D.shape)

#     # SUM all psd2D pixels with label 'r' for 0<=r<=wc
#     # Will miss power contributions in 'corners' r>wc
#     rran = np.arange(1, sh + 1)
#     rk = np.flip(-(np.fft.fftfreq(int(2 * sh), 1))[sh:])
#     psd1D = ndimage.sum(psd2D, r, index=rran)
#     Ns = ndimage.sum(N, r, index=rran)
#     psd1D = psd1D / Ns * rk * 2 * np.pi
#     return psd1D

def _get_rad(data):
    # From https://medium.com/tangibit-studios/2d-spectrum-characterization-e288f255cc59
    h = data.shape[0]
    hc = h // 2
    w = data.shape[1]
    wc = w // 2

    # create an array of integer radial distances from the center
    Y, X = np.ogrid[0:h, 0:w]
    r = np.hypot(X - wc, Y - hc)

    return r

def _get_psd_1d_radial(psd_2d, dx):

    # TODO: Assumes even number of points in psd_2d and square domain

    N = np.min(psd_2d.shape)
    L = int(N * dx)

    # Index radii corresponding to horizontal wavenumber 2*pi/L*r
    r = _get_rad(psd_2d)
    r_int = np.round(r).astype(int)
    rp = np.arange(1, N // 2 + 1)

    # SUM all psd_2d pixels with label 'kh' for 0<=kh<=N/2 * 2*pi*L
    # Will miss power contributions in 'corners' kh>N/2 * 2*pi*L
    # This is still a variance quantity.
    psd_1d = ndimage.sum(psd_2d, r_int, index=rp)

    # Compute prefactor that converts to spectral density and corrects for
    # annulus discreteness
    Ns = ndimage.sum(np.ones(psd_2d.shape), r_int, index=rp)
    #print('Ns',Ns)
    print('Ns',Ns.max())


    kp = 2 * np.pi / L * ndimage.sum(r, r_int, index=rp) / Ns

    psd_1d *= L**2 * kp / (2 * np.pi * N**2 * Ns)

    return psd_1d


#-----------------------------------------------
# 2D SPECTRUM FUNCTION
#-----------------------------------------------
def fpsd2D(x,Fs=1,delta=1):
    #xx = np.fft.fft2(x)
    
    size = x.shape[0]
    npix = size
    # # Perform 2D Fourier Transform
    # fft_field = np.fft.fft2(x)
    # # Compute Energy Spectrum
    # wave_numbers, energy_spectrum = compute_energy_spectrum(fft_field,size)
    # # Plot the Spectrum
    # plt.figure(figsize=(8, 6))
    # plt.loglog(wave_numbers[:-1], energy_spectrum,'b')
    # plt.xlabel('Wave Number (k)')
    # plt.ylabel('Energy Spectrum')
    # plt.title('Turbulent Spectrum with Kolmogorov Cascade')
    # plt.grid(True)
    # plt.show()
    
    #taking the fourier transform
    fourier_image = np.fft.fft2(x)
    
    #Get power spectral density
    fourier_amplitudes = np.abs(fourier_image)**2
    # New
    #fourier_amplitudes= fourier_amplitudes / np.prod(x.shape)
        
    #print(fourier_amplitudes.shape,psd_1d_rad.shape)
    #fourier_amplitudes = psd1
    print('F max ',fourier_amplitudes.max())
    
    #calculate sampling frequency fs (physical distance between pixels)
    #fs = 92e-07/npix
    fs = 1/npix

    #freq_shifted = fs/2 * np.linspace(-1,1,npix)
    #freq = fs/2 * np.linspace(0,1,int(npix/2))
    
    #constructing a wave vector array
    ## Get frequencies corresponding to signal PSD
    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2D = np.meshgrid(kfreq, kfreq)
    
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()
    
    #creating the power spectrum
    #kbins = np.arange(0.5, npix//2+1, 1.) #original
    #kbins = np.arange(0.5, npix//2, 1.) #def
    #kvals = 0.5 * (kbins[1:] + kbins[:-1]) #def

    kbins = np.arange(0, npix//2+1)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    
    #print(knrm.shape,fourier_amplitudes.shape,kbins.shape)
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                         statistic = "mean",
                                         bins = kbins)
    
    # Remove to have same output than fspd
    # But difference in psdx = Abins/(512*256). Why?
    #Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    
    f = fs*kvals
    
    #print("Plotting power spectrum of surface ...")
    #plt.figure(figsize=(10, 10))
    ####plt.loglog(fs/kvals, Abins)
    #plt.loglog(f, Abins)
    #plt.xlabel("Spatial Frequency $k$ [meters]")
    #plt.ylabel("Power per Spatial Frequency $P(k)$")
    #plt.tight_layout()
    #plt.show()
    
    #stop
    return f,Abins

def fpsd1D(x,Fs=1): 
    # Use to compare 1D PEB code and 2D new code
    # Apply on one 1D
    
    size = x.shape[0]
    npix = size
    fourier_image = np.fft.fft(x) # modif
    fourier_amplitudes = np.abs(fourier_image)**2
    fs = 1/npix
    kfreq = np.fft.fftfreq(npix) * npix
    #kfreq2D = np.meshgrid(kfreq, kfreq)
    #knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
    knrm = kfreq
    #fourier_amplitudes = fourier_amplitudes.flatten()
    
    #kbins = np.arange(0.5, npix//2, 1.) #def
    #kvals = 0.5 * (kbins[1:] + kbins[:-1]) #def

    kbins = np.arange(0, npix//2+1)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])

    #print(knrm)
    #print(kbins)
    #stop
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                         statistic = "mean",
                                         bins = kbins)
    # Remove after by myself
    #Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    
    f = fs*kvals
    #print(f.shape,Abins.shape)
    return f,Abins    

def fpsdnew(x,Fs=1,delta=1): 
    # Use to compare 1D PEB code and 2D new code
    # Apply on one 1D
    
    size = x.shape
    npix = min(size)
    if len(size)==2.:
        fourier_image = np.fft.fft2(x)
    else:
        fourier_image = np.fft.fft(x)
        
    # test
    #fourier_image = np.fft.fftshift(fourier_image)  # Shift so k0 is centred
        
    fourier_amplitudes = np.abs(fourier_image)**2
    
    # test
    #fourier_amplitudes = fourier_amplitudes / np.prod(x.shape)

    
    fs = 1/npix
    
    kfreq = np.fft.fftfreq(npix) * npix
    
    knrm = kfreq
    if len(size)==2.:
        kfreq2D = np.meshgrid(kfreq, kfreq)
        knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2).flatten()
        fourier_amplitudes = fourier_amplitudes.flatten()
    
    kbins = np.arange(0.5, npix//2, 1.) #def
    #kvals = 0.5 * (kbins[1:] + kbins[:-1]) #def

    #kbins = np.arange(1, npix//2+1)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    #kvals = kbins-0.5
    #print(kvals)
    #stop

    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                         statistic = "mean",
                                         bins = kbins)
    
    #print('Abins',Abins)
    #print('Abins',Abins.max())
    
    # Remove after by myself
    #Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    
    f = fs*kvals
    #print(f.shape,Abins.shape)
    return f,Abins    

def spectra1D(data,delta):
    f_para,SPECTRE_PARA = fpsdnew(data,delta=delta) # test new code
    k_para              = 2*np.pi*f_para/delta
    VAR_PARA            = np.sum(SPECTRE_PARA[0:-1]*np.diff(k_para))#def
    #VAR_PARA            = np.sum(SPECTRE_PARA*np.diff(k_para))
    #print(k_para.shape,SPECTRE_PARA.shape,VAR_PARA.shape)
    spec_log            = k_para*SPECTRE_PARA/VAR_PARA
    return k_para,SPECTRE_PARA,VAR_PARA,spec_log


def spectra2D(data,delta):
    f_para,SPECTRE_PARA = fpsdnew(data,delta=delta)
    k_para              = 2*np.pi*f_para/delta
    VAR_PARA            = np.sum(SPECTRE_PARA[0:-1]*np.diff(k_para)) #def
    spec_log            = k_para*SPECTRE_PARA/VAR_PARA #def
    
    # FFT from cloudmetrics
    F = np.fft.fft2(data)  # 2D FFT (no prefactor)
    F = np.fft.fftshift(F)  # Shift so k0 is centred
    psd_2d = np.abs(F) ** 2 / np.prod(data.shape)  # Energy-preserving 2D PSD
    psd_1d_rad = _get_psd_1d_radial(psd_2d, delta)
    # Wavenumbers
    N = np.min(data.shape)
    L = delta * N
    k1d = 2 * np.pi / L * np.arange(1, N // 2 + 1)
    print(psd_1d_rad.max(),spec_log.max(),spec_log.max()/psd_1d_rad.max())
    #plt.scatter(psd_1d_rad[1:],spec_log[1:],color='r')
    #plt.show()
    
    plt.plot(k1d,psd_1d_rad,'k')
    #plt.loglog(k_para,spec_log/1000.,'r')
    plt.plot(k_para,spec_log/1000.,'r')
    plt.show()
    
    return k_para,SPECTRE_PARA,VAR_PARA,spec_log


#-----------------------------------------------
# SPECTRUM FUNCTION
#-----------------------------------------------
def fpsd(x,Fs):
    # inputs: x is your fluctuation time serie
    # Fs is the sampling frequency
    N = len(x)
    tm = N/Fs
    df = 1./tm   
    #print(df)
    f = np.arange(0,Fs,df) #f_para
#calcul fft et PSD
    xx = np.fft.fft(x)
    pxx = np.real(xx*np.conj(xx)/(N**2))
    psdx = pxx
#size of psdw
    di = len(psdx)
    di2 = int(np.floor(di/2))
#fold over spectrum
    psdx = 2*psdx[1:di2]
    f = f[1:di2]
#smoothing
    fb = f[0]
    b = f[-1]/fb
    #Ns = np.ceil(6*np.log10(b))
    Ns = np.ceil(6*np.log10(b))
    dfs = pow(10,np.log10(b)/Ns)
    nfm=[]
    fs = np.array([])
    psx= np.array([])
    for ii in range(int(Ns)): 
        sect = np.where((f>=(fb*pow(dfs,ii))) & (f<(fb*pow(dfs,ii+1))))
        nfm.append(len(f[sect]))
        fs = np.append(fs,np.mean(f[sect]))
        psx = np.append(psx,np.mean(psdx[sect]))
    # outputs:
    # f and psdx : frequencies and associated power spectrum
    # fs and psx : smoothed freq and spectrum
    return [f,fs,psdx,psx]

def spectra(data,delta):
    #------------------------------
    # From VERTICAL VELOCITY SPECTRA
    # OUTPUT : 
    # - SPECTRE_W, VAR_W
    # 1D field
    #------------------------------
    #spectre_para = []
    #size         = np.shape(data)
    #print('size ',size,size[0])
    
    #NS           = size
    #if len(size)==1:
    #    data = np.expand_dims(data, axis=0)
    #    NS   = np.shape(data) # new dimensions #_size doesn't change
    #for p in range(NS[0]):
    #    #w_para = data[idz1,p,:]
    #    w_para = data[p,:]  #update 3D -> 2D
    #    f_para = fpsd(w_para,1)[0]
    #    spectre_para.append(fpsd(w_para,1)[2])
     
    f_para       = fpsd(data,1)[0]
    SPECTRE_PARA = fpsd(data,1)[2]
    
    #SPECTRE_PARA = np.mean(spectre_para,axis=0) # one mean work
    k_para       = 2*np.pi*f_para/delta
    VAR_PARA     = np.sum(SPECTRE_PARA[0:-1]*np.diff(k_para))
    
    spec_log     = k_para*SPECTRE_PARA/VAR_PARA

    return k_para,SPECTRE_PARA,VAR_PARA,spec_log

#------------------------------
# PLOT SPECTRA
#------------------------------
def plot_spectra(k_v,y1a,fig_name2\
                 ,y1afit=None,ystd=None
                 ,y1b=None,y1bfit=None
                 ,y2a=None,y2afit=None
                 ,y2b=None,y2bfit=None
                 ,infoch=None,zi=None,zmax=None
                 ,labels=[r'$\mathbf{W - E \;direction}$']):
    
    # Information for plotting
    # infoch : level (zz),
    #          name of the var 1 (namevar)
    
    # Figure Information
    colors=['Orange','Red','Blue','Purple'] 
    size_leg = 40
    size_ticks = 22
    width_spines= 2
    spines = ['left', 'bottom','top','right']
    largeur_fig = 35. # en cm
    hauteur_fig = 22. # en cm
    xsize  = (15,10)
    
    # Start figure
    fig = plt.figure(fig_name2,figsize= (0.39370*largeur_fig,0.39370*hauteur_fig),facecolor='white',dpi=None)
    
    ax = fig.add_subplot(111)
    # Mean value
    ax.plot(k_v,y1a,color=colors[0],linewidth=2,label=labels[0])
    if y1b is not None:
        ax.plot(k_v,y1b,color=colors[1],linewidth=2,label=labels[1])
        
    if y1afit is not None:
        ax.plot(k_v, y1afit, '--',color=colors[0])
    if y1bfit is not None:
        ax.plot(k_v, y1bfit, '--',color=colors[1])
    
    k0max=k_v.max()
    k0min=k0max/10
    #k0 = np.linspace(1e-2,6e-2,1000)#*1000.
    k0 = np.linspace(k0min,k0max,1000)#*1000.
    # pentes en -2/3 (k*k^-5/3=k^-2/3)
    k0scale = 1 #3e-2
    ax.plot(k0,k0scale*k0**(-2/3.),color='gray',linewidth=3,linestyle='--',label=r'$\mathbf{k^{-2/3}}$')
    # pentes en -3 (k*k^-3=k^-2)
    k0scale = 1e2 #3e-2
    ax.plot(k0,k0scale*k0**(-2),color='gray',linewidth=3,linestyle='-',label=r'$\mathbf{k^{-2}}$')

    
    #plt.plot(k_v,3e-2*k_v**(-2/3.),color='gray',linewidth=5,label=r'$\mathbf{k^{-2/3}}$')

    # legends
    lines = plt.gca().get_lines()
    print(len(lines))
    include = np.arange(0,len(lines)-2) #[0,1]
    includeslope = np.arange(len(lines)-2,len(lines))
    print('lines ',[lines[i].get_label() for i in include])
    
    namevar='var';zchar=None
    if infoch is not None:
        namevar = infoch[1]
        zchar = '{:0.2f}'.format(infoch[0])
    title   = r'$\mathbf{{namevar}}$'.replace('namevar',namevar)
    legend1 = plt.legend([lines[i] for i in include],[lines[i].get_label() for i in include],
                         title=title,shadow=True,numpoints=1,loc=2,bbox_to_anchor=(1.,0.85),
                         fontsize=15,title_fontsize=20)
    titleslope   = r'$\mathbf{Slopes}$'
    legend2 = plt.legend([lines[i] for i in includeslope],[lines[i].get_label() for i in includeslope],
                         title=titleslope,shadow=True,numpoints=1,loc=2,bbox_to_anchor=(1.,0.65),
                         fontsize=15,title_fontsize=20)
#    legend2 = plt.legend([lines[i] for i in [1,3]],[lines[i].get_label() for i in include],
#                         title=r'$\mathbf{{namevar}}$'.replace('namevar',namevar),shadow=True,numpoints=1,loc=2,bbox_to_anchor=(1.,0.7),
#                         fontsize=15,title_fontsize=20)

    plt.gca().add_artist(legend1)
    plt.gca().add_artist(legend2)

    if zi is not None:
        zichar =  '{:0.2f}'.format(zi)
        plt.text(1.05,1.05,r'$\mathbf{PBL = {s1} \,km}$'.replace('s1', zichar),
                 fontsize=20,transform=ax.transAxes)
    if zchar is not None:
        plt.text(1.05,0.95,r'$\mathbf{z = {s1} \,km}$'.replace('s1', zchar),fontsize=20,
             transform=ax.transAxes)
    if zmax is not None:
        zmaxchar =  '{:0.2f}'.format(zmax)
        plt.text(1.05,0.90,r'$\mathbf{z_{max} = {s1} \,km}$'.replace('s1', zmaxchar),
                 fontsize=20,transform=ax.transAxes)
        
    #---
    plot_tools.adjust_spines(ax, spines,width_spines)

    xstring = r'$\mathbf{k \;(rad.km^{-1})}$'
    ystring = r'$\mathbf{k \times S_{w}\!(k)/ \sigma_{w}^2}$'
    plot_tools.legend_fig(xstring,ystring,size_leg,size_ticks)
    
    plt.ylim(top=5e0)
    plt.ylim(bottom=5e-4)

    plt.yscale('log')
    plt.xscale('log')
    plt.grid(b=1, which='both', axis='both')
    #---

    secax = ax.secondary_xaxis('top', functions=(lamb2k, lamb2k))
    secax.set_xlabel(r'$\mathbf{\lambda \;(km)}$',fontsize=size_leg-10,labelpad=15)
    for label in secax.get_xticklabels():
        label.set_fontsize(size_ticks-5)
        label.set_fontweight('bold')
    secax.tick_params('both', length=8, width=2, which='major', direction='out')
    secax.tick_params('both', length=4, width=1, which='minor', direction='out')
    #---
    
    # PBL height
    if zi is not None:
        ki=2*np.pi/zi
        ax.axvline(x=ki,color="grey",linestyle='--')
    
    plt.tight_layout()
    #plt.savefig(fig_name2+'.png')
    savefig(fig,pathfig='',namefig=fig_name2\
            ,xsize=xsize)
    return None


#------------------------------
# Plot Length Scale
#------------------------------

def contour_length(x,y,data,
                pathfig='./',namefig='namefig',\
                title='title',zminmax=None,\
                PBLheight=None,relzi=False,\
                cmap0='cividis',\
                lvl=[0.,4.,0.1,1.],\
                xsize = (8,6),fts=16):
    # lvl give levels (max,min) + interval color + interval contour
    
    # Label names
    labelx='Time (hours)'
    labely='Altitude (km)'
    
    # To modified?
    levels         = np.arange(lvl[0],lvl[1],lvl[2]) #[ij for ij in range(0,4,0.1)]
    levels_contour = np.arange(lvl[0],lvl[1],lvl[3])
    
    xx, yy = np.meshgrid(x,y)
    print(xx.shape,yy.shape,data.shape)
    
    # Check negative/positive 
    cmap0 = plot_tools.inv_cmap(data,cmap0)
    
    # Useful?
    cmap   = plt.colormaps[cmap0]
    
    norm   = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    
    # Plot contourf data
    fig    = plt.figure(); ax    = fig.add_subplot()
    CS     = plt.pcolormesh(xx,yy,data,cmap=cmap,norm=norm)
    plt.colorbar(CS)
    CS2    = plt.contour(xx,yy,data,colors='k',linestyles='dashed',\
                         levels=levels_contour,linewidth = 1)
    #ax.clabel(CS2, CS2.levels, inline=True, fmt=fmt, fontsize=10)
    
    t = 'zi' if relzi else ''
    fmt0=lambda x: f'{x:.0f}'+t
    ax.clabel(CS2, CS2.levels, inline=True, fontsize=10,fmt=fmt0)
    
    
    # Plot PBL height
    if PBLheight is not None:
        ax.plot(x,PBLheight,'r--')
    
    ax.set_xlim([0,max(x)])
    if zminmax is not None:
        ax.set_ylim(zminmax)
    
    ax.set_xlabel(labelx,fontsize=fts)
    ax.set_ylabel(labely,fontsize=fts)
    
    plt.title(title)
    
    # adjust figure
    ax.locator_params(axis='x', nbins=5)
    ax.locator_params(axis='y', nbins=7)
    width_spines= 2
    plot_tools.adjust_spines(ax, ['left', 'bottom'],width_spines)
    ax.get_yaxis().set_tick_params(direction='out')
    ax.get_xaxis().set_tick_params(direction='out')
    savefig(fig,pathfig=pathfig,namefig=namefig,fts=fts,xsize=xsize)
    
    
    del fig,ax
    return 

def plot_length(x,y,data,
                pathfig='./',namefig='namefig',\
                title='title',zminmax=None,\
                PBLheight=None,relzi=False,\
                var='Var',\
                #lvl=[0.,4.,0.1,1.],\
                xsize = (5,6),fts=16):
    
    ss = data.shape
    
    # Figure Information
    colors=['Orange','Red','Blue','Purple','Black','Yellow'] 
    cmap   = 'autumn'
    #colors = plt.get_cmap(cmap)(range(ss[1]))
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, ss[1]))
    print(colors[0])
    
    
    size_leg = 14
    size_ticks = 12
    width_spines= 2
    spines = ['left', 'bottom','top','right']
    largeur_fig = 35. # en cm
    hauteur_fig = 22. # en cm
    

    fig    = plt.figure();ax = fig.add_subplot(111)
    fig.set_tight_layout(True)
    for ij in range(ss[1]):
        color = tuple(colors[ij])
        print(ij,color)
        ax.plot(data[:,ij],y,color=color)
        ax.axhline(PBLheight[ij],ls='--',color=color)
    
    axes = plt.axis()
    if (np.sign(axes[0])!=np.sign(np.nanmax(axes[1]))):
        ax.axvline(x=0, color='k', linewidth=1)
    print('axes ',axes)
    #ax.set_xlim([0,max(x)])
    if zminmax is not None:
        ax.set_ylim(zminmax)
    
    # Label names
    xstring = var
    ystring = 'Altitude (km)'
    #plot_tools.legend_fig(xstring,ystring,size_leg,size_ticks)
    ax.set_xlabel(xstring,fontsize=fts)
    ax.set_ylabel(ystring,fontsize=fts)
    
    # adjust figure
    ax.locator_params(axis='x', nbins=5)
    ax.locator_params(axis='y', nbins=7)
    plot_tools.adjust_spines(ax, ['left', 'bottom'],width_spines)
    ax.get_yaxis().set_tick_params(direction='out')
    ax.get_xaxis().set_tick_params(direction='out')

    #plot_tools.adjust_spines(ax, spines, width_spines)
    #plt.tight_layout()
    savefig(fig,pathfig=pathfig,namefig=namefig,fts=fts,xsize=xsize)

    return

