#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 16:24:39 2023

@author: Pierre-Etienne Brilouet , fbrient
"""
import numpy as np
from matplotlib import pyplot as plt
import plot_tools


# Tool box to plot spectra

# secondary axis with lambda
def lamb2k(x):
    return 2*np.pi/x

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
    f = np.arange(0,Fs,df) 
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
    spectre_para = []
    size         = np.shape(data)
    #print('size ',size,size[0])
    NS           = size
    if len(size)==1:
        data = np.expand_dims(data, axis=0)
        NS   = np.shape(data) # new dimensions #_size doesn't change
    for p in range(NS[0]):
        #w_para = data[idz1,p,:]
        w_para = data[p,:]  #update 3D -> 2D
        f_para = fpsd(w_para,1)[0]
        spectre_para.append(fpsd(w_para,1)[2])
        
    
    SPECTRE_PARA = np.mean(spectre_para,axis=0) # one mean work
    k_para       = 2*np.pi*f_para/delta
    VAR_PARA     = np.sum(SPECTRE_PARA[0:-1]*np.diff(k_para))

    return k_para,SPECTRE_PARA,VAR_PARA


#------------------------------
# PLOT SPECTRA
#------------------------------
def plot_spectra(k_v,yall,fig_name2,yfit=None,ystd=None):
    
    # Figure Information
    colors=['Orange','Blue','Red','Purple'] 
    size_leg = 40
    size_ticks = 22
    width_spines= 2
    spines = ['left', 'bottom','top','right']
    largeur_fig = 35. # en cm
    hauteur_fig = 22. # en cm
    
    # Start figure
    fig = plt.figure(fig_name2,figsize= (0.39370*largeur_fig,0.39370*hauteur_fig),facecolor='white',dpi=None)
    
    ax = fig.add_subplot(111)
    # Mean value
    plt.plot(k_v,yall,color=colors[0],linewidth=2,label=r'$\mathbf{S - N \;direction}$')

    # pentes en -2/3
    k0 = np.linspace(1e-2,6e-2,1000)*1000.
    plt.plot(k0,3e-2*k0**(-2/3.),color='gray',linewidth=5,label=r'$\mathbf{k^{-2/3}}$')
    #plt.plot(k_v,3e-2*k_v**(-2/3.),color='gray',linewidth=5,label=r'$\mathbf{k^{-2/3}}$')


    if yfit is not None:
        plt.plot(k_v, yfit, '--',color=colors[0])
    
    # legends
    lines = plt.gca().get_lines()
    print(lines)
    include = [0]
    print('lines ',[lines[i].get_label() for i in [0]])
    
    title   = r'$\mathbf{Vertical \, velocity}$'
    legend1 = plt.legend([lines[i] for i in include],[lines[i].get_label() for i in include],
                         title=title,shadow=True,numpoints=1,loc=2,bbox_to_anchor=(1.,0.9),
                         fontsize=15,title_fontsize=20)
#    legend2 = plt.legend([lines[i] for i in [1,3]],[lines[i].get_label() for i in include],
#                         title=r'$\mathbf{{namevar}}$'.replace('namevar',namevar),shadow=True,numpoints=1,loc=2,bbox_to_anchor=(1.,0.7),
#                         fontsize=15,title_fontsize=20)
    plt.gca().add_artist(legend1)
#    plt.text(1.05,0.95,r'$\mathbf{z = {s1} \,m}$'.replace('s1', str(Z[idz1])),fontsize=20,
#             transform=ax.transAxes)
    #---
    plot_tools.adjust_spines(ax, spines,width_spines)
    xstring = r'$\mathbf{k \;(rad.km^{-1})}$'
    ystring = r'$\mathbf{k \times S_{w}\!(k)/ \sigma_{w}^2}$'
    plot_tools.legend_fig(xstring,ystring,size_leg,size_ticks)
    plt.ylim(bottom=1e-3)
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
    plt.tight_layout()
    plt.savefig(fig_name2+'.png')
    return None