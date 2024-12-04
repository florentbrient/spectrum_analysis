#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 12:03:52 2024

Goal: Calculate and test spectra fluxes

@author: fbrient
"""

import numpy as np
import netCDF4 as nc
import tools0 as tl
from collections import OrderedDict
from copy import deepcopy
import pylab as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
from skimage import measure
import cloudmetrics as cm
import scipy.spatial.distance as sd
from scipy.stats.mstats import gmean
import cm_spectral as spec
from Test_injection_rate import test_injection_rate2D
import gc
from  test_structure_functions import *



def plot_flux(k,E,PI,kPBL=None,Euv=None,\
              y1lab='xlab',y2lab='ylab',\
              namefig='namefig',plotlines=False,\
              xsize=(10,18),fts=18,lw=2):
    
    # Start plot 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=xsize)
    ax1.loglog(k,E)
    if Euv is not None:
        ax1.loglog(k,Euv,'b--')
    if kPBL is not None:
        ax1.axvline(x=kPBL,color='k',ls='--')
    #ax1.loglog([6*kv2.max(),kv2.max()],[(6*kv2.max())**(-5/3),kv2.max()**(-5/3)],'k--')

    if plotlines:
        k0max=k.max()
        k0min=k0max/2
        k1max=k0max/2
        k1min=k0max/8
        k0 = np.linspace(k0min,k0max,1000)#*1000.
        k1 = np.linspace(k1min,k1max,1000)#*1000.
        
        # pentes en -2/3 (k*k^-5/3=k^-2/3)
        k1scale = 3e-2
        ax1.plot(k1,k1scale*k1**(-5/3.),color='gray',linewidth=3,linestyle='--',label=r'$\mathbf{k^{-5/3}}$')
        # pentes en -3 (k*k^-3=k^-2)
        k0scale = 3e-3
        ax1.plot(k0,k0scale*k0**(-3),color='gray',linewidth=3,linestyle='-',label=r'$\mathbf{k^{-3}}$')
    
        # legends
    #    lines = plt.gca().get_lines()
        #lines = ax1.get_lines()
        #print(len(lines))
        #include = np.arange(0,len(lines)-2) #[0,1]
        #includeslope = np.arange(len(lines)-2,len(lines))
        #print('lines ',[lines[i].get_label() for i in include])
    
        #legend1 = ax1.legend([lines[i] for i in include],[lines[i].get_label() for i in include],
        #                     title=None,shadow=True,numpoints=1,loc=2,bbox_to_anchor=(1.,0.85),
        #                     fontsize=15,title_fontsize=20)
        ax1.legend(title=None,shadow=True,numpoints=1,loc=2,
                             bbox_to_anchor=(0.2,0.2),
                             fontsize=12,title_fontsize=20)


    ax1.set_title('Original Signal')
    ax1.set_xlabel('Wavenumber')
    ax1.set_ylabel(y1lab)
    
    ax2.semilogx(k,PI)
    ax2.axhline(y=0,color='k')
    ax2.axvline(x=kPBL,color='k',ls='--')
    ax2.set_xlabel('Wavenumber')
    ax2.set_ylabel(y2lab)
    
    
    #pathfig=y2lab+'_'+namefig+'.png'
    namefig=namefig.replace('XXXX',y2lab)+'.png'
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
    gc.collect()
    return kk,Uf_k,Uf2_k


# Path of the file
vtyp = 'V5-5-1'
#vtyp = 'V5-7-0'


if vtyp == 'V5-5-1':
    path0="/home/fbrient/GitHub/objects-LES/data/"
    case ='IHOPNW';sens='IHOP0';prefix='006';vtype='V0001';nc4='nc';OUT='OUT.'
    #case ='IHOP';sens='trlRu0x0';prefix='004';vtype='V0301';nc4='nc4';OUT=''
    #case ='FIRE';sens='Ls2x0';prefix='024';vtype='V0301';nc4='nc4';OUT=''
    #case ='BOMEX';sens='Ru0NW';prefix='012';vtype='V0301';nc4='nc4';OUT=''
    path    = path0+case+'/'+sens+'/'
    file    = 'sel_'+sens+'.1.'+vtype+'.'+OUT+prefix+'.'+nc4
    var1D  = ['vertical_levels','S_N_direction','W_E_direction'] #Z,Y,X
elif vtyp == 'V5-7-0':
    path0="/home/fbrient/MNH/"+vtyp+"/"
    case ='FIRE3D';sens='FI1024';prefix='FIR1k'
    vtype='V0005';time='002';nc4='nc';OUT='OUT.'    
    path    = path0+case+'/'+sens+'/'
    # FIR1k.1.V0001.OUT.002.nc
    file    = prefix+'.1.'+vtype+'.'+OUT+time+'.'+nc4
    var1D  = ['level','nj','ni'] #Z,Y,X



    
file    = path+file

# Open the netcdf file
DATA    = nc.Dataset(file,'r')

# Dimensions
namez  = var1D[0]
data1D,nzyx,sizezyx = [OrderedDict() for ij in range(3)]
for ij in var1D:
  data1D[ij]  = DATA[ij][:] #/1000.
  nzyx[ij]    = data1D[ij][1]-data1D[ij][0]
  sizezyx[ij] = len(data1D[ij])

xy     = [data1D[var1D[1]],data1D[var1D[2]]] #x-axis and y-axis
nxny   = nzyx[var1D[1]]*nzyx[var1D[2]] #km^2
ALT    = data1D[namez]
dz     = [0.5*(ALT[ij+1]-ALT[ij-1]) for ij in range(1,len(ALT)-1)]
dz.insert(0,ALT[1]-ALT[0])
dz.insert(-1,ALT[-1]-ALT[-2])
nxnynz = np.array([nxny*ij for ij in dz]) # volume of each level
nxnynz = np.repeat(np.repeat(nxnynz[:, np.newaxis, np.newaxis]
                             , sizezyx[var1D[1]], axis=1), sizezyx[var1D[2]], axis=2)
dx     = nzyx[var1D[1]]
dy     = nzyx[var1D[2]]


# Find Boundary-layer height (zi)
inv               = 'THLM'
threshold         = 0.25
#idxzi,toppbl,grad = tl.findpbltop(inv,DATA,var1D,offset=threshold)
idxzi = tl.findpbltop(inv,DATA,var1D,offset=threshold)
PBLheight = ALT[idxzi]
kPBL      = 2*np.pi/PBLheight


# Vertical velocity fiels
UT = np.squeeze(DATA['UT'])
VT = np.squeeze(DATA['VT'])
WT = np.squeeze(DATA['WT'])

# Substract horizontal mean
Anom = ''
anomHor = False
if anomHor:
    UT = tl.anomcalc(UT)
    VT = tl.anomcalc(VT)
    WT = tl.anomcalc(WT)
    Anom = '_Anom'

# Calulate U grad U
# UT*(dUT/dX + dVT/dX + dWT/dX)
# +VT*(dUT/dY + dVT/dY + dWT/dY)
# +WT*(dUT/dZ + dVT/dZ + dWT/dZ)

z,y,x=[data1D[ij] for ij in var1D]

#x=data1D['W_E_direction']
#y=data1D['S_N_direction']
#z=data1D['vertical_levels']

#[dUT_dx,dUT_dy]=np.gradient(UT,x,y)
#[dVT_dx,dVT_dy]=np.gradient(VT,x,y)
#[dWT_dx,dWT_dy]=np.gradient(WT,x,y)

# Gradients
gradients = compute_gradients(UT, VT, WT, dx, dy, dz)
(du_dx, dv_dx, dw_dx, 
 du_dy, dv_dy, dw_dy, 
 du_dz, dv_dz, dw_dz) = gradients

ugradu = UT*du_dx+VT*du_dy+WT*du_dz
ugradv = UT*dv_dx+VT*dv_dy+WT*dv_dz
ugradw = UT*dw_dx+VT*dw_dy+WT*dw_dz

enstrophy=False
if enstrophy:
    ######## Enstrophy
    # Compute vorticity
    VORTX = dw_dy-dv_dz
    VORTY = du_dz-dw_dx
    VORTZ = dv_dx-du_dy
    
    gradients = compute_gradients(VORTX, VORTY, VORTZ, dx, dy, dz)
    (dvx_dx, dvy_dx, dvz_dx, 
     dvx_dy, dvy_dy, dvz_dy, 
     dvx_dz, dvy_dz, dvz_dz) = gradients
    
    ugrad_vx = UT*dvx_dx+VT*dvx_dy+WT*dvx_dz
    ugrad_vy = UT*dvy_dx+VT*dvy_dy+WT*dvy_dz
    ugrad_vz = UT*dvz_dx+VT*dvz_dy+WT*dvz_dz
    ############
        

# Z of interest (zi/2 for instance)
fracmax = 1.2
if 'BOMEX' in case:
    fracmax = 2.5

fracziall = np.arange(0,fracmax,0.1)
#fracziall = np.arange(0.4,0.6,0.1)
for iz,fraczi in enumerate(fracziall):
    print('***')
    print('Start loop for ',fraczi)
    idx    = int(fraczi*idxzi)
    
    # Compute U_vect
    [UT2D,VT2D,WT2D] = [ij[idx,:,:] for ij in [UT,VT,WT]]
    kv2, E_1d_rad, E_1d_azi = spec.compute_spectra(
        [UT2D,VT2D,WT2D],dx=dx,periodic_domain=True,apply_detrending=False,
        window=None,fact=0.5)
    
    # It doesn't work
    #test_injection_rate2D(UT2D,VT2D,dx=dx,dy=dy,k=kv2)
    #stop
    
    # Test compute structure function
    # Compute power spectrum
    sampling_rate = dx
    iline    = 120
    UT2Dline = UT2D[iline,:]
    freq, power_spectrum = compute_power_spectrum(UT2Dline, sampling_rate)
    # Compute structure function
    r_values = np.linspace(1, 10, 10)*dx  # Lag distances to evaluate S(r)
    S_r = compute_structure_function(UT2D, x, r_values)
    # Power Spectrum
    plt.subplot(1, 2, 1)
    plt.loglog(freq, power_spectrum)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power Spectrum')
    plt.title('Power Spectrum of Wind Velocity')
    
    # Structure Function as S(r)
    k_values = 2 * np.pi / r_values
    plt.subplot(1, 2, 2)
    plt.semilogx(k_values, S_r, marker='o')
    plt.xlabel('Wavenumber k [1/m]')
    plt.ylabel('S3(k)')
    plt.title('Third-Order Structure Function S3 vs k')
    
    stop
    
    
    
    variance2D = np.trapz(E_1d_rad, x=kv2)
    print('variance 2D ',variance2D)
    print('variance real ',np.var(0.5*(UT2D**2.+VT2D**2.+WT2D**2.)) )
    
    kv2b, Euv_1d_rad, Euv_1d_azi = spec.compute_spectra(
        [UT2D,VT2D],dx=dx,periodic_domain=True,apply_detrending=False,
        window=None,fact=0.5)
    if enstrophy:
        [VORTX2D,VORTY2D,VORTZ2D] = [ij[idx,:,:] for ij in [VORTX,VORTY,VORTZ]]
        kv3, Ez_1d_rad, Ez_1d_azi = spec.compute_spectra(
            [VORTX2D,VORTY2D,VORTZ2D],dx=dx,periodic_domain=True,apply_detrending=False,
            window=None,fact=0.5)
    
    
    # Energy
    # Calculate the low-pass filtered velocity field
    
    print('Start compute Low Freq U,V,W')
    kk,Uf_k,Uf2_k = compute_uBF(UT2D,kk=kv2,dx=dx,dy=dy,filter=0)
    kk,Vf_k,Vf2_k = compute_uBF(VT2D,kk=kv2,dx=dx,dy=dy,filter=0)
    kk,Wf_k,Wf2_k = compute_uBF(WT2D,kk=kv2,dx=dx,dy=dy,filter=0)
    
    if enstrophy:
        # Enstrophy
        # Calculate the low-pass filtered vorticity field
        print('Start compute Low Freq VORTICITY')
        kk,VORTXf_k,VORTXf2_k = compute_uBF(VORTX2D,kk=kv3,dx=dx,dy=dy)
        kk,VORTYf_k,VORTYf2_k = compute_uBF(VORTY2D,kk=kv3,dx=dx,dy=dy)
        kk,VORTZf_k,VORTZf2_k = compute_uBF(VORTZ2D,kk=kv3,dx=dx,dy=dy)
    
    # Initiate
    if iz == 0:
        PI_E = np.empty((len(fracziall),len(kk)))
        PI_Z = np.empty((len(fracziall),len(kk)))
        E1drad_all   = np.empty((len(fracziall),len(kk)))
        Euv1drad_all = np.empty((len(fracziall),len(kk)))
        Ez1drad_all  = np.empty((len(fracziall),len(kk)))
    
    E1drad_all[iz,:]   = E_1d_rad
    Euv1drad_all[iz,:] = Euv_1d_rad
    if enstrophy:
        Ez1drad_all[iz,:]  = Ez_1d_rad
    
    # Calculate the non-linear energy flux
    print('Start compute non-linear energy flux')
    for idxk,k_idx in enumerate(kk):
        tmp_PI = Uf_k[idxk,:,:]*ugradu[idx,:,:]+\
                 Vf_k[idxk,:,:]*ugradv[idx,:,:]+\
                 Wf_k[idxk,:,:]*ugradw[idx,:,:]
        PI_E[iz,idxk] = np.mean(tmp_PI)
    del tmp_PI
    
    #v2
    #PI_E2 = np.zeros(len(kk))
    #for idxk,k_idx in enumerate(kk):
    #    tmp_PI = Uf2_k[idxk,:,:]*ugradu[idx,:,:]+\
    #             Vf2_k[idxk,:,:]*ugradv[idx,:,:]+\
    #             Wf2_k[idxk,:,:]*ugradw[idx,:,:]
    #    PI_E2[idxk] = np.mean(tmp_PI)    
    
    if enstrophy:
        # Calculate the non-linear enstrophy flux
        print('Start compute non-linear enstrophy flux')
        for idxk,k_idx in enumerate(kk):
            tmp_PI = VORTXf_k[idxk,:,:]*ugrad_vx[idx,:,:]+\
                     VORTYf_k[idxk,:,:]*ugrad_vy[idx,:,:]+\
                     VORTZf_k[idxk,:,:]*ugrad_vz[idx,:,:]
            PI_Z[iz,idxk] = np.mean(tmp_PI)
        del tmp_PI
    
    # Start plot 
    pathfig="../figures/"+vtyp+"/3D/"
    pathfig+=case+'/';tl.mkdir(pathfig)
    pathfig+=sens+'/';tl.mkdir(pathfig)
    pathfig+='Spectral_flux/';tl.mkdir(pathfig)
    
    namefig=pathfig+'XXXX_'+case+'_'+prefix+'_'+"{:.0f}".format(100*fraczi)
    
    plot_flux(kv2,E_1d_rad,PI_E[iz,:],kPBL=kPBL,Euv=Euv_1d_rad,\
                  y1lab='E(k)',y2lab='PiE'+Anom,
                  plotlines=True,namefig=namefig)
    
    if enstrophy:
        plot_flux(kv3,Ez_1d_rad,PI_Z[iz,:],kPBL=kPBL,\
                  y1lab='Z(k)',y2lab='PiZ'+Anom,namefig=namefig)
    
    #plot_flux(kv2,E_1d_rad,PI_E2,kPBL=kPBL,Euv=Euv_1d_rad,\
    #              y1lab='E(k)',y2lab='Pi_E_v2',namefig=namefig)    


# Sum of PI_E
weights= [dz[int(fraczi*idxzi)] for fraczi in fracziall]
PI_E_sum=np.average(PI_E, axis=0, weights=weights)
Erad_sum=np.average(E1drad_all, axis=0, weights=weights)
Euvrad_sum=np.average(Euv1drad_all, axis=0, weights=weights)
print(PI_E_sum.shape)

plot_flux(kv2,Erad_sum,PI_E_sum,kPBL=kPBL,Euv=Euvrad_sum,\
          y1lab='E(k)',y2lab='PiE'+Anom+'_Sum',
          plotlines=True,namefig=namefig)


NZ = len(fracziall)
colors = plt.cm.Reds_r(np.linspace(0, 1, NZ))
fig, ax1 = plt.subplots(1, 1, figsize=(12,10))
for iz,fraczi in enumerate(fracziall):
    plt.semilogx(kv2,PI_E[iz,:],color=colors[iz])
plt.semilogx(kv2,PI_E_sum,color='b')
ax1.axhline(y=0,color='k')
ax1.set_xlabel('Wavenumber')
ax1.set_ylabel('Pi_E')
namefig=pathfig+'XXXX_'+case+'_'+prefix
namefig=namefig.replace('XXXX','PiE'+Anom+'_all')+'.png'
tl.savefig2(fig, namefig)
plt.close()  
    
 


