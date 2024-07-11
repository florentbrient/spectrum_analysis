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
import cloudmetrics as cm
import cm_spectral as spec


def plot_flux(k,E,PI,kPBL=None,Euv=None,\
              y1lab='xlab',y2lab='ylab',\
              namefig='namefig',\
              xsize=(10,18),fts=18,lw=2):
    
    # Start plot 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=xsize)
    ax1.loglog(k,E)
    if Euv is not None:
        ax1.loglog(k,Euv,'b--')
    if kPBL is not None:
        ax1.axvline(x=kPBL,color='k',ls='--')
    #ax1.loglog([6*kv2.max(),kv2.max()],[(6*kv2.max())**(-5/3),kv2.max()**(-5/3)],'k--')
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

def compute_uBF(U,kk=None,dx=1,dy=1):
    # Return low-pass filtered U for different k
    # Two methods
    
    
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
            
        # Filter 2 : Create the Gaussian low-pass filter
        filter_hat = np.exp(-(k**2) / (2 * k_idx**2))
        #plt.plot(filter_hat)
        U_hatf2 = U_hat * filter_hat
        Uf2 = np.fft.ifftn(U_hatf2)
        
        if idx==0: # Save
            Uf_k  = np.array(len(kk)*[np.zeros(Uf.shape)])
            Uf2_k = np.array(len(kk)*[np.zeros(Uf2.shape)])
        
        # Test Wind Averaged (FB!!)
        #Uf  -= np.mean(Uf)
        #Uf2 -= np.mean(Uf2)
        
        Uf_k[idx,:,:],Uf2_k[idx,:,:] = Uf,Uf2
        
        if pltcont:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            cs1 = ax1.contourf(U,levels=levels)
            plt.colorbar(cs1, ax=ax1)
            cs2 = ax2.contourf(Uf,levels=levels)
            #plt.tight_layout()
            plt.colorbar(cs2, ax=ax2)
            plt.show()
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            cs1 = ax1.contourf(U,levels=levels)
            plt.colorbar(cs1, ax=ax1)
            cs2 = ax2.contourf(Uf2,levels=levels)
            #plt.tight_layout()
            plt.colorbar(cs2, ax=ax2)
            plt.show()
    
    return kk,Uf_k,Uf2_k


# Path of the file
path0="/home/fbrient/GitHub/objects-LES/data/"
case ='IHOPNW';sens='IHOP0';prefix='006';vtype='V0001';nc4='nc';OUT='OUT.'
#case ='IHOP';sens='Ru0x0';prefix='004';vtype='V0301';nc4='nc4';OUT=''
#case ='FIRE';sens='Ls2x0';prefix='021';vtype='V0301';nc4='nc4';OUT=''
case ='BOMEX';sens='Ru0NW';prefix='008';vtype='V0301';nc4='nc4';OUT=''

vtyp = 'V5-7-0'
if OUT=='':
    vtyp = 'V5-5-1'

    
path    = path0+case+'/'+sens+'/'
file    = 'sel_'+sens+'.1.'+vtype+'.'+OUT+prefix+'.'+nc4
file    = path+file

# Open the netcdf file
DATA    = nc.Dataset(file,'r')

# Dimensions
var1D  = ['vertical_levels','S_N_direction','W_E_direction'] #Z,Y,X
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
dx     = nzyx['W_E_direction']
dy     = nzyx['S_N_direction']


# Find Boundary-layer height (zi)
inv               = 'THLM'
threshold         = 0.25
#idxzi,toppbl,grad = tl.findpbltop(inv,DATA,var1D,offset=threshold)
idxzi = tl.findpbltop(inv,DATA,var1D,offset=threshold)
PBLheight = ALT[idxzi]
kPBL      = 2*np.pi/PBLheight


# Vertical velocity fiels
UT = DATA['UT']
VT = DATA['VT']
WT = DATA['WT']

# Calulate U grad U
# UT*(dUT/dX + dVT/dX + dWT/dX)
# +VT*(dUT/dY + dVT/dY + dWT/dY)
# +WT*(dUT/dZ + dVT/dZ + dWT/dZ)

x=data1D['W_E_direction']
y=data1D['S_N_direction']
z=data1D['vertical_levels']

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
fraczi = 0.9

for fraczi in np.arange(0,1.2,0.1):
    idx    = int(fraczi*idxzi)
    
    # Compute U_vect
    [UT2D,VT2D,WT2D] = [ij[idx,:,:] for ij in [UT,VT,WT]]
    kv2, E_1d_rad, E_1d_azi = spec.compute_spectra(
        [UT2D,VT2D,WT2D],dx=dx,periodic_domain=True,apply_detrending=False,
        window=None,fact=0.5)
    
    kv2b, Euv_1d_rad, Euv_1d_azi = spec.compute_spectra(
        [UT2D,VT2D],dx=dx,periodic_domain=True,apply_detrending=False,
        window=None,fact=0.5)
    
    [VORTX2D,VORTY2D,VORTZ2D] = [ij[idx,:,:] for ij in [VORTX,VORTY,VORTZ]]
    kv3, Ez_1d_rad, Ez_1d_azi = spec.compute_spectra(
        [VORTX2D,VORTY2D,VORTZ2D],dx=dx,periodic_domain=True,apply_detrending=False,
        window=None,fact=0.5)
    
    
    # Energy
    # Calculate the low-pass filtered velocity field
    kk,Uf_k,Uf2_k = compute_uBF(UT2D,kk=kv2,dx=dx,dy=dy)
    kk,Vf_k,Vf2_k = compute_uBF(VT2D,kk=kv2,dx=dx,dy=dy)
    kk,Wf_k,Wf2_k = compute_uBF(WT2D,kk=kv2,dx=dx,dy=dy)
    
    # Enstrophy
    # Calculate the low-pass filtered vorticity field
    kk,VORTXf_k,VORTXf2_k = compute_uBF(VORTX2D,kk=kv3,dx=dx,dy=dy)
    kk,VORTYf_k,VORTYf2_k = compute_uBF(VORTY2D,kk=kv3,dx=dx,dy=dy)
    kk,VORTZf_k,VORTZf2_k = compute_uBF(VORTZ2D,kk=kv3,dx=dx,dy=dy)
    
    
    
    # Calculate the non-linear energy flux
    PI_E = np.zeros(len(kk))
    for idxk,k_idx in enumerate(kk):
        tmp_PI = Uf_k[idxk,:,:]*ugradu[idx,:,:]+\
                 Vf_k[idxk,:,:]*ugradv[idx,:,:]+\
                 Wf_k[idxk,:,:]*ugradw[idx,:,:]
        PI_E[idxk] = np.mean(tmp_PI)
    
    #v2
    #PI_E2 = np.zeros(len(kk))
    #for idxk,k_idx in enumerate(kk):
    #    tmp_PI = Uf2_k[idxk,:,:]*ugradu[idx,:,:]+\
    #             Vf2_k[idxk,:,:]*ugradv[idx,:,:]+\
    #             Wf2_k[idxk,:,:]*ugradw[idx,:,:]
    #    PI_E2[idxk] = np.mean(tmp_PI)    
        
    # Calculate the non-linear enstrophy flux
    PI_Z = np.zeros(len(kk))
    for idxk,k_idx in enumerate(kk):
        tmp_PI = VORTXf_k[idxk,:,:]*ugrad_vx[idx,:,:]+\
                 VORTYf_k[idxk,:,:]*ugrad_vy[idx,:,:]+\
                 VORTZf_k[idxk,:,:]*ugrad_vz[idx,:,:]
        PI_Z[idxk] = np.mean(tmp_PI)
    
    # Start plot 
    pathfig="../figures/"+vtyp+"/3D/"
    pathfig+=case+'/';tl.mkdir(pathfig)
    pathfig+=sens+'/';tl.mkdir(pathfig)
    pathfig+='Spectral_flux/';tl.mkdir(pathfig)
    
    namefig=pathfig+'XXXX_'+case+'_'+prefix+'_'+"{:.0f}".format(100*fraczi)
    
    plot_flux(kv2,E_1d_rad,PI_E,kPBL=kPBL,Euv=Euv_1d_rad,\
                  y1lab='E(k)',y2lab='Pi_E_v1',namefig=namefig)
    
    plot_flux(kv3,Ez_1d_rad,PI_Z,kPBL=kPBL,\
                  y1lab='Z(k)',y2lab='Pi_Z_v1',namefig=namefig)
    
    #plot_flux(kv2,E_1d_rad,PI_E2,kPBL=kPBL,Euv=Euv_1d_rad,\
    #              y1lab='E(k)',y2lab='Pi_E_v2',namefig=namefig)    
    
    
        
    
 


