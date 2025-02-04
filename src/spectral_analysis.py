#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:16:41 2025

Spectral analysis of 3D output of Meso-NH simulations
Spectrum calculated by Janssens et al (20)
Link: https://github.com/cloudsci/cloudmetrics/ 
The goal of the script is to SAVE computation of spectrum
    
Outputs (save in NetCDF)
- 1D spectra
- Spectra flux (energy and enstrophy)
- variance
- PBL height


First start: 30 Jan 2025
@author: fbrient
"""

import numpy as np
import netCDF4 as nc
import glob
import tools as tl
import cloudmetrics as cm
import time
import xarray as xr
import sys


file = sys.argv[1] # name of the file

# Open information from 'info_run_JZ.txt'
pathinfo  = '../infos/'
fileinfo  = pathinfo+'info_run.txt'
info_dict = tl.read_info(fileinfo)
info_list = ['vtyp','machine','path','case','sens','prefix','OUT','nc4']

ivar = {}
for tmp in info_list:
    ivar[tmp] = info_dict[tmp]
    

# Path of your simulations
path    = '/'.join([ivar[ij] for ij in ['path','vtyp','case','sens']])+'/'
if 'subdir' in info_dict.keys():
    path+=info_dict['subdir']+'/'

# Path for the saved output or figures
pathfig0=info_dict['path']
if 'pathfig' in info_dict.keys():
    pathfig0=info_dict['pathfig']
pathfig0= pathfig0+ivar['vtyp']+'/'

if 'pathsave' in info_dict.keys():
    pathsave=info_dict['pathsave']

# Import the list of files
# importfiles=True
# if importfiles:
#     file0  = ivar['prefix']+'.1.'+'*'+'.'+ivar['OUT']+'*'+'.'+ivar['nc4']
#     files  = glob.glob(path+'/'+file0, recursive=True)
# else:
#     # add your files here
#     fil0   = "FIR1k.1.V0010.OUT.002.nc"
#     files  = [path+'/'+file0]
#print(path+'/'+file0,files)

# Names of vertical axis
var1D = ['level','nj','ni'] #Z,Y,X

#for file in files:
    
print('*****')
print('Start analysis of ',file)
print('*****')

# Open the netcdf file
DATA    = nc.Dataset(file,'r')

# Open dimensions
nxnynz,data1D,dx,dy,dz= tl.dimensions(DATA,var1D)
z,y,x  = [data1D[ij] for ij in var1D]

# Find Boundary-layer height (zi)
inv       = 'THLM'
threshold = 0.25
idxzi     = tl.findpbltop(inv,DATA,var1D,offset=threshold)
PBLheight = z[idxzi] # km
kPBL      = tl.z2k(PBLheight) #rad/km

# Vertical velocity fiels
UT = np.squeeze(DATA['UT'])
VT = np.squeeze(DATA['VT'])
WT = np.squeeze(DATA['WT'])

# Substract horizontal mean
anomHor = True
if anomHor:
    UT = tl.anomcalc(UT)
    VT = tl.anomcalc(VT)
    WT = tl.anomcalc(WT)
    
# Gradients
gradients = tl.compute_gradients(UT, VT, WT, dx, dy, dz)
(du_dx, dv_dx, dw_dx, 
 du_dy, dv_dy, dw_dy, 
 du_dz, dv_dz, dw_dz) = gradients

# U*grad U
ugradu = UT*du_dx+VT*du_dy+WT*du_dz
ugradv = UT*dv_dx+VT*dv_dy+WT*dv_dz
ugradw = UT*dw_dx+VT*dw_dy+WT*dw_dz    

# Calculate Enstrophy fluxes?
enstrophy=False
if enstrophy:
    ######## Enstrophy
    # Compute vorticity
    VORTX = dw_dy-dv_dz
    VORTY = du_dz-dw_dx
    VORTZ = dv_dx-du_dy
    
    gradients = tl.compute_gradients(VORTX, VORTY, VORTZ, dx, dy, dz)
    (dvx_dx, dvy_dx, dvz_dx, 
     dvx_dy, dvy_dy, dvz_dy, 
     dvx_dz, dvy_dz, dvz_dz) = gradients
    
    ugrad_vx = UT*dvx_dx+VT*dvx_dy+WT*dvx_dz
    ugrad_vy = UT*dvy_dx+VT*dvy_dy+WT*dvy_dz
    ugrad_vz = UT*dvz_dx+VT*dvz_dy+WT*dvz_dz
    ############


# Initialisation of output
nx,nz = len(x),len(z)
nkv = int(nx/2)
E1dr = np.zeros(( nkv, nz ))
E1da = np.zeros(( 72,  nz )) # 360° divided by 5° angles
PI_E = np.zeros(( nkv, nz ))
PI_Z = np.zeros(( nkv, nz ))
var  = np.zeros(nz)

# Compute spectra at each altitude
#zloop=[z[10]] # zloop=z by defaut
zloop=z
for idx,zi in enumerate(zloop):
    # Compute 2D TKE
    [UT2D,VT2D,WT2D] = [ij[idx,:,:] for ij in [UT,VT,WT]]
    TKE2D  = pow(tl.anomcalc(UT2D),2.)\
            +pow(tl.anomcalc(VT2D),2.)\
            +pow(tl.anomcalc(WT2D),2.)
    
    kv, E1dr[:,idx], E1da[:,idx] = cm.scalar.compute_spectra(
        TKE2D,dx=dx,periodic_domain=True,apply_detrending=False,
        window=None)

    var[idx]=np.var(TKE2D)        
    # Verify if the variance between the spectra and the field is not too different
    #tl.checkvariance(kv,E_1d_rad,TKE2D)
    
    #######################################
    # COmputation of spectral energy flux
    #######################################
    
    # Calculate the low-pass filtered velocity field
    print('Start compute Low Freq U,V,W')
    time1 = time.time()
    kk,Uf_k = tl.compute_uBF(UT2D,kk=kv,dx=dx,dy=dy)
    kk,Vf_k = tl.compute_uBF(VT2D,kk=kv,dx=dx,dy=dy)
    kk,Wf_k = tl.compute_uBF(WT2D,kk=kv,dx=dx,dy=dy)
    time2 = time.time()
    print('%s function took %0.3f s' % ("Spectra Energy", (time2-time1)))
    
    # Calculate the non-linear energy flux
    print('Start compute non-linear energy flux')
    for idxk,k_idx in enumerate(kk):
        tmp_PI = Uf_k[idxk,:,:]*ugradu[idx,:,:]+\
                 Vf_k[idxk,:,:]*ugradv[idx,:,:]+\
                 Wf_k[idxk,:,:]*ugradw[idx,:,:]
        PI_E[idxk,idx] = np.mean(tmp_PI)
    del tmp_PI
    
    if enstrophy:
        # Enstrophy
        [VORTX2D,VORTY2D,VORTZ2D] = [ij[idx,:,:] for ij in [VORTX,VORTY,VORTZ]]
        # Calculate the low-pass filtered vorticity field
        print('Start compute Low Freq VORTICITY')
        kk,VORTXf_k = tl.compute_uBF(VORTX2D,kk=kv,dx=dx,dy=dy)
        kk,VORTYf_k = tl.compute_uBF(VORTY2D,kk=kv,dx=dx,dy=dy)
        kk,VORTZf_k = tl.compute_uBF(VORTZ2D,kk=kv,dx=dx,dy=dy)
        # Calculate the non-linear enstrophy flux
        print('Start compute non-linear enstrophy flux')
        for idxk,k_idx in enumerate(kk):
            tmp_PI = VORTXf_k[idxk,:,:]*ugrad_vx[idx,:,:]+\
                     VORTYf_k[idxk,:,:]*ugrad_vy[idx,:,:]+\
                     VORTZf_k[idxk,:,:]*ugrad_vz[idx,:,:]
            PI_Z[idxk,idx] = np.mean(tmp_PI)
        del tmp_PI
    
    
# Save output for each nc file
# Data to be saved¨
#  kv, E1dr, E1da, PI_E, PI_Z

# Name of NetCDF file to save
# Take relevant information from the name file
tab = file.split('/')[-1].split('.')
prefix,vinfo, tinfo = tab[0],tab[2],tab[4]
file_netcdf='_'.join(['Spectra',prefix,vinfo,tinfo])
file_netcdf1=pathsave+file_netcdf+'.nc'

kvazi=np.arange(0,360,5) # for azimuthal

# data_dims={}
# # Dimensions
# data_dims[0]=kv
# data_dims[1]=np.arange(0,360,5) # for azimuthal
# data_dims[2]=z
# # Variables to save
# data={}
# data['E1dr']=E1dr
# data['E1da']=E1da
# data['PI_E']=PI_E
# data['PI_Z']=PI_Z
# data['PBL']=PBLheight
# data['var']=var
# tl.writenetcdf(file_netcdf1,data_dims,data)
    
    
ds = xr.Dataset(
    {
    "E1dr": (("kv", "z"), E1dr),
    "E1da": (("kvazi", "z"), E1da),
    "PI_E": (("kv", "z"), PI_E),
    "PI_Z": (("kv", "z"), PI_Z),
    "variance": (("z",),var),
    },
    coords={"kv":kv, "z": z, "kvazi":kvazi}
)
# Add a scalar variable (e.g., a global attribute)
ds["PBL"] = PBLheight  # A single value

# Save to NetCDF (overwrites if exists)
file_netcdf2=pathsave+file_netcdf+'.nc'
ds.to_netcdf(file_netcdf2)
    
    
    









