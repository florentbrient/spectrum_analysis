#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 16:34:02 2024

@author: fbrient
"""

import numpy as np
import netCDF4 as nc
import tools0 as tl
import tools_for_spectra as tls
import random
import glob

def read_info(fileinfo):
    with open(fileinfo, 'r', encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    #print(lines)
    info_dict=dict()
    for ij in lines:
        tmp = ij.rstrip('\n').strip().split(': ')
        key=tmp[0]
        info_dict[key]=tmp[1]
    #print(info_dict)
    return info_dict


# Open information from 'info_run_JZ.txt'
pathinfo  = '../infos/'
fileinfo  = pathinfo+'info_run.txt'
info_dict = read_info(fileinfo)

vtyp   =info_dict['vtyp']
machine=info_dict['machine']
path0  =info_dict['path']
case   =info_dict['case']
sens   =info_dict['sens']
prefix =info_dict['prefix']
OUT     =info_dict['OUT']
nc4     =info_dict['nc4']
if 'subdir' in info_dict.keys():
    subdir=info_dict['subdir']
else:
    subdir=''
if 'pathfig' in info_dict.keys():
    pathfig0=info_dict['pathfig']
else:
    pathfig0=path0
#vtype='V0005';time='002';nc4='nc';OUT='OUT.'

oldtype=False
if oldtype:
    # Model type
    #vtyp = 'V5-5-1'
    vtyp = 'V5-7-0'
    # Path of the file 
    machine='jean-zay'
    if machine=='jean-zay':
        path0='/lustre/fsstor/projects/rech/whl/rces071/MNH-'+vtyp+'/' #+'/FI1024/REF/'
        pathfig0=path0
    elif machine=='dell':
        path0="/home/fbrient/GitHub/objects-LES/data/"
        pathfig0="/home/fbrient/GitHub/spectrum_analysis/figures/"+vtyp+'/'
    
    
    if vtyp == 'V5-5-1':
        #path0="/home/fbrient/GitHub/objects-LES/data/"
        case ='IHOPNW';prefix='IHOP0';time='006';vtype='V0001';nc4='nc';OUT='OUT.'
        #case ='IHOP';sens='trlRu0x0';prefix='004';vtype='V0301';nc4='nc4';OUT=''
        #case ='FIRE';sens='Ls2x0';prefix='024';vtype='V0301';nc4='nc4';OUT=''
        case ='BOMEX';prefix='Ru0NW';time='012';vtype='V0301';nc4='nc4';OUT=''
        sens   = prefix
        path   = path0+case+'/'+prefix+'/'
        file   = 'sel_'+prefix+'.1.'+vtype+'.'+OUT+time+'.'+nc4
        var1D  = ['vertical_levels','S_N_direction','W_E_direction'] #Z,Y,X
    elif vtyp == 'V5-7-0':
        #path0="/home/fbrient/MNH/"+vtyp+"/"
        case ='FIRE3D';sens='FI1024';prefix='FIR1k';subdir='REF'
        vtype='V0005';time='002';nc4='nc';OUT='OUT.'    
        path    = path0+case+'/'+sens+'/'+subdir+'/'
        # FIR1k.1.V0001.OUT.002.nc
        file    = prefix+'.1.'+vtype+'.'+OUT+time+'.'+nc4
        var1D  = ['level','nj','ni'] #Z,Y,X
else:
    path   = path0+vtyp+'/'+case+'/'+sens+'/'+subdir+'/'
    file0  = prefix+'.1.'+'*'+'.'+OUT+'*'+'.'+nc4
    files  = glob.glob(path+file0, recursive=True)
    print(files)
    
pretreatment = False
if pretreatment:
    var1D = ['vertical_levels','S_N_direction','W_E_direction']
else:
    var1D = ['level','nj','ni'] #Z,Y,X

for file in files:
    
    #name of the file
    namefile=file.split('/')[-1]
    vtyp    =namefile.split('.')[2]
    time    =namefile.split('.')[4]
    
    # Open the netcdf file
    file    = path+file
    DATA    = nc.Dataset(file,'r')
    
    # Define pathfig
    pathfig = tls.mk_pathfig(pathfig0,case=case,sens=sens,func='Structures_function')
    # Name figure : S3u_FIRE3D_FIR1k_V0005_002_80
    namefig0=pathfig+'{XXX}_'+case+'_'+prefix+'_'+vtype+'_'+time+'_{ZZZ}'+'{NNN}'
        
    # Open dimensions
    nxnynz,data1D,dx,dy,dz= tls.dimensions(DATA,var1D)
    z,y,x  = [data1D[ij] for ij in var1D]
    
    # Find Boundary-layer height (zi)
    inv               = 'THLM'
    threshold         = 0.25
    #idxzi,toppbl,grad = tl.findpbltop(inv,DATA,var1D,offset=threshold)
    idxzi = tl.findpbltop(inv,DATA,var1D,offset=threshold)
    PBLheight = z[idxzi]
    
    # Z of interest (zi/2 for instance)
    fracmax = 1.2
    if 'BOMEX' in case:
        fracmax = 2.5
        
    # Vertical velocity fiels
    UT = np.squeeze(DATA['UT'])
    VT = np.squeeze(DATA['VT'])
    WT = np.squeeze(DATA['WT'])
    
    # Information for computing structure functions
    sampling_rate = 1/dx
    nx            = UT.shape[-1]
    nr            = 100 # How much distance to compute lagged distance, structure 
    r_values      = np.linspace(1, nr, nr)*dx  # Lag distances to evaluate S(r)
    nc            = 100 # How many "samples" are averaged?
    ilines        = random.sample(list(np.linspace(0,nx-1,nx).astype(int)),nc)
    
    fracziall     = np.arange(0,fracmax,0.1)
    #fracziall = np.arange(0.7,0.8,0.1)
    NF            = len(fracziall)
    for iz,fraczi in enumerate(fracziall):
        print('***')
        print('Start loop for ',fraczi,'zi')
        idx    = int(fraczi*idxzi)
        
        # 2D valocity fields
        [UT2D,VT2D,WT2D] = [ij[idx,:,:] for ij in [UT,VT,WT]]
        
        ############################
        # compute structure function
        ############################
    
        # Compute power spectrum
        if iz==0:
            pow_u,pow_v,pow_w =[np.zeros((NF,nc,int(nx/2)+1)) for ij in range(3)]
            Su_r,Sv_r,Sw_r    =[np.zeros((NF,nc,nr)) for ij in range(3)]
        for il,iline in enumerate(ilines):
            # different cases
            # U along x, V along y
            # U along y, V along y
            UT2Dline = UT2D[iline,:] # U along x
            VT2Dline = VT2D[:,iline] # V along y
            WT2Dline = WT2D[iline,:] # W along x
            # The frequency domain will span from 0 to fs/2 in spatial frequencies (Nyquist frequency),
            freq, pow_u[iz,il,:] = tls.compute_power_spectrum(UT2Dline, sampling_rate,nperseg=nx)
            #Pour info: 2*np.pi*freq.max() = kv2.max()
            # Compute structure function
            Su_r[iz,il,:] = tls.compute_structure_function(UT2Dline, x, r_values)
            freq, pow_v[iz,il,:] = tls.compute_power_spectrum(VT2Dline, sampling_rate,nperseg=nx)
            Sv_r[iz,il,:] = tls.compute_structure_function(VT2Dline, x, r_values)
            freq, pow_w[iz,il,:] = tls.compute_power_spectrum(WT2Dline, sampling_rate,nperseg=nx)
            Sw_r[iz,il,:] = tls.compute_structure_function(WT2Dline, x, r_values)
            
        # Plot Power Spectrum and structure function
        zchar = "{:.0f}".format(100*fraczi)
        for norm in [True, False]:
            normch = ''
            if norm:
                normch='_norm'
            
            namefig=namefig0.format(XXX='S3u',ZZZ=zchar,NNN=normch)
            tls.plot_structure_function(freq,pow_u[iz,:,:],Su_r[iz,:,:],nx,r_values,
                                        norm=norm,PBL=PBLheight,namefig=namefig)
            namefig=namefig0.format(XXX='S3v',ZZZ=zchar,NNN=normch)
            tls.plot_structure_function(freq,pow_v[iz,:,:],Sv_r[iz,:,:],nx,r_values,
                                        norm=norm,PBL=PBLheight,namefig=namefig)
            namefig=namefig0.format(XXX='S3w',ZZZ=zchar,NNN=normch)
            tls.plot_structure_function(freq,pow_w[iz,:,:],Sw_r[iz,:,:],nx,r_values,
                                    norm=norm,PBL=PBLheight,namefig=namefig)
            
    
    # Plot averaged
    pow_u_all,Su_r_all=[np.mean(tab,axis=0) for tab in [pow_u,Su_r]]
    pow_v_all,Sv_r_all=[np.mean(tab,axis=0) for tab in [pow_v,Sv_r]]
    pow_w_all,Sw_r_all=[np.mean(tab,axis=0) for tab in [pow_w,Sw_r]]
    for norm in [True, False]:
        normch = ''
        if norm:
            normch='_norm'
        namefig=namefig0.format(XXX='S3u',ZZZ='avg',NNN=normch)
        tls.plot_structure_function(freq,pow_u_all,Su_r_all,nx,r_values,
                                    norm=norm,PBL=PBLheight,namefig=namefig)
        namefig=namefig0.format(XXX='S3v',ZZZ='avg',NNN=normch)
        tls.plot_structure_function(freq,pow_v_all,Sv_r_all,nx,r_values,
                                    norm=norm,PBL=PBLheight,namefig=namefig)
        namefig=namefig0.format(XXX='S3w',ZZZ='avg',NNN=normch)
        tls.plot_structure_function(freq,pow_w_all,Sw_r_all,nx,r_values,
                                norm=norm,PBL=PBLheight,namefig=namefig)
    
    del UT,VT,WT
