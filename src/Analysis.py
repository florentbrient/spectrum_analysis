#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 17:42:14 2023

@author: fbrient
"""
import netCDF4 as nc
#from netCDF4 import Dataset
import matplotlib as mpl
#mpl.use('Agg')
from matplotlib import pyplot as plt
#import matplotlib.cm as cm
import numpy as np
import scipy.stats as st
import os
import spectra
import tools0 as tl
import time
from scipy.stats import binned_statistic
#from scipy.optimize import curve_fit
#import plot_tools as pltl
#import gc
import cloudmetrics as cm

# test speed up savefig
from PIL import Image
from moviepy.video.io.bindings import mplfig_to_npimage

def savefig2(fig, path):
    Image.fromarray(mplfig_to_npimage(fig)).save(path)

def infocas(model,vers='v5.5.1',dx='?',dz='?'):
    textcas0={};textcas=None
    
    dt='1'
    if dx!='?':
        dx='{:0.0f}'.format(dx)
    if dz!='?':
        dz='{:0.0f}'.format(dz)
        
    versionMNH = '2D Meso-NH model ('+vers+')'
    
    textcas0['BOMEX2D']='{MNH} \nBOMEX Cumulus (Dx={XX}m, Dz={ZZ}m, Dt={TT}s)'
    textcas0['IHOP2D']='{MNH} \nIHOP Clear sky (Dx={XX}m, Dz={ZZ}m, Dt={TT}s)'
    textcas0['IHOP2DNW']='{MNH} \nIHOP Clear sky No Winds (Dx={XX}m, Dz={ZZ}m, Dt={TT}s)'
    textcas0['FIRE2Dreal']='{MNH} \nFIRE Stratocumulus (Dx={XX}m, Dz={ZZ}m, Dt={TT}s)'
    textcas0['ASTEX2D']='{MNH} \nASTEX StCu-Cu (Dx={XX}m, Dz={ZZ}m, Dt={TT}s)'
    textcas0['ARMCU2D']='{MNH} \nARMCu Cumulus (Dx={XX}m, Dz={ZZ}m, Dt={TT}s)'
    textcas0['RICO2D']='{MNH} \nRICO Cumulus (Dx={XX}m, Dz={ZZ}m, Dt={TT}s)'


    if model in textcas0.keys():
        textcas=textcas0[model]
        textcas=textcas.format(MNH=versionMNH,XX=dx,ZZ=dz,TT=dt)
        
    print('textcas ',model,textcas)
    return textcas

def mkdir(path):
   try:
     os.mkdir(path)
   except:
     pass

def find_offset(var):
    offset = 1.
    if var in ('RVT','RNPM','RCT'):
        offset = 1000.
    return offset

def removebounds(tmp):
    if len(tmp.shape) == 1:
     tmp = tmp[1:-1]
    elif len(tmp.shape) == 2:
     tmp = tmp[1:-1,1:-1]
    else:
     print('Problem ajustaxes')
    return tmp

def ajax(axe,of=1,rmb=False):
    axe = axe*of
    if rmb:
        axe=removebounds(axe)
    return axe

def variance(tmp):
    return np.var(tmp) #ddof =0 by default (maximum likelihood estimate of the variance )

def skewness(tmp):
    return st.skew(tmp)

def findlevels(var, Anom, model=None):
    levels = {}; tmp=None
    
    levels['Mean'] = {}
    levels['Anom'] = {}
    
    levels['Mean']['SVT004']=[0,9,0.1]
    levels['Mean']['SVT005']=[0,400,10]
    levels['Mean']['SVT006']=[0,200,5]
    levels['Mean']['RVT']   =[5,10,0.05]
    levels['Mean']['RNPM']  =[6,12,0.05]
    #THLM in Celsius
    levels['Mean']['THLM']  =[22,31,0.05]
    #levels['WT']    =[-2.,2.,0.05]
    levels['Mean']['WT']    =[-6.,6.,0.05]
    levels['Mean']['DIVUV'] =[-0.04,0.04,0.005]
    levels['Mean']['RVT']   =[0,17,1]
    levels['Mean']['RCT']   =[0,1,0.05]
        
    levels['Anom']['RNPM']  =[-1,1,0.05]
    levels['Anom']['THLM']  =[-1,1,0.05]
    levels['Anom']['THV']   =[-1,1,0.01]
    levels['Anom']['WT']    =[-10.,10.,0.1]
    levels['Anom']['PABST'] =[-6.,6.,0.1]
    
    if 'BOMEX' in model or 'RICO' in model:
        levels['Mean']['RNPM']  =[5,18,0.05]
    if 'ASTEX' in model:
        levels['Mean']['SVT006']=[0,10,0.1]
        levels['Mean']['WT']=[-2.,2.,0.05]
        
    levels_sel = levels['Mean']
    if Anom !=0:
        levels_sel = levels['Anom']
    
    if var in levels_sel.keys():
        tmp0  = levels_sel[var]
        tmp   = np.arange(tmp0[0],tmp0[1],tmp0[2])
        if len(tmp0)>3:
            tmp = np.append(tmp0[3],tmp)
    return tmp

def findlvl(var,model,varname='VAR'):
    # contour plot
    lvl     = [0.,4.,0.1,1.]
    
    if "FIRE" in model and varname != 'WT':
        lvl = [0.,45.,0.5,5.]
        
    if 'skew' in var:
        lvl=[-3,3,0.1,1.]
    if 'sig2' in var:
        lvl=[0,1,0.01,0.1]
    return lvl


def findunits(fl_dia,var):
    try:
        units=fl_dia[var].units
    except:
        units = '-'
        pass
    if var in ('RVT','RNPM','RCT'):
        units='g/kg'
    elif var in ('THLM','THV','THT'):
        units='°C'
    elif var in ('WT'):
        units='m/s'
    return units

def findname(var,dash=False):
    varname=var;varname0={}
    varname0['RVT']='Specific humidity'
    varname0['RNPM']='Total humidity'
    varname0['RCT']='Liquid water content'
    varname0['SVT004']='Surface tracer'
    varname0['SVT005']='Cloud-base tracer'
    varname0['SVT006']='Cloud/PBL-top tracer'
    varname0['WT']='Vertical velocity'
    varname0['THV']='Virtual potential temperature'
    if var in varname0.keys():
        varname=varname0[var]
        if dash:
            varname = varname.replace(' ',' \, ')
    return varname

# def findinfos(fl_dia,var):
#     if var in fl_dia.variables:
#         units=fl_dia[var].units
#         varname=fl_dia[var].standard_name
#     if var in ('RVT','RNPM','RCT'):
#         units='g/kg'
#     elif var in ('THLM','THV'):
#         units=fl_dia['THT'].units
#         varname=var
#     return units,varname
        
def findextrema(model):
    zminmax = None
    zmax  = {'FIRE':0.8, 'BOMEX':2, 'ARMCU':3,\
             'IHOP':2.5, 'ASTEX':2.5, 'RICO':3,}
    for x in zmax.keys():
        if x in model:
            zminmax=[0,zmax[x]]
    return zminmax

def findcmap(var,Anom=0):
    cmapall = {}
    cmapall['Mean'] = {}
    cmapall['Anom'] = {}

    # by default
    cmap= 'Blues_r'
    if Anom!=0:
        cmap='RdBu_r'
        
    cmapall['Mean']['WT']='RdBu_r'
    cmapall['Mean']['SVT004']='Reds'
    cmapall['Mean']['SVT005']='Greens'
    cmapall['Mean']['SVT006']='Greens'
    cmapall['Mean']['DIVUV']='RdBu_r'
    cmapall['Mean']['RNPM']='Blues_r'
    cmapall['Mean']['THLM']='Reds_r'
    
    cmapall['Anom']['RNPM'] = 'BrBG'
    
    cmap_s=cmapall['Mean']
    if Anom!=0:
        cmap_s=cmapall['Anom']
    
    if var in cmap_s.keys():
        cmap=cmap_s[var]
    return cmap

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
    
    levels = findlevels(var1c,Anom=Anom,model=model)
    cmap   = findcmap(var1c,Anom=Anom)
    if model is not None:
        dx=(xz[1,0]-xz[0,0])*1000.
        dz=(zx[0,1]-zx[0,0])*1000.
        textcas=infocas(model,dx=dx,dz=dz)
    
    for ij in np.arange(len(ax)):
        if ij==1: # Caution - remove data, level
            data=data2
            levels=findlevels(var2c,Anom=Anom,model=model)
            cmap=findcmap(var2c,Anom=Anom)
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
        
        if var2 is not None and joingraph:
            # Add contour
            levels=findlevels(var2c,Anom=Anom,model=model)
            if levels is not None:
                levels=levels[::10]
            #cmap_r = pltl.reverse_colourmap(cm.get_cmap(findcmap(var2c)))
            cmap_r = findcmap(var2c,Anom=Anom)
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
    savefig2(fig,filesave + '.png')
    time2 = time.time()
    print('%s function took %0.3f ms' % ("Savefig 2 ", (time2-time1)*1000.0))

    #fig.savefig(filesave + '.pdf')
    plt.close()
    #gc.collect()
    return fig,ax


def func(x, a, b, c, d):
  return a + b * x + c * x ** 2 + d * x ** 3

def integrale_trapez(yy_x,x):
# ===========================
  # compute the integral yy_x dx using the trapezoidal rule
    dx = x[1:]-x[:-1] # wavenumbre ingrement between two points
    # Compute the integral over the whole spectrum using the trapezoidal rule
    integr = np.sum(0.5*(yy_x[1:]+yy_x[:-1])*dx)
    return integr


#####
# This code aims to calculate and draw 
# -variance
# -skewness
# -spectrum analysis
# -length scale
# from of 2D-field outputs of the MNH model (v5.5.1)
# Update to use 3D field


# Choices
# 2D or 3D simualtions
Dim='3D'
# Make figures
makefigures=1
# Plot cloud in makefigures
pltcloud=True
# Plot Anomalies
pltanom=False
# Join graph of var1 and var2
joingraph=False
# Plot turbulent spectra
pltspectra=True

Anom = 0; anomch=''
if pltanom:
    Anom = 1
    anomch='Anomaly of '

# Version MNH
Version="V5-5-1"
#Version="V5-7-0"

# 2D
if Dim=='2D':
    file0 = 'EXP.1.VTIME.OUT.TTIME.nc'
    VTIME = 'V0001'
    model = 'RICO2D'
    #'FIRE2Dreal' #'IHOP2D' , 'BOMEX2D', 'IHOP2DNW','BOMEX2D', 'ARMCU2D', 'ASTEX2D'
    T0    = 1
    TMAX  = 760 #400 #840
    if model == 'FIRE2Dreal':
        #EXP   = 'FIR2D'
        EXP   = 'F2DNW'
        #EXP   = 'F2DBI'
        #EXP   = 'FNWx2'
        VTIME = 'V0001'
        TMAX  = 1440 #2880 #720
    elif model == 'IHOP2D':
        EXP   = 'IHOP0'
        TMAX  = 840
    elif model == 'IHOP2DNW':
        EXP   = 'IHOP2'
        TMAX  = 8*60
    elif model == 'BOMEX2D':
        #EXP   = 'BOM2D'
        EXP   = 'B2DNW'
        VTIME = 'V0001'
        TMAX  = 500
    elif model == 'RICO2D':
        EXP   = 'R2DNW'
        VTIME = 'V0001'
        TMAX  = 720
    elif model == 'ARMCU2D':
        #EXP   = 'Ru0x0'
        EXP   = 'Ru0NW'
        VTIME = 'V0001'
        TMAX  = 900
    elif model == 'ASTEX2D':
        EXP   = 'ASTEX'
        VTIME = 'V0001'
        #VTIME = 'L1V01'
        VTIME = 'h1V01'
        #T0    = 1680
        TMAX  = 840
    path0 = '/home/fbrient/MNH/'
    path0+=Version+'/'+model+'/'
    hours = np.arange(T0,TMAX,1)
    dt    = 1./60. # minutes 
    #hours = np.arange(160,169,1)
    name_xy = ['ni','nj','level']
    var1D   = [name_xy[-1],name_xy[0]]
    rmb     = False # true by default?
elif Dim=='3D':
    # 3D
    VTIME = "V0001"
    model,EXP = "IHOPNW","IHOP0"
    #model,EXP,VTIME = "FIREWIND","WDFor","V0301"
    #model,EXP,VTIME = "FIRE","Ls2x0","V0301"
    
    # Simulations from Jean-Zay (new MNH)
    path='/home/fbrient/GitHub/objects-LES/data/'
    file0 = 'sel_EXP.1.VTIME.OUT.TTIME.nc'
    
    # Old Simulations (from CNRM)
    #file0 = 'sel_EXP.1.VTIME.OUT.TTIME.nc'
    
    T0,TMAX,DT    = 1,6,1
    if "FIRE" in model:
       T0,TMAX,DT    = 3,24,3
       if EXP=="Ls2x0":
           file0=file0.replace('OUT.','')
           file0=file0.replace('.nc','.nc4')
       
    path0 = '/home/fbrient/GitHub/objects-LES/data/'+model+'/'+EXP+'/'
    hours = np.arange(T0,TMAX+DT,DT)
    dt    = 1 # saving each hour
    
    name_xy = ['W_E_direction','S_N_direction','vertical_levels']
    var1D   = name_xy[::-1]
    # 3D field has properties axis such as
    # vertical_levels, S_N_direction, W_E_direction
    rmb     = False

file0 = file0.replace('EXP',EXP)
file0 = file0.replace('VTIME',VTIME)
file0 = path0+file0

# Variable to study
#var1c  = 'RNPM', 'THV', 'SVT004','RCT','LWP, 'TKE'
var1c  = 'RNPM'


var2c  = None
#var2c  = 'WT', 'THLM', 'THV', 'SVT006'

covar = None
#covar = var1c
#if covar is not None:
 #   var2c = None

varall  = ''.join([var1c,var2c]) if var2c is not None else var1c
varall  = 'c'+''.join([var1c,var2c]) if covar is not None else var1c

varall2 = varall

if pltanom:
    varall2+='_anom'
if joingraph:
    varall2+='_join'

# Infos figures
zminmax = findextrema(model)

# Smoothing SpectraS
smoothspectra=1
i0=0;ilog=0
if Dim=='2D':
    smoothspectra=60 #60 #can be changed

# Polytfit Spectra
fitpoly = True
xpoly   = 40
yfit    = None

# Labels spectra
lab_spectra = [r'$\mathbf{W - E \;direction}$']
if Dim=='3D':
    lab_spectra += [r'$\mathbf{S - N \;direction}$']
    lab_spectra2D = [r'$\mathbf{2D \;domain}$']

# How to find PBL top    
inv       = 'THLM'
offset    = 0.25
    
# Save figures
#textcas=infocas(model)
pathsave0="../figures/"+Version+'/'
pathsave0+=Dim+"/";mkdir(pathsave0)
pathsave0+=model+"/";mkdir(pathsave0)
pathsave0+=EXP+"/";mkdir(pathsave0)
pathsave1=pathsave0 # For final plots
pathsave0+=varall2+"/";mkdir(pathsave0)
filesave0=pathsave0+'_'.join([varall2,EXP,VTIME])+'_TTIME'

if covar:
    pathsave2=pathsave1+varall+'/';mkdir(pathsave2) # For spectra
else:
    pathsave2=pathsave1+var1c+'/';mkdir(pathsave2) # For spectra
filesptr0=pathsave2+'spectra_'+'_'.join([varall,EXP,VTIME]) \
    +'_TTIME'+'_zALT_s'+'{0:02}'.format(smoothspectra)


# File NetCDF
path_netcdf="../data/"
file_netcdf=path_netcdf
file_netcdf+='data'+Dim+'_'+model+'_'+EXP+'_'+var1c
file_netcdf+='_'+Version
file_netcdf+='_s'+'{0:02}'.format(smoothspectra)
file_netcdf+='_p'+'{0:02}'.format(xpoly)
file_netcdf+='.nc'

# Check if the file exists
overwrite=True
isfile=(not overwrite)*os.path.isfile(file_netcdf)

# Initiate
var2 = None
# Number of axis (x and y, or x only)
Na    = len(var1D)-1

if not isfile:
    time_start     = time.time()
    for ih,hour in enumerate(hours):
        file      = file0.replace('TTIME','{0:03}'.format(hour))
        filesptr1 = filesptr0.replace('TTIME','{0:03}'.format(hour))
        print(file)
        fl_dia = nc.Dataset(file, 'r' )
        
        if ih==0:
            # Open dims
            ofdim = 1e-3
            ni=ajax(np.array(fl_dia[name_xy[0]][:]),of=ofdim,rmb=rmb)
            nj=ajax(np.array(fl_dia[name_xy[1]][:]),of=ofdim,rmb=rmb)
            #nj_u=ajax(np.array(fl_dia['nj_u'][:]),of=ofdim,rmb=rmb)
            #ni_u=ajax(np.array(fl_dia['ni_u'][:]),of=ofdim,rmb=rmb)
            #ni_v=ajax(np.array(fl_dia['ni_v'][:]),of=ofdim,rmb=rmb)
            #nj_v=ajax(np.array(fl_dia['nj_v'][:]),of=ofdim,rmb=rmb)
            level=ajax(fl_dia[name_xy[2]][:],of=ofdim,rmb=rmb)
            #level_w=ajax(fl_dia['level_w'][:],of=ofdim,rmb=rmb)
            
            
            delta_x = ni[1]-ni[0]
            if len(nj)>1:
                delta_y = nj[1]-nj[0]
            delta_z = level[1]-level[0]
            
            zx,xz = np.meshgrid(level,ni)
            #print('level : ',level,xz,zx)
            
            NL      = len(level)
            NH      = len(hours)
            NHsp    = int(len(hours)/smoothspectra)
            
            # Init
            sig2,skew  = [np.zeros((NH,NL))*np.nan for ij in range(2)]
            lambda_max = np.zeros((Na,NHsp,NL))*np.nan
            lambda_fit = np.zeros((Na,NHsp,NL))*np.nan
            lambda_cm  = np.zeros((Na,NHsp,NL))*np.nan
            idxzi      = np.zeros(NH,dtype=int)
            wstar      = np.zeros(NH)
            if Dim=='3D':
                lambda_max2D = np.zeros((NHsp,NL))*np.nan
                lambda_fit2D = np.zeros((NHsp,NL))*np.nan
                lambda_cm2D  = np.zeros((NHsp,NL))*np.nan
        
        timech = None
        if 'time' in fl_dia.variables.keys():
            timech = fl_dia.variables['time']
            timech = nc.num2date(timech[:],timech.units,only_use_python_datetimes=False)
            timech = timech[0].strftime('%Y-%m-%d (%H:%M:%S)')
    
        time1     = time.time()
        idxzi[ih] = tl.findpbltop(inv,fl_dia,var1D,offset=offset)
        time2     = time.time()
        print('%s function took %0.3f ms' % ("Find PBL", (time2-time1)*1000.0))
        #print(hour,idxzi[ih])
        #print(level[idxzi[ih]])
    
        # Open variables
        var   = tl.createnew(var1c,fl_dia,var1D)
        #print(var1c,var)
        #var   = fl_dia[var1c][:].squeeze() #level, ni
        #var   = var[:].squeeze()
        ofdim = find_offset(var1c)
        print(var1c,ofdim,rmb)
        var   = ajax(var,of=ofdim,rmb=rmb)
        
        # Second variable (optionnal)
        if var2c is not None:
            var2   = tl.createnew(var2c,fl_dia,var1D)
            #var2  = fl_dia[var2c][:].squeeze()
            var2  = ajax(var2,find_offset(var2c),rmb=rmb) #.T
            
        if covar is not None:
            var  = tl.anomcalc(var)
            var2 = tl.anomcalc(var2)
            var  = var*var2
    

        # Omega star (To remove non-turbulent layer)
        # Not updated to remove bounds
        wstar[ih]   = tl.createnew('Wstar',fl_dia,var1D,idxzi=idxzi[ih])
        # To be corrected
        #wstar[ih] = 0.0
        print('Wstar = ',ih,wstar[ih])
        
    ######################
    ### START ANALYSIS ###
    ######################
        
        # Figure 2D
        if makefigures and Dim=="2D":
            filesave = filesave0.replace('TTIME','{0:03}'.format(hour))
            
            RCT=None
            if pltcloud and ('RCT' in fl_dia.variables):
                RCT = ajax(fl_dia['RCT'][:].squeeze().T,of=1000.,rmb=rmb)
            varplot, varplot2 = var,var2
#            print('var,var2 ',var.shape,var2.shape)
#            stop
            if pltanom:
                varplot=tl.anomcalc(varplot)
                if varplot2 is not None:
                    varplot2=tl.anomcalc(varplot2) #.T) #.T
                    
            title=[anomch+findname(var1c)+' ('+findunits(fl_dia,var1c)+')']
            if var2c is not None:
                title+=[anomch+findname(var2c)+' ('+findunits(fl_dia,var2c)+')']
                varplot2= varplot2.T
            fig,ax   = plot2D(varplot.T,xz,zx,filesave,
                              var1c=var1c,title=title,
                              zminmax=zminmax,Anom=Anom,
                              labelx='x (km)',labely='z (km)',
                              RCT=RCT,idx_zi=idxzi[ih],
                              data2=varplot2,var2c=var2c,
                              timech=timech,joingraph=joingraph,
                              model=model)
            #plt.contourf(xi,zi,var.T)
            #plt.show()
            del RCT
            
            
        # Temporal averaging
        i0+=1
        log_spec=(i0 == smoothspectra)
        
        NX,NY = len(ni),len(nj)
        NS    = int(np.floor(NX/2))-1 #round(NX/2)
        print('NX,NS: ',NX,NS)
        
        if Dim=='2D' and ih==0:   
            spec_log_avg=np.zeros((Na,NL,NS)) 
            nb_avg      =np.zeros(NL) 
            NX,NY       =1,1
            axis0       =(0,2)
        if Dim=='3D':
            spec_log_avg  =np.zeros((Na,NL,NS)) 
            spec_log2D_avg=np.zeros((NL,NS)) 
            nb_avg        =np.zeros(NL) # not using smoothspectra
            axis0         =(1,2)
        
        zi = level[idxzi[ih]]
        print('**** PBL height zi: ',zi)
        
        if var.shape[0] > 1:
            
            # Remove non-turbulent levels
            wstd = np.nanstd(fl_dia['WT'],axis=axis0)
            
            # Filtering non turbulent layer
            # Non turbulent defined as std(w) < 0.10 w_star
            # Convection only when sigma_w > 10 % w_star
            minturb = 0.10
            level_filtered = np.where(wstd/wstar[ih]>minturb)[0]
            # Loop only filtered level ONLY !
            #[print('wtsd ',im,wstar[ih]) for im in wstd]
            #print(level_filtered)
        else:
            # 2D field such as LWP
            level_filtered = [0]
        
        
        print('**** Start level_filtered')
        timeA = time.time()
        for ik,zz in enumerate(level_filtered):
            #print(ik,zz)
            if Dim=='2D':
                var0    = var[zz,:]
                var0all = var0
            else:
                var0    = var[zz,:,:]
                var0all = var0.flatten()
                
            # Variance and skewness on full scale
            #sigma2     = variance(var0all)
            #skew0      = skewness(var0all)
            # Save data
            sig2[ih,zz] = variance(var0all)
            skew[ih,zz] = skewness(var0all)
            
            nb_avg[zz]+= 1.
             
            # Saving spectra
            if Dim=='2D':
                #k_v,SPECTRE_V,VAR_V,spec_log0 = spectra.spectra(var0,delta_x)
                k_v,SPECTRE_V,VAR_V,spec_log0 = spectra.spectra1D(var0,delta_x)
                spec_log_avg[0,zz,:]+=spec_log0
            
            if Dim=='3D':
                # Need to verify direction
                for ij in range(NX):
                    var1   = var0[ij,:] # each spectrum along W_E_direction
                    
                    # v1 PEB
                    #time1     = time.time()
                    #k_v,SPECTRE_V,VAR_V,spec_log0 = spectra.spectra(var1,delta_x)
                    #time2     = time.time()
                    #print('%s function took %0.3f ms' % ("PEB code", (time2-time1)*1000.0))
                    
                    # v2 : Me
                    #time1     = time.time()
                    k_v,SPECTRE_V,VAR_V,spec_log0 = spectra.spectra1D(var1,delta_x)
                    #time2     = time.time()
                    #print('%s function took %0.3f ms' % ("New code", (time2-time1)*1000.0))
                    spec_log_avg[0,zz,:]+=spec_log0 #S-N
                    
                for ij in range(NY):
                    var1   = var0[:,ij] # each spectrum along S_N_direction
                    # v1 PEB
                    #k_v,SPECTRE_V,VAR_V,spec_log0 = spectra.spectra(var1,delta_y)
                    # v2
                    k_v,SPECTRE_V,VAR_V,spec_log0 = spectra.spectra1D(var1,delta_y)
                    spec_log_avg[1,zz,:]+=spec_log0 #W-E
                spec_log_avg[0,zz,:]/=NX
                spec_log_avg[1,zz,:]/=NY
                
                
                # Test 2D
                k_v2D,SPECTRE_V2D,VAR_V2D,spec_log2D_avg[zz,:] =\
                    spectra.spectra2D(var0,delta_x)
                    
                # spec_log2D is from cloudmetrics    
                spec=spec_log2D_avg[zz,:]
                #print('VAR: ',np.mean(var0),np.var(var0),np.var(var0)*np.pi/4.)
                #tmpint = integrale_trapez(spec,k_v2D)
                # Les 2 premiers sont égales, le 3eme quasiment aux 2 autres
                #print('INTEGRALE ',tmpint)
                #print('INT 2: ', np.trapz(spec, x=k_v2D))
                #var_psd = np.sum(spec) * 2 * np.pi / (np.min(var0.shape)*delta_x)
                #print('INT 3 ',var_psd)
                #print('INT 4 ', np.sum(spec))
                
                    
                #print(k_v,k_v2D)
                #print(k_v,2*np.pi/k_v2D)
                #spectra.plot_spectra(k_v2D,spec_log2D,'test_spectra2D')
                #stop
                
                
            del SPECTRE_V,VAR_V,spec_log0

        timeB = time.time()
        print('%s function took %0.3f ms' % ("Loop Level_fitered ", (timeB-timeA)*1000.0))

        # Smoothing spectra
        if log_spec:
            min_nb=smoothspectra/2
            timeX = time.time()
            # No filter for 2D
            for ik,zz in enumerate(level):
                #print(ik,zz,nb_avg[ik],min_nb)
                if nb_avg[ik]>min_nb:
                    #print(ik,zz,' IN LOOP')
                    infoch    = [zz,findname(var1c,dash=True)]
                    #NIJ=spec_log_avg.shape[0]
                    spec_plot = np.zeros((Na,NS))
                    spec_fit  = np.zeros((Na,NS))
                    for ij in range(Na):
                        #spec_plot[ij,:] = spec_log_avg[ij,ik,:] / smoothspectra #old
                        spec_plot[ij,:] = spec_log_avg[ij,ik,:] / nb_avg[ik]

                        if not np.isnan(spec_plot[ij,:]).all():
                            lambda_max[ij,ilog,ik] = spectra.lamb2k(k_v[np.argmax(spec_plot[ij,:])])
                            lambda_cm[ij,ilog,ik]  = cm.scalar.spectral_length_median(k_v, spec_plot[ij,:])
                            
                            if fitpoly:
                                xp   = k_v
                                p    = np.poly1d(np.polyfit(xp, spec_plot[ij,:], xpoly))
                                spec_fit[ij,:] = p(xp)

                                #p, pcov = curve_fit(func, xp, spec_plot[ij,:])
                                #spec_fit[ij,:] = func(xp, *p)
                                lambda_fit[ij,ilog,ik] = spectra.lamb2k(xp[np.argmax(spec_fit[ij,:])])

                    # Plot spectra            
                    if pltspectra and not np.isnan(spec_plot[0,:]).all(): # and ik==40.:
                        filesptr  = filesptr1.replace('ALT','{:4.1f}'.format(zz*1000.))
                        
                        y1b,y1bfit = None,None
                        if Na>1:
                           y1b,y1bfit= spec_plot[1,:],spec_fit[1,:]
                        zmax = None
                        if fitpoly:
                            zmax = lambda_fit[ij,ilog,ik]
                        spectra.plot_spectra(k_v,spec_plot[0,:],filesptr,\
                                             y1afit=spec_fit[0,:],\
                                             y1b=y1b,y1bfit=y1bfit,\
                                             pltfit=False,\
                                             infoch=infoch,zi=zi,zmax=zmax,\
                                             labels=lab_spectra)
                    # Analysis of 2D field
                    if Dim == '3D':
                        spec_plot2D = spec_log2D_avg[ik,:]
                        if not np.isnan(spec_plot2D).all():
                            lambda_max2D[ilog,ik] = spectra.lamb2k(k_v2D[np.argmax(spec_plot2D)])
                            if fitpoly:
                                xp   = k_v2D
                                p    = np.poly1d(np.polyfit(xp, spec_plot2D, xpoly))
                                spec_fit2D = p(xp)
                                lambda_fit2D[ilog,ik] = spectra.lamb2k(xp[np.argmax(spec_fit2D)])
                            if pltspectra:
                                filesptr  = filesptr1.replace('ALT','{:4.1f}'.format(zz*1000.))
                                filesptr  = filesptr.replace('spectra','spectra2D')
                                zmax = None
                                if fitpoly:
                                    zmax = lambda_fit2D[ilog,ik]
                                spectra.plot_spectra(k_v,spec_plot2D,filesptr,\
                                                     y1afit=spec_fit2D,\
                                                     pltfit=False,\
                                                     infoch=infoch,zi=zi,zmax=zmax,\
                                                     labels=lab_spectra2D)
                                    
                            # Calculate length scale from cloudmetrics
                            #wavenumbers, psd_1d_radial, psd_1d_azimuthal = cm.scalar.compute_spectra(var[ik,:,:],dx=delta_x)
                            #print(ilog,ik,lambda_max2D[ilog,ik])
                            #spectral_median0 = cm.scalar.spectral_length_median(wavenumbers, psd_1d_radial)
                            spectral_median2D = cm.scalar.spectral_length_median(k_v2D, spec_plot2D)
                            
                            #lambda_cm0[ilog,ik] = spectral_median0
                            lambda_cm2D[ilog,ik] = spectral_median2D

                            #print('Plot Graph Test')
                            #plt.loglog(wavenumbers,psd_1d_radial,'k')
                            #plt.loglog(k_v,spec_plot2D,'r')
                            #plt.loglog(k_v,spec_plot[0,:],'b')
                            #plt.loglog(k_v,spec_plot[1,:],'b--')
                            #print(spectral_median0,spectral_median1)
                            #plt.show()
                            #stop
                    del spec_plot,spec_fit
                    
            # Save spec_fit?

            # End of spectrum averaging - Reinit
            i0=0; ilog+=1
            nb_avg = np.zeros(NL)
            spec_log_avg=np.zeros((Na,NL,NS))
            
            timeY = time.time()
            print('%s function took %0.3f ms' % ("Log_Spec ", (timeY-timeX)*1000.0)) #6.4s
        
            
        #plt.plot(lambda_max[ih,:],level)
        #plt.show()    
        del var,fl_dia
        #gc.collect()
        print('**** End loop')


    
    # end time
    time_end     = time.time()
    print('%s tooks %0.3f s' % ("*!*!*!* The program", (time_end-time_start)))

        
    # does not work for smoothspectra=1
    idxsp           = np.arange(smoothspectra-1,NH,smoothspectra) 
    hourspectra     = hours[idxsp]*dt
    PBLspectra      = level[idxzi][idxsp]
    
    
    ### Saving netcdf
    # PBLspectra
    # sig2, skew
    # Name file
    # data_model_EXP.nc
    
    #file_netcdf='data_'+model+'_'+EXP+'.nc'
    data_dims={}
    # Dimensions
    if Dim=='3D':
        data_dims[0]=var1D[-2:]
    else:
        data_dims[0]=var1D[-1]
    data_dims[1]=hourspectra
    data_dims[2]=level
    hours       =hours* dt
    data_dims[3]=hours # all hours for 2D
    # Variables to save
    data={}
    data['lambda_max']=lambda_max
    data['lambda_fit']=lambda_fit
    data['PBLspectra']=PBLspectra
    data['sig2']=sig2
    data['skew']=skew
    data['lambda_cm']=lambda_cm
    if Dim=='3D':
        data['lambda_max2D']=lambda_max2D
        data['lambda_fit2D']=lambda_fit2D
        data['lambda_cm2D']=lambda_cm2D
    
    tl.writenetcdf(file_netcdf,data_dims,data)

else:
    datach=['lambda_max','lambda_fit','lambda_cm',\
            'PBLspectra',\
            'skew','sig2']
    if Dim=='3D':
        datach += ['lambda_max2D','lambda_fit2D']
        datach += ['lambda_cm2D']
    hourspectra,level,hours0,\
    data = tl.opennetcdf(file_netcdf,datach)

    
########################
### Spectra analysis ###
########################

# Plot spectra
#zi,hourspectrai = np.meshgrid(level,hourspectra)
#CS    = plt.contourf(hourspectrai,zi,lambda_max)
#cbar  = plt.colorbar(CS)
#plt.show()

# Relative to zi
relzi = True
PBL   = data['PBLspectra']

#lambdach=['lambda_max']
#if fitpoly:
#    lambdach+=['lambda_fit']
#if Dim=='3D':
#    lambdach+=['lambda_max2D','lambda_fit2D']
    
# All variables saved in data
lambdach = list(data.keys())

#title0 = 'Length scale (LLL - DDD)'
title0 = 'VAR'


for ik,lamb in enumerate(lambdach):
    #title0   = title0.replace('LLL',lamb.split('_')[-1])   
    title1   = title0.replace('VAR',lamb)
    namefig1 = '_'.join([lamb,var1c,EXP,model])
    namefig1 += '_s'+'{0:02}'.format(smoothspectra)
    
    
    x,y   = hourspectra,level
    lvl = findlvl(lamb,model,varname=var1c)
    
    tmp0     = data[lamb]
    Naplot = Na
    if len(tmp0.shape)==2: #'2D' in lamb:
        Naplot = 1
        tmp0 = tmp0[np.newaxis,...]
    elif len(tmp0.shape)==1:
        Naplot=0
        
    for ij in range(Naplot):
        #axspec   = '-'.join(var1D[:2][-1-ij].split('_'))
        namefig2 = namefig1
        title2   = title1
        if Naplot>1:
            axspec   = '-'.join(var1D[1:][-1-ij].split('_')[:-1])
            namefig2 = namefig2+'_'+axspec
            title2  += ' ('+axspec+')' #title2.replace('DDD',axspec)
            
        tmp = tmp0[ij,:,:]
        tmp = tmp.T # 2D
        if x.shape[0] != tmp.shape[1]: # average hours
            tmpnew = []
            for ih in x:
                print(int(ih))
                idx = (dt*hours>=hourspectra[int(ih)-1]-1) & (dt*hours<hourspectra[int(ih)-1])
                aa = np.mean(tmp[:,idx],axis=1)
                tmpnew.append(aa)                
            tmp = np.array(tmpnew).T
            
        
        relziplot = False
        if relzi and 'lambda' in lamb:
            tmp /= PBL
            title2   += ' relative to zi (-)'
            namefig2 += '_zi'
            relziplot = True
        #else:
        #    title2   += ' in km'
        
        # Is it a 1D data?
        plot2D=True
        # Problem: A corriger pour LWP !!
        #if len(tmp[~np.isnan(tmp)].shape)==1:
        #    tmp = tmp[~np.isnan(tmp)]
        #    plot2D=False
    
        spectra.contour_length(x,y,tmp,\
                        pathfig=pathsave2,
                        namefig=namefig2,\
                        title=title2,plot2D=plot2D,
                        zminmax=zminmax,\
                        lvl=lvl,\
                        PBLheight=PBL,relzi=relziplot)
            
        if plot2D:
            namefig3 = namefig2+'_plot'
            spectra.plot_length(x,y,tmp,\
                        pathfig=pathsave2,
                        namefig=namefig3,\
                        title=title2,var=lamb,\
                        zminmax=zminmax,\
                        #lvl=lvl,\
                        PBLheight=PBL,relzi=relziplot)

# Variance, skewness
#zk,hoursi = np.meshgrid(level,hours)
#CS    = plt.contourf(hoursi,zk,data['skew'])
#cbar  = plt.colorbar(CS)
#plt.show()

