#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 16:52:24 2023

@author: fbrient
"""
import numpy as np
import Constants as CC
from scipy import integrate
from scipy import ndimage
import netCDF4 as nc
from netCDF4 import Dataset
#import gc
import os
import scipy.stats as st
from copy import deepcopy
import scipy as sp
from scipy.spatial import cKDTree

# test speed up savefig
#from PIL import Image
#from moviepy.video.io.bindings import mplfig_to_npimage



def mkdir(path):
   try:
     os.mkdir(path)
   except:
     pass

#def savefig2(fig, path):
#    Image.fromarray(mplfig_to_npimage(fig)).save(path)

def repeat(zz,ss):
    #if len(ss)==1:
    zz  = np.repeat(zz[ :,np.newaxis],ss[0],axis=1)        
    if len(ss)==2:
        zz  = np.repeat(zz[ :,:, np.newaxis],ss[1],axis=2)
    return zz

def ajax(axe,of=1,rmb=False):
    axe = axe*of
    if rmb:
        axe=removebounds(axe)
    return axe

def removebounds(tmp):
    if len(tmp.shape) == 1:
     tmp = tmp[1:-1]
    elif len(tmp.shape) == 2:
     tmp = tmp[1:-1,1:-1]
    else:
     print('Problem ajustaxes')
    return tmp

def resiz(tmp): # should be removed by pre-treatment
    return np.squeeze(tmp)

def tryopen(vv,DATA):
    try:
        tmp = resiz(DATA[vv][:])
    except:
        print('Error in opening ',vv)
        tmp = None
    return tmp

def anomcalc(tmp):
    # ip = 0 (3D)
    ss    = tmp.shape
    if len(ss)==2:
        mean = np.mean(tmp,axis=-1)
        mean = np.repeat(mean[ :,np.newaxis],ss[-1],axis=-1)
    else:    
        mean  = np.mean(np.mean(tmp,axis=2),axis=1)
        mean    = np.repeat(np.repeat(mean[ :,np.newaxis, np.newaxis],ss[1],axis=1),tmp.shape[2],axis=2)
    data = tmp-mean
    return data

def tht2temp(THT,P):
#    ss   = P.shape
    p0   = 100000. #np.ones(ss)*100000. #p0
    exner= np.power(P/p0,CC.RD/CC.RCP)
    temp = THT*exner
    return temp

def createrho(T,P):
    ss   = T.shape
    RR   = np.ones(ss)*CC.RD
    rho  = P/(RR*T)
    return rho

def divergence(field, dx, axis=0):
    "return the divergence of a n-D field"
    #data3 = np.sum(data2,axis=0)
    return np.gradient(field, dx, axis=axis)


def variance(tmp):
    return np.var(tmp) #ddof =0 by default (maximum likelihood estimate of the variance )

def skewness(tmp):
    return st.skew(tmp)


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


# Infos for figures
def infosfigures(cas, var, mtyp='Mean',relative=False):
   cmaps   = {'Mean':'Greys_r','Anom':'RdBu_r'} # by default

   # Switch color label
   switch = []
   # switch  = ['RNPM','RVT']
   if var in switch:
       cmaps['Anom'] = cmaps['Anom'].split('_')[0]
   # Grey color
   greyvar = ['Reflectance']
   if var in greyvar:
     cmaps['Mean'] = 'Greys_r'

   zmin  = 0
   # km
   zmax  = {'IHOP':2, 'FIRE':1.0, 'BOMEX':2, 'ARMCU':2}
   if relative:
       zmax = {'IHOP':2, 'FIRE':1.5, 'BOMEX':3, 'ARMCU':3}

   vrang = {}
   vrang['Mean'] = {'WT':[-5.0,5.0,0.2],
                    'LWP':[0.00,0.14,0.002],
                    'RNPM'  :[0.005,0.01,0.0002],
                    'WINDSHEAR' :[0.,1.,0.02],
                    'REHU':[0.5,0.8,0.01],
                    'Reflectance' :[0.1,1.,0.01],
                    'THLM' : [298,306,2],
                    'RCT' : [0,0.0007,1e-5],
                    'DIVUV': [-0.006,0.009,0.003],
                    'VORTZ': [-0.01,0.01,0.001],
                    }
   vrang['Anom'] = {'WT'   :[-0.8,0.8,0.05],
                    'DIVUV':[-0.05,0.05,0.005],
                    'THV'  :[-1.0,0.6,0.02],
                    'THLM' :[-1.0,1.0,0.02],
                    'RNPM' :[-0.002,0.002,0.0001],
                    'REHU' :[-0.15,0.15,0.005],
                    'PABST':[-2,2,0.1]
                    }

   # Modified range for some case
   modif = {}
   modif['BOMEX'] = {'THV':0.2,'PABST':0.2}
   modif['FIRE']  = {'RNPM':0.2}

   zminmax = None
   if cas in zmax.keys():
     zminmax=[zmin,zmax[cas]]

   levels = None
   if var in vrang[mtyp].keys():
     vmin,vmax,vdiff = vrang[mtyp][var][:]
     mr = 1
     if cas in modif.keys():
         if var in modif[cas].keys():
             mr=modif[cas][var]
     print('Range : ',vmin,vmax,vdiff,var,find_offset(var),mr)
     vmin,vmax,vdiff = [ij*find_offset(var)*mr for ij in (vmin,vmax,vdiff)]
     nb              = abs(vmax-vmin)/vdiff+1
     levels          = [vmin + float(ij)*vdiff for ij in range(int(nb))]

   infofig = {}
   infofig['cmap']    = cmaps #[mtyp]
   infofig['zminmax'] = zminmax
   infofig['levels']  = levels
   return infofig

# Offset
def find_offset(var):
    offset = {}
    var1000=['LWP','IWP','RC','RT','RVT','RCT','RNPM']
    for ij in var1000:
       offset[ij] = 1000.
    var100 =['lcc','mcc']
    for ij in var100:
       offset[ij] = 100.
    
    rho0=1.14; RLVTT=2.5e6;RCP=1004.
    offset['E0']=rho0*RLVTT
    offset['Q0']=rho0*RCP
    offset['DTHRAD']=86400.

    off    = 1.
    if var in offset.keys():
       off = offset[var]
    return off


#def find_offset(var):
#    offset = 1.
#    if var in ('RVT','RNPM','RCT'):
#        offset = 1000.
#    return offset

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
    # contour plot for summarizing plot
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

def createnew(vv,DATA,var1D,idxzi=None):
    vc   = {'THLM' :('THT','PABST','RCT'),
            'THV'  :('THT','RVT','RCT'),
            'DIVUV':('UT',var1D[1]),\
            'TKE'  :('UT','VT','WT'),\
            'RNPM' :('RVT','RCT') }
    [vc.update({ij:('THT','PABST',var1D[0])}) for ij in ['PRW','LWP','Reflectance']]
    [vc.update({ij:('WT','RVT','RCT','PABST',var1D[0])}) for ij in ['Wstar','Tstar','Thetastar']]

    
    # ATTENTION: PEUT ETRE DES ERREURS 2D/3D !!!!
    
    tmp = tryopen(vv,DATA)

    data = []
    if vv in vc.keys() and tmp is None:
        data      = [tryopen(ij,DATA) for ij in vc[vv]]
        if vv == 'THV' : # THV = THT * (1 + 0.61 RVT - RCT)
            a1      = 0.61
            if data[1] is None:
                data[1] = np.zeros(data[0].shape)
            if data[2] is None:
                data[2] = np.zeros(data[0].shape)
            tmp     = data[0] * (np.ones(data[0].shape) +a1*data[1] - data[2] )
        if vv == 'THLM':
            #thetal = theta - L/Cp Theta/T Ql
            tmp = tryopen('THLM',DATA)
            if tmp is None:
                if data[2] is None: # No clouds
                    data[2] = np.zeros(data[0].shape)
                tmp = data[0] -(\
                     (data[0]/tht2temp(data[0],data[1]))\
                    *(CC.RLVTT/CC.RCP)*data[2])
            tmp = tmp-273.15 # Celsius
        if vv == 'RNPM':
            tmp = tryopen('RNPM',DATA)
            if tmp is None:
                tmp = data[0]
                #print(tmp)
                if data[1] is not None: # No clouds
                    tmp += data[1]
        #if vv == 'DIVUV':
            # DIVUV = DU/DX + DV/DY
        #    dx   = data[1][2]-data[1][1]; print(dx)
        #    tmpU = divergence(data[0], dx, axis=-1) # x-axis
        #    tmp  = tmpU #+ tmpV
            
        if 'TKE' in vv:
            tmp = 0.5*(anomcalc(data[0])**2.\
                  + anomcalc(data[1])**2\
                  + anomcalc(data[2])**2)
            if vv == 'TKEM':
                  tmp = np.sqrt(tmp / (data[0]**2+data[1]**2+data[2]**2) ) 
        if vv == 'PRW' or vv == 'LWP' or vv == 'Reflectance':
            TA   = tht2temp(data[0],data[1])
            rho  = createrho(TA,data[1])
            name= 'RVT'
            if vv == 'LWP' or vv == 'Reflectance':
                name = 'RCT'
            RCT = tryopen(name,DATA)
            if RCT is not None:
                ss  = RCT.shape
                print(ss)
                if len(ss)==3.:
                    zz = repeat(data[2],(ss[1],ss[2]))
                    tmp = np.zeros((1,ss[1],ss[2]))
                    for  ij in range(len(data[2])-1):
                        tmp[0,:,:] += rho[ij,:,:]*RCT[ij,:,:]*(zz[ij+1,:,:]-zz[ij,:,:])
                else:
                    zz = repeat(data[2],[ss[1]])
                    tmp = np.zeros((1,ss[1]))
                    for  ij in range(len(data[2])-1):
                        tmp[0,:] += rho[ij,:]*RCT[ij,:]*(zz[ij+1,:]-zz[ij,:])
                #zz  = np.repeat(np.repeat(data[2][ :,np.newaxis, np.newaxis],ss[1],axis=1),ss[2],axis=2)

            else:
                tmp = None
            if vv == 'Reflectance' and tmp is not None:
                rho_eau  = 1.e+6 #g/m3
                reff     = 5.*1e-6  #10.e-9 #m #
                g        = 0.85
                tmp      = tmp*1000. #kg/m2  --> g/m2
                tmp      = 1.5*tmp/(reff*rho_eau)
                trans    = 1.0/(1.0+0.75*tmp*(1.0-g))  # Quentin Libois
                tmp      = 1.-trans
        if vv == "Wstar" or vv == 'Tstar' or vv=='Thetastar':
            # W_star = g*zi*H0/theta_v(0-zi) #First version
            # Deardroff velocity
            # W_star = (g/T_v * zi * (w'th_v')_surf)^1/3
            # 'Wstar':('WT','RVT','RCT','PABST',var1D[0])
            WT  = data[0]
            print('Check wstar')
            print(WT.shape) # alt,y,x ou alt,x
            THV = createnew('THV',DATA,var1D)
            TV  = tht2temp(THV,data[3])
            #print(THV[0,15],TV[0,15])
            ss  = WT.shape 
            if len(ss)==3:
                WT  = np.reshape(WT,(ss[0],ss[1]*ss[2]))
                THV = np.reshape(THV,(ss[0],ss[1]*ss[2]))
                TV  = np.reshape(TV,(ss[0],ss[1]*ss[2]))
            zz   = data[4]
            
            # Find maximum of W'Thv' (near surface)
            wthv = 0.
            for ij,zz1 in enumerate(zz):
                tmp  = WT[ij,:]*(THV[ij,:]-np.nanmean(THV[ij,:]))
                #print(ij,wthv,np.nanmean(tmp))
                wthv = max(wthv,np.nanmean(tmp))
                #print(ij,zz1,wthv)
            
            
            
            #wthv=WT[0,:]*(THV[0,:]-np.nanmean(THV[0,:])) #surface
            #print(THV[0,:]-np.nanmean(THV[0,:]))
            #wthv=np.nanmean(wthv)
            if idxzi is not None:
                print('PBL top ',zz[idxzi])
                tmp=CC.RG*zz[idxzi]*wthv/np.nanmean(TV[0:idxzi,:])
                #print(CC.RG,zz[idxzi],wthv,np.nanmean(TV[0:idxzi,:]))
                tmp=pow(tmp,1./3.)
            #print('tmp ',tmp,zz[idxzi])
            if vv == 'Tstar' and tmp is not None :
                #t_star = H/w_star
                tmp = zz[idxzi]/tmp # en secondes
            if vv == 'Thetastar' and tmp is not None :
                #theta_star = Qs/w_star
                tmp = wthv/tmp
    return tmp




def findpbltop(typ,DATA,var1D,idx=None,offset=0.25):
    tmp     = createnew(typ,DATA,var1D)
    print('len ',len(tmp.shape))
    if len(tmp.shape)==2:
        temp   = np.nanmean(tmp,axis=(-1))
    else:
        temp   = np.nanmean(tmp,axis=(1,2,))
            
    #THLMint = integrate.cumtrapz(temp)/np.arange(1,len(temp))
    THLMint = integrate.cumulative_trapezoid(temp)/np.arange(1,len(temp))
    #DT      = temp[:-1]-(THLMint+offset)
    # Modif
    DT      = temp[1:]-(THLMint+offset)
    idx     = np.argmax(DT>0)
    return idx


def writenetcdf(file_netcdf,data_dims,data
                ,keyall = ['sig2','skew']):
    # save data in a netcdf file
    #file_netcdf='data_'+model+'_'+EXP+'.nc'
    
    # Open or create file
    try: ncfile.close()  # just to be safe, make sure dataset is not already open.
    except: pass
    ncfile = Dataset(file_netcdf,mode='w',format='NETCDF4_CLASSIC') 
    print(ncfile)

    # Creating dimensions
    xy_dim  = ncfile.createDimension('xy',len(data_dims[0]))   # xy axis # 1 for 2D, 2 for 3D
    time_dim  = ncfile.createDimension('time', len(data_dims[1])) # time axis (can be appended to).
    level_dim = ncfile.createDimension('level',len(data_dims[2])) # level axis
    timeall_dim  = ncfile.createDimension('timeall', len(data_dims[3])) # all timing axis (can be appended to).
    for dim in ncfile.dimensions.items():
        print(dim)
    
    ncfile.title='File: '+file_netcdf
    print(ncfile.title)
    
    # Creating variables
    xy = ncfile.createVariable('xy', 'S1', ('xy',))
    xy.long_name = 'Number of lat/lon dims'
    time = ncfile.createVariable('time', np.float64, ('time',))
    time.units = 'hours since 1979-01-01'
    time.long_name = 'time'
    level = ncfile.createVariable('level', np.float64, ('level',))
    level.units = 'kilometers'
    level.long_name = 'Altitude'
    timeall = ncfile.createVariable('timeall', np.float64, ('timeall',))
    timeall.units = 'hours since 1979-01-01'
    timeall.long_name = 'timeall'

    # units
    #units = dict()
    #units['lambda_max']=('xy','time','level')   
    
    # Writing data
    xy[:]    = data_dims[0]
    time[:]  = data_dims[1]
    level[:] = data_dims[2]
    timeall[:]=data_dims[3]
    for key in data.keys():
        if len(data[key].shape)==3: units=('xy','time','level')
        elif len(data[key].shape)==2: units=('time','level') 
        else: units=('time')
        if key in keyall:
            units=tuple([ij.replace('time','timeall') for ij in units])
        
        print('units ',units)
        tmp = ncfile.createVariable(key,np.float64,units) # note: unlimited dimension is leftmost
        tmp.units = '' # degrees Kelvin
        tmp.standard_name = key # this is a CF standard name
        tmp[:]   = data[key]
    
    ncfile.close()
    
def opennetcdf(file_netcdf,datach):
    # Open or create file
    # Needed variables: 
    # hourspectra,level,hours
    # lambda_max,lambda_fit,PBLspectra
    # skew,

    ncfile = nc.Dataset(file_netcdf, 'r' )
    hourspectra = ncfile['time'][:]
    hours       = ncfile['timeall'][:]
    level       = ncfile['level'][:]

    data={}
    for ij in datach:
        data[ij] =ncfile[ij][:]

    del ncfile

    return hourspectra,level,hours,data

# Object
def svttyp(case,sens):
   nbplus     = 0
   svt        = {}
   svt['FIRE']={'All'}
   svt['IHOP']={'IHODC','IHOP5'}
   svt['IHOPNW']={'All'}
   svt['AYOTTE']={'All'}
   svt['FIREWIND']={'All'}
   svt['FIRENOWIND']={'All'}
   if case in svt.keys():
     if sens in svt[case] or 'All' in svt[case]:
       nbplus=1
   return nbplus

def def_object(nbplus=0,AddWT=0):
  # Routine that discriminate object in terms of
  # conditional sampling m (m=2 -> 1 s.t.d.)
  # minimum volume Vmin (Vmin = )
  #thrs   = 1
  #thch   = str(thrs).zfill(2)
  #nbmin  = 100 #100 #1000
  objtyp = ['updr','down','down','down']
  objnb  = ['001' ,'001' ,'002' ,'003' ]
  if nbplus == 1:
     objnb = [str(int(ij)+3).zfill(3) for ij in objnb]
     
  WTchar=['' for ij in range(len(objtyp))]
  if AddWT  == 1:
    WTchar = ['_WT','_WT','_WT','_WT']
  
  typs = [field+'_SVT'+objnb[ij]+WTchar[ij] for ij,field in enumerate(objtyp)]
  print(typs)
  return typs, objtyp

def do_unique(tmp):
    tmp[tmp>0]=1
    return tmp

def delete_smaller_than(mask,obj,minval):
  sizes = ndimage.sum(mask,obj,np.unique(obj[obj!=0]))
  del_sizes = sizes < minval
  del_cells = np.unique(obj[obj!=0])[del_sizes]
  # new version
  ss        = obj.shape
  objf      = obj.flatten()
  ind       = np.in1d(objf,del_cells)
  objf[ind] = 0
  obj       = objf.reshape(ss)
  return obj

def do_delete2(objects,mask,nbmin,rename=True,clouds=None):
    nbmax   = np.max(objects)
    print(nbmax,nbmin)
    #time1 = time.time()
    objects = delete_smaller_than(mask,objects,nbmin)
    #time2 = time.time()
    #print('%s function took %0.3f ms' % ("delete smaller", (time2-time1)*1000.0))
    if clouds is not None:
        print('filter clouds not None')
        objects = delete_clouds(objects,clouds)

    #print np.max(objects),len(np.unique(objects))
    if rename :
        labs = np.unique(objects)
        objects = np.searchsorted(labs, objects)
    nbr = len(np.unique(objects))-1 # except 0
    print('\t', nbmax - nbr, 'objects were too small')
    return objects,nbr

def delete_clouds(obj,cld,min=1):
    #mask       = do_unique(deepcopy(obj))
    maskclouds = do_unique(deepcopy(cld))
    maskclouds *= obj
    del_cells  = np.unique(maskclouds[maskclouds!=0])
    print('del cells : ', del_cells)
    print('unique : ',np.unique(obj[obj!=0]))
    
    # Remove all object that have clouds
    ss        = obj.shape
    objf      = obj.flatten()
    ind       = np.in1d(objf,del_cells)
    objf[ind] = 0
    obj       = objf.reshape(ss)
    
    return obj

def find_nearest_neighbors(data, size=None):
    # From cloudmetrics
    # FIXME not sure if boxsize (periodic BCs) work if domain is not square
    tree = cKDTree(data, boxsize=size)
    dists = tree.query(data, 10)
    #print('dists ',dists.shape,dists)
    nn_dist = np.sort(dists[0][:, 1])
    return nn_dist

def neighbor_distance(cloudproperties, mindist=0):
    """Calculate nearest neighbor distance for each cloud.
       periodic boundaries

    Note: 
        Distance is given in pixels.

    See also: 
        :class:`scipy.spatial.cKDTree`:
            Used to calculate nearest neighbor distances. 

    Parameters: 
        cloudproperties (list[:class:`RegionProperties`]):
            List of :class:`RegionProperties`
            (see :func:`skimage.measure.regionprops` or
            :func:`get_cloudproperties`).
        mindist
            Minimum distance to consider between centroids.
            If dist < mindist: centroids are considered the same object

    Returns: 
        ndarray: Nearest neighbor distances in pixels.
    """
    centroids = [prop.centroid for prop in cloudproperties]
    indices   = np.arange(len(centroids))
    neighbor_distance = np.zeros(len(centroids))
    centroids_array = np.asarray(centroids)

    for n, point in enumerate(centroids):
        #print n, point
        # use all center of mass coordinates, but the one from the point
        mytree = sp.spatial.cKDTree(centroids_array[indices != n])
        dist, indexes = mytree.query(point,k=len(centroids)-1)
        #ball  =  mytree.query_ball_point(point,mindist)
        distsave=dist[dist>mindist]   
        #print distsave

        #if abs(centroids_array[indexes[0]][0]-point[0])>100.:
        #  print centroids_array[indexes[0]]
        #  print n,point#,[centroids_array[ij] for ij in indexes]

        neighbor_distance[n] = distsave[0]

    return neighbor_distance
