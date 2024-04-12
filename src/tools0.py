#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 16:52:24 2023

@author: fbrient
"""
import numpy as np
import Constants as CC
from scipy import integrate
import netCDF4 as nc
from netCDF4 import Dataset
import gc

def repeat(zz,ss):
    #if len(ss)==1:
    zz  = np.repeat(zz[ :,np.newaxis],ss[0],axis=1)        
    if len(ss)==2:
        zz  = np.repeat(zz[ :,:, np.newaxis],ss[1],axis=2)
    return zz

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

def createnew(vv,DATA,var1D,idxzi=None):
    vc   = {'THLM' :('THT','PABST','RCT'),
            'Wstar':('WT','RVT','RCT','PABST',var1D[0]),
            'THV'  :('THT','RVT','RCT'),
            'DIVUV':('UT',var1D[1]),\
            'RNPM' :('RVT','RCT') }
    [vc.update({ij:('THT','PABST',var1D[0])}) for ij in ['PRW','LWP','Reflectance']]
    
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
        if vv == 'DIVUV':
            # DIVUV = DU/DX + DV/DY
            dx   = data[1][2]-data[1][1]; print(dx)
            tmpU = divergence(data[0], dx, axis=-1) # x-axis
            tmp  = tmpU #+ tmpV
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
                zz = repeat(data[2],(ss[1],ss[2]))

                if len(ss)==3.:
                    tmp = np.zeros((1,ss[1],ss[2]))
                    for  ij in range(len(data[2])-1):
                        tmp[0,:,:] += rho[ij,:,:]*RCT[ij,:,:]*(zz[ij+1,:,:]-zz[ij,:,:])
                else:
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
        if vv == "Wstar":
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
            wthv = 0.
            for ij,zz1 in enumerate(zz):
                tmp  = WT[ij,:]*(THV[ij,:]-np.nanmean(THV[ij,:]))
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
            print('tmp ',tmp)
    return tmp




def findpbltop(typ,DATA,var1D,idx=None,offset=0.25):
    tmp     = createnew(typ,DATA,var1D)
    print('len ',len(tmp.shape))
    if len(tmp.shape)==2:
        temp   = np.nanmean(tmp,axis=(-1))
    else:
        temp   = np.nanmean(tmp,axis=(1,2,))
            
    THLMint = integrate.cumtrapz(temp)/np.arange(1,len(temp))
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