#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 16:52:24 2023

@author: fbrient
"""
import numpy as np
import Constants as CC
from scipy import integrate


def resiz(tmp): # should be removed by pre-treatment
    return np.squeeze(tmp)

def tryopen(vv,DATA):
    try:
        tmp = resiz(DATA[vv][:])
    except:
        print('Error in opening ',vv)
        tmp = None
    return tmp

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

def createnew(vv,DATA,var1D):
    vc   = {'THLM' :('THT','PABST','RCT'),
            'DIVUV':('UT',var1D[1]),\
            'RNPM' :('RVT','RCT') }
    [vc.update({ij:('THT','PABST',var1D[0])}) for ij in ['PRW','LWP','Reflectance']]
    
    tmp = tryopen(vv,DATA)

    data = []
    if vv in vc.keys() and tmp is None:
        data      = [tryopen(ij,DATA) for ij in vc[vv]]
        if vv == 'THLM':
            #thetal = theta - L/Cp Theta/T Ql
            tmp = tryopen('THLM',DATA)
            if tmp is None:
                if data[2] is None: # No clouds
                    data[2] = np.zeros(data[0].shape)
                tmp = data[0] -(\
                     (data[0]/tht2temp(data[0],data[1]))\
                    *(CC.RLVTT/CC.RCP)*data[2])
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
            #dy   = data[3][2]-data[3][1]; print(dy)
            tmpU = divergence(data[0], dx, axis=-1) # x-axis
            #tmpV = divergence(data[1], dy, axis=-2) # y-axis
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
                tmp = np.zeros((ss[1],ss[2]))
                zz  = np.repeat(np.repeat(data[2][ :,np.newaxis, np.newaxis],ss[1],axis=1),ss[2],axis=2)
                for  ij in range(len(data[2])-1):
                    tmp[:,:] += rho[ij,:,:]*RCT[ij,:,:]*(zz[ij+1,:,:]-zz[ij,:,:])
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

    return tmp


def findpbltop(typ,DATA,var1D,idx=None,offset=0.25):
    tmp     = createnew(typ,DATA,var1D)
    print('len ',len(tmp.shape))
    if len(tmp.shape)==2:
        temp   = np.nanmean(tmp,axis=(-1))
    else:
        temp    = np.nanmean(tmp,axis=(1,2,))
            
    THLMint = integrate.cumtrapz(temp)/np.arange(1,len(temp))
    #DT      = temp[:-1]-(THLMint+offset)
    # Modif
    DT      = temp[1:]-(THLMint+offset)
    idx     = np.argmax(DT>0)
    return idx
