#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 17:42:14 2023

@author: fbrient
"""
import netCDF4 as nc
import numpy as np
import os
import spectra
import tools0 as tl
import time
import cloudmetrics as cm
import makefigs as mf


def opensimu(Dim):
    if Dim=='2D':     # 2D
        file0 = 'EXP.1.VTIME.OUT.TTIME.nc'
        VTIME = 'V0001'
        model = 'FIRE2Dreal'
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
        #model,EXP = "IHOPNW","IHOP0"
        #model,EXP,VTIME = "FIREWIND","WDFor","V0301"
        #model,EXP,VTIME = "FIRE","Ls2x0","V0301"
        model,EXP,VTIME = "BOMEX","Ru0x0","V0301"
        
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
        if "BOMEX" in model:
            T0,TMAX,DT    = 8,8,1
            #if EXP=="Ru0NW":
            file0=file0.replace('OUT.','')
            file0=file0.replace('.nc','.nc4')
           
        path0 = path+model+'/'+EXP+'/'
        hours = np.arange(T0,TMAX+DT,DT)
        dt    = 1 # saving each hour
        
        name_xy = ['W_E_direction','S_N_direction','vertical_levels']
        var1D   = name_xy[::-1]
        # 3D field has properties axis such as
        # vertical_levels, S_N_direction, W_E_direction
        rmb     = False

    return model,EXP,VTIME,path0,file0,hours,dt,name_xy,var1D,rmb


#####
# This code aims to calculate and draw 
# -variance
# -skewness
# -spectrum analysis
# -length scale
# from of 2D-field outputs of the MNH model (v5.5.1)
# Update to use 3D field


####################
## User choices
####################
# 2D or 3D simualtions
Dim='3D'
# Make figures
makefigures=0
# Plot cloud in makefigures
pltcloud=True
# Plot Anomalies
pltanom=False
# Join graph of var1 and var2
joingraph=False
# Plot turbulent spectra
pltspectra=True
# Plot 2D FFT
plotFFT2D=False

Anom = 0; anomch=''
if pltanom:
    Anom = 1
    anomch='Anomaly of '

# Version MNH
Version="V5-5-1"
#Version="V5-7-0"

# Open simulation (2D or 3D)
model,EXP,VTIME,path0,file0,hours,dt,name_xy,var1D,rmb = opensimu(Dim)


file0 = file0.replace('EXP',EXP)
file0 = file0.replace('VTIME',VTIME)
file0 = path0+file0

# Variable to study
var1c  = 'RNPM'
var2c  = None

covar = None
#covar = var1c
#if covar is not None:
 #   var2c = None

varall  = ''.join([var1c,var2c]) if var2c is not None else var1c
if covar is not None:
    varall  = 'c'+''.join([var1c,var2c])
varall2 = varall

if pltanom:
    varall2+='_anom'
if joingraph:
    varall2+='_join'

# Infos figures
zminmax = tl.findextrema(model)

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
pathsave0+=Dim+"/";tl.mkdir(pathsave0)
pathsave0+=model+"/";tl.mkdir(pathsave0)
pathsave0+=EXP+"/";tl.mkdir(pathsave0)
pathsave1=pathsave0 # For final plots
pathsave0+=varall2+"/";tl.mkdir(pathsave0)
filesave0=pathsave0+'_'.join([varall2,EXP,VTIME])+'_TTIME'

if covar:
    pathsave2=pathsave1+varall+'/';tl.mkdir(pathsave2) # For spectra
else:
    pathsave2=pathsave1+var1c+'/';tl.mkdir(pathsave2) # For spectra
filesptr0=pathsave2+'spectra_'+'_'.join([varall,EXP,VTIME]) \
    +'_TTIME'+'_zALT_s'+'{0:02}'.format(smoothspectra)
filefft0=pathsave2+'fft2D_'+'_'.join([varall,EXP,VTIME]) \
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
        filefft1  = filefft0.replace('TTIME','{0:03}'.format(hour))
        print(file)
        fl_dia = nc.Dataset(file, 'r' )
        
        if ih==0:
            # Open dims
            ofdim = 1e-3
            ni=tl.ajax(np.array(fl_dia[name_xy[0]][:]),of=ofdim,rmb=rmb)
            nj=tl.ajax(np.array(fl_dia[name_xy[1]][:]),of=ofdim,rmb=rmb)
            level=tl.ajax(fl_dia[name_xy[2]][:],of=ofdim,rmb=rmb)
            
            delta_x = ni[1]-ni[0]
            if len(nj)>1:
                delta_y = nj[1]-nj[0]
            delta_z = level[1]-level[0]
            
            zx,xz = np.meshgrid(level,ni)
            
            NL      = len(level)
            NH      = len(hours)
            NHsp    = int(len(hours)/smoothspectra)
            
            # Init
            sig2,skew  = [np.zeros((NH,NL))*np.nan for ij in range(2)]
            lambda_max = np.zeros((Na,NHsp,NL))*np.nan
            lambda_fit = np.zeros((Na,NHsp,NL))*np.nan
            lambda_cm  = np.zeros((Na,NHsp,NL))*np.nan
            idxzi      = np.zeros(NH,dtype=int)
            wstar,tstar,thetastar = [np.zeros(NH) for ij in range(3)]

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
    
        # Open variables
        var   = tl.createnew(var1c,fl_dia,var1D)
        ofdim = tl.find_offset(var1c)
        print(var1c,ofdim,rmb)
        var   = tl.ajax(var,of=ofdim,rmb=rmb)
        
        # Second variable (optionnal)
        if var2c is not None:
            var2   = tl.createnew(var2c,fl_dia,var1D)
            var2  = tl.ajax(var2,tl.find_offset(var2c),rmb=rmb) #.T
            
        if covar is not None:
            var  = tl.anomcalc(var)
            var2 = tl.anomcalc(var2)
            var  = var*var2
    
        # PBL height    
        zi = level[idxzi[ih]]
        print('**** PBL height zi (km): ',zi)

        # Omega star (To remove non-turbulent layer)
        # Not updated to remove bounds
        wstar[ih]     = tl.createnew('Wstar',fl_dia,var1D,idxzi=idxzi[ih])
        tstar[ih]     = (1000.*zi)/wstar[ih] # secondes
        thetastar[ih] = tl.createnew('Thetastar',fl_dia,var1D,idxzi=idxzi[ih])
#        print(ih,wstar[ih],tstar[ih],thetastar[ih])
#        stop

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
                RCT = tl.ajax(fl_dia['RCT'][:].squeeze().T,of=1000.,rmb=rmb)
            varplot, varplot2 = var,var2
#            print('var,var2 ',var.shape,var2.shape)
#            stop
            if pltanom:
                varplot=tl.anomcalc(varplot)
                if varplot2 is not None:
                    varplot2=tl.anomcalc(varplot2) #.T) #.T
                    
            title=[anomch+tl.findname(var1c)+' ('+tl.findunits(fl_dia,var1c)+')']
            if var2c is not None:
                title+=[anomch+tl.findname(var2c)+' ('+tl.findunits(fl_dia,var2c)+')']
                varplot2= varplot2.T
            fig,ax   = mf.plot2D(varplot.T,xz,zx,filesave,
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
        NS    = int(np.floor(NX/2))#-1 #round(NX/2)
        print('NX,NS: ',NX,NS)
        
        if Dim=='2D' and ih==0:   
            spec_log_avg=np.zeros((Na,NL,NS-1)) 
            nb_avg      =np.zeros(NL) 
            NX,NY       =1,1
            axis0       =(0,2)
        if Dim=='3D':
            spec_log_avg  =np.zeros((Na,NL,NS-1)) 
            spec_log2D_avg=np.zeros((NL,NS)) 
            nb_avg        =np.zeros(NL) # not using smoothspectra
            axis0         =(1,2)
        
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
            sig2[ih,zz] = tl.variance(var0all)
            skew[ih,zz] = tl.skewness(var0all)
            
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
                    
                    # v2 : Me (Erreur in spectra?)
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
                # New way to compute spectra, but it is maybe related to variance
                # Impossible de find variance
                # Use cloud metrics instead
                #k_v2D,SPECTRE_V2D,VAR_V2D,spec_log2D_avg[zz,:] =\
                #    spectra.spectra2D(var0,delta_x)
                
            
                # spec_log2D from cloudmetrics  
                k_v2D, psd_1d_rad, psd_1d_azi = cm.scalar.compute_spectra(
                    var0,
                    dx=delta_x,
                    periodic_domain=True,
                    apply_detrending=False,
                    window=None,
                )
                
                # Plot 2D Fourier
                if plotFFT2D:
                    filefft   = filefft1.replace('ALT','{:4.1f}'.format(level[zz]*1000.))
                    mf.plot2Dfft(var0,delta_x,filesave=filefft)
                
                # Save spec_log2D
                spec_log2D_avg[zz,:] = psd_1d_rad
                
                # Test FB: Divide par variance
                spec_log2D_avg[zz,:] = psd_1d_rad/np.var(var0)
                
                #spec=spec_log2D_avg[zz,:]
                
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
                    infoch    = [zz,tl.findname(var1c,dash=True)]
                    #NIJ=spec_log_avg.shape[0]
                    spec_plot = np.zeros((Na,NS-1))
                    spec_fit  = np.zeros((Na,NS-1))
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
                                spectra.plot_spectra(k_v2D,spec_plot2D,filesptr,\
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
                            
                            variance2D = np.trapz(spec_plot2D, x=k_v2D)
                            print('variance2D ',variance2D)

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
            spec_log_avg=np.zeros((Na,NL,NS-1))
            
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
    lvl = tl.findlvl(lamb,model,varname=var1c)
    
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

