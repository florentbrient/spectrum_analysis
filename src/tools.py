#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:23:37 2025

Toolbox for spectral analysis

@author: fbrient
"""

from collections import OrderedDict
import numpy as np
from scipy import integrate
import Constants as CC
from netCDF4 import Dataset
import os
import xarray as xr
from glob import glob
import pylab as plt
from PIL import Image
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import LogFormatterSciNotation
import matplotlib.cm as cm


# Read txt file for informations
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

def read_netcdfs(files, dim):
    # glob expands paths with * to a list of files, like the unix shell
    paths = sorted(glob(files))
    datasets = [xr.open_dataset(p) for p in paths]
    combined = xr.concat(datasets, dim)
    return combined


# Read dimensions informations in the NetCDF file
def dimensions(DATA,var1D):
    # Dimensions
    data1D,nzyx,sizezyx = [OrderedDict() for ij in range(3)]
    for ij in var1D:
      data1D[ij]  = DATA[ij][:] #/1000.
      nzyx[ij]    = data1D[ij][1]-data1D[ij][0]
      sizezyx[ij] = len(data1D[ij])

    #xy     = [data1D[var1D[1]],data1D[var1D[2]]] #x-axis and y-axis
    nxny   = nzyx[var1D[1]]*nzyx[var1D[2]] #km^2
    ALT    = data1D[var1D[0]]
    dz     = [0.5*(ALT[ij+1]-ALT[ij-1]) for ij in range(1,len(ALT)-1)]
    dz.insert(0,ALT[1]-ALT[0])
    dz.insert(-1,ALT[-1]-ALT[-2])
    nxnynz = np.array([nxny*ij for ij in dz]) # volume of each level
    nxnynz = np.repeat(np.repeat(nxnynz[:, np.newaxis, np.newaxis]
                                 , sizezyx[var1D[1]], axis=1), sizezyx[var1D[2]], axis=2)
    dx     = nzyx[var1D[1]]
    dy     = nzyx[var1D[2]]

    return nxnynz,data1D,dx,dy,dz


# Find Boundary-layer top (zi)
def findpbltop(typ,DATA,var1D,idx=None,offset=0.25):
    tmp     = createnew(typ,DATA,var1D)
    #print('len ',len(tmp.shape))
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

# Convert altitude to wavenumber
def z2k(z):
    return 2*np.pi/z

# Squeez data
def resiz(tmp): # should be removed by pre-treatment
    return np.squeeze(tmp)

# Open a variable (try or error)
def tryopen(vv,DATA):
    try:
        tmp = resiz(DATA[vv][:])
    except:
        print('Error in opening ',vv)
        tmp = None
    return tmp

# Make Dir
def mkdir(path):
   try:
     os.mkdir(path)
   except:
     pass

# Fin nearest point
def near(array,value):
    idx=(abs(array-value)).argmin()
    return idx

# Calculate the anamoly relative to the horizontal mean
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

#Repeat axis in one additional dimension
def repeat(zz,ss):
    #if len(ss)==1:
    zz  = np.repeat(zz[ :,np.newaxis],ss[0],axis=1)        
    if len(ss)==2:
        zz  = np.repeat(zz[ :,:, np.newaxis],ss[1],axis=2)
    return zz

# Save NetCDF file (for spectra)
def writenetcdf(file_netcdf,data_dims,data):
    
    # Open or create file
    try: ncfile.close()  # just to be safe, make sure dataset is not already open.
    except: pass

    # Check if file exists and remove it
    if os.path.exists(file_netcdf):
        os.remove(file_netcdf)


    ncfile = Dataset(file_netcdf,mode='w',format='NETCDF4_CLASSIC') 
    print('Your NetCDF file is ',ncfile)

    # Creating dimensions
    kv_dim  = ncfile.createDimension('kv',len(data_dims[0]))   # wavenumber kv
    kvazi_dim  = ncfile.createDimension('kvazi', len(data_dims[1])) # 72 (azimutahl spectra)
    level_dim = ncfile.createDimension('level',len(data_dims[2])) # level axis
    for dim in ncfile.dimensions.items():
        print(dim)
    
    ncfile.title='File: '+file_netcdf
    print(ncfile.title)
    
    # Creating variables
    kv = ncfile.createVariable('kv', np.float64, ('kv',))
    kv.units = 'rad/km'
    kv.long_name = 'Wavenumber dims'
    kvazi = ncfile.createVariable('kvazi', np.float64, ('kvazi',))
    kvazi.units = 'Angle'
    kvazi.long_name = 'kv azimuthal'
    level = ncfile.createVariable('level', np.float64, ('level',))
    level.units = 'kilometers'
    level.long_name = 'Altitude'  
    
    # Writing data
    kv[:]    = data_dims[0]
    kvazi[:]  = data_dims[1]
    level[:] = data_dims[2]
    for key in data.keys():
        if key=='E1da': units=('kvazi','level')
        elif  key=='var': units=('level')
        else: units=('kv','level')
        
        print('units ',units)
        tmp = ncfile.createVariable(key,np.float64,units) # note: unlimited dimension is leftmost
        tmp.units = '' # degrees Kelvin
        tmp.standard_name = key # this is a CF standard name
        tmp[:]   = data[key]
    ncfile.close()

################ Spectra calculation #####################


# Calculate gradients in the 3 dimensions
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


def checkvariance(k,E,field):
    varspec = np.trapz(E,x=k)
    varfield= np.var(field)
    print('variance Spectra ',varspec)
    print('variance Field ',varfield)
    perc=100*varspec/varfield
    sentence='The spectra represent {perc}% of the variance of the field'
    print(sentence.format(perc=perc))
    return None


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
    
    # Compute low-pass filtered (2D)
    print('** start U_hat **')
    U_hat = np.fft.fft2(U)
    
    # Apply the filter in the frequency domain
    #U_hatf = U_hat * filter_hat
    
    #pltcont = False
    #levels = np.linspace(U.min(),U.max(),10)
    
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
        
        # I comment the second filter
        # Therefore the filter input is not used !
        # Uf2_k = None    
        # if filter>0:
        #     # Filter 2 : Create the Gaussian low-pass filter
        #     filter_hat = np.exp(-(k**2) / (2 * k_idx**2))
        #     #plt.plot(filter_hat)
        #     U_hatf2 = U_hat * filter_hat
        #     Uf2 = np.fft.ifftn(U_hatf2)
        #     if idx==0: # Save
        #         Uf2_k = np.array(len(kk)*[np.zeros(Uf2.shape)])
        #     Uf2_k[idx,:,:]=Uf2
        
        # if pltcont:
        #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        #     cs1 = ax1.contourf(U,levels=levels)
        #     plt.colorbar(cs1, ax=ax1)
        #     cs2 = ax2.contourf(Uf,levels=levels)
        #     #plt.tight_layout()
        #     plt.colorbar(cs2, ax=ax2)
        #     plt.show()
               
        #     if filter>0:
        #         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        #         cs1 = ax1.contourf(U,levels=levels)
        #         plt.colorbar(cs1, ax=ax1)
        #         cs2 = ax2.contourf(Uf2,levels=levels)
        #         #plt.tight_layout()
        #         plt.colorbar(cs2, ax=ax2)
        #         plt.show()
    
    del Uf,U_hat,U_hatf
    #gc.collect()
    return kk,Uf_k #,Uf2_k


################ Figures #####################

# Convert to np array
def fig_to_np_array(fig):
    """Convert a Matplotlib figure to a NumPy array."""
    fig.canvas.draw()  # Draw the canvas to update it
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return buf.reshape(h, w, 3)

# Save figure
def savefig2(fig, path):
    #Image.fromarray(mplfig_to_npimage(fig)).save(path)
    np_image = fig_to_np_array(fig)
    Image.fromarray(np_image).save(path)
    
# adjust spines in figures
def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 0))  # outward by 10 points
            #print(dir(spine))
            # FB : Need to update with Python3
            #spine.set_smart_bounds(True)
        else:
            spine.set_color('none')  # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])

def plot_flux(k,E,PI=None,kPBL=None,Euv=None,\
              y1lab='xlab',y2lab='ylab',\
              normalized=False,\
              namefig='namefig',plotlines=False,\
              xsize=(12,16),fts=18,lw=2.5):
    
    # Start plot
    fig, ax1 = plt.subplots(1,1,figsize=xsize)
    ax2 = False
    if PI is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=xsize)

    # If multiple time
    ss = np.shape(E)
    k  = np.repeat(k.data[np.newaxis, :], ss[0], axis=0)  # Adds new axis first
    if ss[0] > 1 or normalized:
        normalized=True
        # Normalisation of k by kPBL
        for ij in range(ss[0]):
            print(ij,k.shape,kPBL.shape)
            k[ij,:]=k[ij,:]/kPBL[ij]

    # colors
    if ss[0] == 1:
        colors = ['b']
        labels = ['All']
    else:
        # Create colormap and labels
        colors = cm.twilight_shifted(np.linspace(0, 1, ss[0]))
        labels = ['t+'+str(ij).zfill(2)  for ij in range(ss[0])] 
    
    # Plot Spectra and Energy
    for ij in range(ss[0]):
        ax1.loglog(k[ij,:],E[ij,:],lw=lw, color=colors[ij], label=labels[ij])
        if ax2:
            ax2.semilogx(k[ij,:],PI[ij,:],lw=lw, color=colors[ij], label=labels[ij])

#    if Euv is not None:
#        ax1.loglog(k,Euv,'b--')

    if plotlines:
        k0max=k.max()
        k0min=k0max/2
        k1max=k0max/2
        k1min=k0max/8
        k0 = np.linspace(k0min,k0max,1000)
        k1 = np.linspace(k1min,k1max,1000)
        
        k0scale = 3e-3
        k1scale = 3e-2
        if normalized:
            k0scale,k1scale=1,1
        # pentes en -5/3
        ax1.plot(k1,k1scale*k1**(-5/3.),color='gray',linewidth=3,linestyle='--',label=r'$\mathbf{k^{-5/3}}$')
        # pentes en -3
        ax1.plot(k0,k0scale*k0**(-3),color='gray',linewidth=3,linestyle='-',label=r'$\mathbf{k^{-3}}$')
        
        # Legends
        ax1.legend(title=None,shadow=True,numpoints=1,loc=2,
                             bbox_to_anchor=(0.1,0.2),
                             fontsize=12,title_fontsize=20)
        
    #ax1.set_title('Original Signal')
    ax1.set_ylabel(y1lab,fontsize=fts)
    if ax2:
        ax2.set_ylabel(y2lab,fontsize=fts)
        ax2.axhline(y=0,color='k')
    
    xline=0
    if normalized:
        xline = 1
    elif kPBL is not None :
        xline = kPBL
    if xline is not None:
        ax1.axvline(x=xline,color='k',ls='--')
        if ax2:
            ax2.axvline(x=xline,color='k',ls='--')
        
    # Format x-axis and y-axis in scientific notation
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))  # Defines range for scientific notation
   
    axall = [ax1]
    if ax2:
        axall+=[ax2]
    for ax in axall:
        #ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        ax.set_xlabel('Wavenumber',fontsize=fts)
        ax.tick_params(axis='both', labelsize=fts)
        adjust_spines(ax,['left', 'bottom'])
     
    # Save figure
    namefig=namefig.replace('XXXX',y2lab)+'.png'
    savefig2(fig, namefig)
    plt.close('all')
    
    return None


################ MESO-NH #####################

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

# Create new variables in Meso-NH
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





