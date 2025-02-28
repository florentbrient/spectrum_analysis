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
from matplotlib.ticker import LogLocator, LogFormatterSciNotation
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter1d
import sys
from scipy import ndimage
from scipy.spatial import cKDTree


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
    
    zz = tryopen(var1D[0],DATA)
    #print(zz)
    #THLMint = integrate.cumtrapz(temp)/np.arange(1,len(temp))
    #THLMint = integrate.cumulative_trapezoid(temp)/np.arange(1,len(temp))
    # Correction
    THLMint = integrate.cumulative_trapezoid(temp,zz,initial=0)/(zz-zz[0])
    #print(THLMint)
    
    #DT      = temp[:-1]-(THLMint+offset)
    # Modif
    DT      = temp-(THLMint+offset)
    idx     = np.argmax(DT>0)
    #print(DT)
    
    # Plot results
    # plt.figure(figsize=(6, 6))
    # plt.plot(temp, zz, 'ro-', label="Raw Temperature")
    # plt.plot(THLMint, zz, 'bo-', label="Altitude-Weighted Cumulative Avg Temp")
    # plt.axhline(y=zz[idx],color='k')
    # plt.xlabel("Temperature (°C)")
    # plt.ylabel("Altitude (m)")
    # plt.show()
    # sys.exit()
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

# Smooth line
def smooth(y,sigma=2):
    return gaussian_filter1d(y, sigma=sigma)

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



################ Objects #####################
# Object
def svttyp(case,sens):
   nbplus     = 0
   svt        = {}
   svt['FIRE3D']={'All'}
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
    maskclouds = do_unique(cld)
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
        mytree = cKDTree(centroids_array[indices != n])
        dist, indexes = mytree.query(point,k=len(centroids)-1)
        #ball  =  mytree.query_ball_point(point,mindist)
        distsave=dist[dist>mindist]   
        #print distsave

        #if abs(centroids_array[indexes[0]][0]-point[0])>100.:
        #  print centroids_array[indexes[0]]
        #  print n,point#,[centroids_array[ij] for ij in indexes]

        neighbor_distance[n] = distsave[0]

    return neighbor_distance




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


def plot_time(time,Gamma,
              lambdaIn=None,
              namex='Aspect Ratio (-)',
              namefig='test',title=None,
              xsize=(14,10),fts=18,lw=2.5):
    
    # Start plot
    fig, ax = plt.subplots(1,1,figsize=xsize)
    
    # Modify time (by deltaT/2)
    time = time+(time[1]-time[0])/2.
    
    # Plot all Gamma
    keys = Gamma.keys()
    for ij,key in enumerate(keys):
        ax.plot(time,Gamma[key],lw=lw,label=key)
    if lambdaIn is not None:
        color='green'
        ax.plot(time,lambdaIn,color=color,ls='--',lw=lw,label=r'$\epsilon_{in}$')
    
    # Wood and Hartmann 2006
    #k1,k2 =30.,40.
    #ax.axhspan(k1, k2, color="grey", alpha=0.3)  # Adjust alpha for transparency
    
    # Legends
    ax.legend(title=None,shadow=True,numpoints=1,loc=2,
               bbox_to_anchor=(0.0,0.9),
               fontsize=12,title_fontsize=20)
    
    if title is not None:
        plt.title(title)

    ax.set_xlabel('Time (hours)',fontsize=fts)    
    ax.set_ylabel(namex,fontsize=fts)
    ax.tick_params(axis='both', labelsize=fts)
    adjust_spines(ax,['left', 'bottom'])
        
    # Save figure
    namefig+='.png'
    savefig2(fig, namefig)
    plt.close('all')
    
    return None
    
    

def plot_flux(k,E,PI=None,kPBL=None,kin=None,Euv=None,\
              y1lab='xlab',y2lab='ylab',\
              normalized=False,logx=True,\
              namefig='namefig',plotlines=False,\
              smooth=None,\
              xsize=(12,10),fts=18,lw=2.5):
    
    # Start plot
    fig, ax1 = plt.subplots(1,1,figsize=xsize)
    ax2 = False
    if PI is not None:
        xsize=(12,16)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=xsize)

    # By default
    namex = 'Wavenumber'

    # If multiple time
    ss = np.shape(E)
    k  = np.repeat(k.data[np.newaxis, :], ss[0], axis=0)  # Adds new axis first
    if ss[0] > 1 or normalized:
        normalized=True
        # Normalisation of k by kPBL
        for ij in range(ss[0]):
            k[ij,:]=k[ij,:]/kPBL[ij]
            namex = r"$k/k_h$"

    # colors
    if ss[0] == 1:
        colors = ['b']
        labels = [namefig.split('_')[-1]]
    else:
        # Create colormap and labels
        # def twilight_shifted
        colors = cm.Spectral(np.linspace(0, 1, ss[0]))
        dt     = 1
        labels = ['t+'+str(ij+dt).zfill(2)  for ij in range(ss[0])] 
    
    # Plot Spectra and Energy
    for ij in range(ss[0]):
        ax1.semilogx(k[ij,:],E[ij,:],lw=lw, color=colors[ij], label=labels[ij])
        if smooth is not None:
            ax1.semilogx(k[ij,:],smooth[ij,:],lw=lw-1, color='r',ls='--')
        if ax2:
            ax2.semilogx(k[ij,:],PI[ij,:],lw=lw, color=colors[ij], label=labels[ij])

    if logx:
        ax1.set_yscale("log")

    if plotlines and logx:
        k0max=k.max()/2
        k0min=k.max()/4
        k1max=k.max()/4
        k1min=k.max()/10
        k0 = np.linspace(k0min,k0max,1000)
        k1 = np.linspace(k1min,k1max,1000)
        
        # scale to plot slopes - calculate offset
        offset = kPBL[-1]**(-5/3.) # what the slope see
        if normalized:
            offset = 1.
        mean   = E[-1,near(k[-1,:],kPBL[-1])] # what the real y-axis plot
        k1scale = mean/offset #3e-2
        k0scale = mean/kPBL[-1]**(-3.)#*(k1min/k0min) #3e-3
        print(mean,offset,k1scale)
        
        # pentes en -5/3
        ax1.plot(k1,k1scale*k1**(-5/3.),color='gray',linewidth=3,linestyle='--',label=r'$\mathbf{k^{-5/3}}$')
        # pentes en -3
        #ax1.plot(k0,k0scale*k0**(-3),color='gray',linewidth=3,linestyle='-',label=r'$\mathbf{k^{-3}}$')
        
        # Legends
    ax1.legend(title=None,shadow=True,numpoints=1,loc=2,
               bbox_to_anchor=(1.0,1.0),
               fontsize=12,title_fontsize=20)
        
    #ax1.set_title('Original Signal')
    ax1.set_ylabel(y1lab,fontsize=fts)
    if logx:
        ax1.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=10))
        ax1.yaxis.set_major_formatter(LogFormatterSciNotation())
    if ax2:
        ax2.set_ylabel(y2lab,fontsize=fts)
        ax2.axhline(y=0,color='k')
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-3, 3))  # Uses scientific notation when needed
        ax2.yaxis.set_major_formatter(formatter)
        ax2.yaxis.get_offset_text().set_fontsize(12)  # Adjust font size as needed
    
    # Second axis (for Spectra only)
    second_axis='bottom'
    if second_axis == 'top':
        func = z2k; namex2 = r'$\mathbf{\lambda \;(m)}$'
        if normalized:
            namex2 = r'$\mathbf{\lambda/H \;(-)}$'
            func=(lambda x: 1 / x)
        secax = ax1.secondary_xaxis('top', functions=(func, func))
        secax.set_xlabel(namex2,fontsize=fts,labelpad=15)
        for label in secax.get_xticklabels():
            label.set_fontsize(fts)
            #label.set_fontweight('bold')
        secax.tick_params('both', length=8, width=2, which='major', direction='out')
        secax.tick_params('both', length=4, width=1, which='minor', direction='out')
    else:
        func = z2k; namex2 = r'$\lambda \;(m)}$'
        if normalized:
            namex2 = r'$\mathbf{\lambda/H \;(-)}$'
            func=(lambda x: 1 / x)
        secax = ax1.secondary_xaxis('bottom', functions=(func, func))
        secax.spines["bottom"].set_position(("data", -1))
        secax.set_xlabel(namex2,fontsize=fts,labelpad=-40)
        for label in secax.get_xticklabels():
            label.set_fontsize(fts)
            #label.set_fontweight('bold')
        secax.tick_params(axis="x", length=8, width=2, which='major', direction='in',pad=-30, colors="red")
        secax.tick_params(axis="x", length=4, width=1, which='minor', direction='in',pad=-30, colors="red")
        secax.xaxis.label.set_color("red")
        
        # It doesn't work
        # Add a second set of tick labels below the x-axis
        # secax = ax1.twiny()  # Create a twin x-axis (it will align automatically)
        # #secax.set_scale("log")
        # secax.set_xlim(ax1.get_xlim())  # Ensure both axes align
        
        # # Define tick locations for wavelength (bottom)
        # wavelength_ticks = np.logspace(-4, -1, num=4)  # Example: 10^2, 10^3, 10^4
        # ax1.set_xticks(wavelength_ticks)
        # ax1.xaxis.set_major_locator(LogLocator(base=10.0))  # Log ticks

        # # Convert wavelength ticks to wave number for top axis
        # wavenumber_ticks = z2k(wavelength_ticks[::-1])  # Reverse order for correct alignment
        # secax.set_xticks(wavenumber_ticks)
        # secax.xaxis.set_major_locator(LogLocator(base=10.0))  # Log ticks
        # secax.set_xticklabels([f"{k:.1e}" for k in wavenumber_ticks])  # Scientific notation
        
        # # Set axis labels
        # secax.set_xlabel(r"Wave Number ($k = 2\pi/\lambda$) [1/nm]", fontsize=12, labelpad=10)

        # Add grey shaded area between aspect 30 and 40 (See Wood and Hartmann, 2006)
        # If normalized, Axis is k/kH -> Pour z/H = 40 --> k/kH = 1/40
        if normalized:
            k30 = 1./30.
            k40 = 1./40.
            ax1.axvspan(k40, k30, color="grey", alpha=0.3)  # Adjust alpha for transparency
    
    xline=0
    if normalized:
        xline = 1
    else:
        if kPBL is not None :
            xline = kPBL
    
      
    if not logx:
        ax1.axhline(y=0,color='k')
    # Format x-axis and y-axis in scientific notation
    #formatter.set_powerlimits((-3, 3))  # Defines range for scientific notation
   
    axall = [ax1]
    if ax2:
        axall+=[ax2]
    for ax in axall:
        if xline is not None:
            ax.axvline(x=xline,color='k',ls='--')
        if kin is not None:
            ax.axvline(x=kin,color='g',ls='--')
        
        ax.set_xlabel(namex,fontsize=fts)
        ax.tick_params(axis='both', labelsize=fts)
        adjust_spines(ax,['left', 'bottom'])
     
    # useful?
    #plt.tight_layout()
    # Save figure
    namefig=namefig.replace('XXXX',y2lab)+'.png'
    savefig2(fig, namefig)
    plt.close('all')
    
    return None

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





