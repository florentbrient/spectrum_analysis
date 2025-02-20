#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 12:03:52 2024

Goal: Find enegy injection using

@author: fbrient
"""

import numpy as np
import netCDF4 as nc
import tools as tl
from collections import OrderedDict
from copy import deepcopy
import pylab as plt
import matplotlib as mpl
from skimage import measure
import cloudmetrics as cm
#import scipy.spatial.distance as sd
from scipy.stats.mstats import gmean
import sys


def plot_figure(datavar,xy,
                var="WT",
                dataobj=None,
                title=None,namefig='test',
                xsize=(12,9),fts=18,lw=2):
    # Infos for the figure
    mtyp    = 'Mean'
    infofig = tl.infosfigures(prefix,var,mtyp=mtyp)
    levels  = infofig['levels']
#    zminmax = infofig['zminmax']
    xx, yy  = np.meshgrid(xy[0],xy[1])
    cmap    = infofig['cmap'][mtyp]
    norm    = None # default
    nmin,nmax = np.nanmin(datavar),np.nanmax(datavar)
    if levels is not None:
        nmin,nmax = np.nanmin(levels),np.nanmax(levels)
    if (np.sign(nmin)!=np.sign(nmax)) and nmin!=0:
        cmap  = infofig['cmap']['Anom']
        norm = mpl.colors.Normalize(vmin=nmin, vmax=abs(nmin))
        
    # Start plot 
    fig    = plt.figure(figsize=xsize)
    ax     = fig.add_subplot(111)
    CS     = ax.contourf(xx,yy,datavar,levels=levels,cmap=cmap,norm=norm,extend='both')
    
    if dataobj is not None:
        CS2   = ax.contour(xx,yy,dataobj,colors='k',linewidths=lw/6.)
    
    #Colorbar
    cbar   = plt.colorbar(CS,)
    cbar.ax.tick_params(labelsize=fts)
    
    # Title
    if title is not None:
        plt.title(title)
    
    # Save figure
    namefig+='.png'
    tl.savefig2(fig, namefig)
    plt.close('all')
    
    return None

# Path of the file
testing =False
if testing:
    path0="/home/fbrient/GitHub/objects-LES/data/"
    case    = 'IHOPNW'
    sens    = 'IHOP0'
    prefix  = '006'
    vtype   = 'V0001'
    sens2   = sens
    
    pathout = "../figures/zview/"
    tl.mkdir(pathout)
    pathout+= case+'/'
    #pathout+= prefix+'/'
    tl.mkdir(pathout)
    
    # The Netcdf file
    path    = path0+case+'/'+sens+'/'
    file    = 'sel_'+sens2+'.1.'+vtype+'.OUT.'+prefix+'.nc'
    file    = path+file
else:
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
    pathout=info_dict['pathfig']
    #if 'pathfig' in info_dict.keys():
    #    pathfig0=info_dict['pathfig']
    #pathfig0= pathfig0+ivar['vtyp']+'/'

    file=path+file

    tab = file.split('/')[-1].split('.')
    prefix,vinfo, tinfo = tab[0],tab[2],tab[4]

#case,sens,sens2,prefix,vtype = 'FIRE3D','FI1024','FIR1k','020','V0001'
#file = "/home/fbrient/GitHub/objects-LES/data/FIRE3D/FI1024/sel_FIR1k.1.V0001.OUT.020.nc"



# Open the netcdf file
DATA    = nc.Dataset(file,'r')


# Dimensions
#var1D  = ['vertical_levels','S_N_direction','W_E_direction'] #Z,Y,X
var1D = ['level','nj','ni'] #Z,Y,X

namez  = var1D[0]
data1D,nzyx,sizezyx = [OrderedDict() for ij in range(3)]
for ij in var1D:
  data1D[ij]  = DATA[ij][:]/1000.
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
dx     = nzyx[var1D[-1]]

# Find Boundary-layer height (zi)
inv               = 'THLM'
threshold         = 0.25
#idxzi,toppbl,grad = tl.findpbltop(inv,DATA,var1D,offset=threshold)
idxzi = tl.findpbltop(inv,DATA,var1D,offset=threshold)

################ Figures without objects ##############################
# Dir for figures
#namefig0=pathout+'zview_{VVV}_{TIME}_idx{ZZZ}'
namefig0=pathout+'_'.join(['zview','{VVV}',prefix,vinfo,tinfo,'idx{ZZZ}'])

idxplot = [2,idxzi]

keys = DATA.variables.keys()
for key in keys:
    print('key ',key)
    data = tl.resiz(DATA[key])
    print(data.shape)
    if len(data.shape)==3.:
        for idx in idxplot:
            dataplot = data[idx,:,:]
            title = key+' (z='+str(ALT[idx])+' km)'
            namefig = namefig0.format(VVV=key,ZZZ=str(idx))
            plot_figure(dataplot,xy,
                        var=key,
                        title=title,
                        namefig=namefig)




################ Objects ##############################
plotobj = False
if plotobj:
    ################
    # Find objects #
    ################
    
    # Name of tracer in the file
    nbplus = tl.svttyp(case,sens) #1
    # Object based on tracer concentration AND vertical velocity?
    AddWT  = 1
    # Name of objects
    typs, objtyp = tl.def_object(nbplus=nbplus,AddWT=AddWT)
    # Select only updraft and downdraft
    typs   = [ij for ij in typs if 'updr' in ij or '006' in ij or '003' in ij]
    #typs   = [ij for ij in typs if 'updr' in ij]
    
    # Threshold for conditional sampling
    thrs   = 2 # 2 by default
    thch   = str(thrs).zfill(2)
    # Vmin definition
    minchar = 'volume' #unit
    if minchar == 'volume':
       vmin   = 0.02 # by default (km^3)
       suffixmin = '_vol'+str(vmin)
    
    # Compute objects once
    dataobjs = {}
    nameobjs = []
    mask     = []
    for typ in typs:
      print(typ)
      nameobj   = typ+'_'+thch
      try:
        dataobj   = DATA[nameobj][:]
    
        tmpmask = tl.do_unique(dataobj)*nxnynz
    
        dataobjs[nameobj],nbr  = tl.do_delete2(dataobj,tmpmask,\
                vmin,rename=True,\
                clouds=None)
        del tmpmask         
        mask.append(tl.do_unique(deepcopy(dataobjs[nameobj])))
      except:
        print('Error')
        dataobjs[nameobj] = None
        
      nameobjs += [nameobj] #updr_SVT001_WT_02
            
    
    
    
    # Variables of interest
    var='WT'
    data = DATA[var]
    covar = True # calculate covariance
    if covar:
        var2 = 'WT'
        # Calculate Var1'*Var2' (exemple W'Theta')
        data=tl.anomcalc(data)*tl.anomcalc(DATA[var2])
    
    # Z of interest (zi/2 for instance)
    idx = int(idxzi/4)
    
    # Select objects in 2D?
    sel2D = True; relab = True
    vmin  = 80 # in pixels !
    
    
    keys = dataobjs.keys()
    for key in keys:
        print(key)
        obj = dataobjs[key][idx,:,:]
        tmp = data[idx,:,:]
        
        if sel2D:
            if relab: # with new labels
              obj      = measure.label(obj, connectivity=2)
            plt.contourf(obj);plt.colorbar();plt.show()
            objmask = tl.do_unique(deepcopy(obj))
            obj,nbr = tl.do_delete2(obj,objmask,vmin,rename=True)
            plt.contourf(obj);plt.colorbar();plt.show()
            print('nbr  ',key,nbr)
            
        ##### Calculate indexes
        cloudprop       = measure.regionprops(obj)
        pos = [prop.centroid for prop in cloudprop]
        
        # SCAI and mean distance between objects
        output=cm.objects.scai(obj,\
                               periodic_domain=False,\
                               return_nn_dist=True,\
                               dx=dx)
        print('Mean geometric neighbour distance between objects (km)')
        print(output[1])
        
        # Check
        #dist = sd.pdist(pos)
        #print('CHECK D0 : ',gmean(dist))
        
        # Mean distance with nearest neightbor
        nd = tl.neighbor_distance(cloudprop,mindist=0)
        print('Mean geometric nearest neighbour distance (km)')
        print(gmean(nd)*dx)
        
        # Check with cloud metrics
        #nearest =  cm.utils.find_nearest_neighbors(obj)
        #nearest =  tl.find_nearest_neighbors(np.asarray(pos))
        #print('CHECK: Nearest from clousmetrics: ',gmean(nearest))
        
        
        # Figure
        plot_figure(tmp,xy,dataobj=obj)
    
    
 


