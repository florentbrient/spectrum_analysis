#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 15:50:49 2024

@author: fbrient
"""
from pyhdf.SD import SD,SDC 
import matplotlib.pyplot as plt
import numpy as np
import cloudmetrics



path0 = "/home/fbrient/Dropbox/GitHub/cloudmetrics/Data/Download/DataTerra/"
file0 = "MOD06_L2.A2022182.1530.061.2022183021354.hdf"

file = path0+file0
print(file)

f = SD(file,SDC.READ)

# List of variables
variables = list(f.datasets())
print(variables)

varname = 'Cloud_Water_Path' 
#varname = 'Cloud_Fraction'
sds_obj = f.select(varname)
print(sds_obj)

lat = f.select('Latitude').get()
lon = f.select('Longitude').get()


# Dim CWP
#Cell_Along_Swath_1km:mod06', 'Cell_Across_Swath_1km:mod06

offs = 0
sc = 1
fv = 0
for key, value in sds_obj.attributes().items():
    if key == "add_offset":
        offs = value
    elif key == "scale_factor":
        sc = value
    elif key == "_FillValue":
        fv = value
fie = sds_obj.get()
fie[fie == fv] = 0
fie = (fie - offs) * sc


# Plot figure
fig = plt.figure()
ax = fig.add_subplot()
#ax.imshow(lat,lon,fie)
pos = ax.imshow(fie, extent=[np.min(lon), np.max(lon), np.min(lat), np.max(lat)],vmax=250)
cbar = fig.colorbar(pos,ax=ax, extend='both')
cbar.minorticks_on()
plt.show()

#df.loc[0, varname] = fie


#iorg = cloudmetrics.mask.iorg_objects(mask=fie, periodic_domain=False)

#Spectra
square_fie = fie[0:min(fie.shape),0:min(fie.shape)]
wavenumbers, psd_1d_radial, psd_1d_azimuthal = cloudmetrics.scalar.compute_spectra(square_fie)
spectral_length_moment = cloudmetrics.scalar.spectral_length_moment(wavenumbers, psd_1d_radial)

