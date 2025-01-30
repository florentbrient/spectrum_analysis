# -*- coding: utf-8 -*-
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import math
from math import radians, sin, cos, acos
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.ticker import *
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


# Fonction qui permet de tracer un rectangle sur une carte
# Utile pour tracer la zone de convection WMDW
def plot_rectangle(m, lonmin,lonmax,latmin,latmax,color,linewidth,label):
    xs = [lonmin,lonmax,lonmax,lonmin,lonmin]
    ys = [latmin,latmin,latmax,latmax,latmin]
    m.plot(xs, ys,latlon = True,linewidth=linewidth,color=color,label=label)  
def plot_rectangle2(m, lonmin,lonmax,latmin,latmax,color,linewidth):
    xs = [lonmin,lonmax,lonmax,lonmin,lonmin]
    ys = [latmin,latmin,latmax,latmax,latmin]
    m.plot(xs, ys,latlon = True,linewidth=linewidth,color=color,alpha=1)
#-----------------------------------------------------------------------------------------
def plot_rectangle3(m, lonmin,lonmax,latmin,latmax,color,linewidth):
    xs = [lonmin,lonmax,lonmax,lonmin,lonmin]
    ys = [latmin,latmin,latmax,latmax,latmin]
    m.plot(xs, ys,latlon = True,linewidth=linewidth,color=color,alpha=1)
#-----------------------------------------------------------------------------------------
def plot_rectangle4(lonmin,lonmax,latmin,latmax,color,linewidth):
    xs = [lonmin,lonmax,lonmax,lonmin,lonmin]
    ys = [latmin,latmin,latmax,latmax,latmin]
    plt.plot(xs, ys,linewidth=linewidth,color=color,alpha=1,zorder=10)
#-----------------------------------------------------------------------------------------
def plot_contour_from_lonlat(lon,lat,color,linewidth,alpha):
    ind_min_lat = np.unravel_index(np.argmin(lat, axis=None), lat.shape)
    ind_min_lon = np.unravel_index(np.argmin(lon, axis=None), lon.shape)
    ind_max_lat = np.unravel_index(np.argmax(lat, axis=None), lat.shape)
    ind_max_lon = np.unravel_index(np.argmax(lon, axis=None), lon.shape)
    xs = [lon[ind_min_lat],lon[ind_min_lon],lon[ind_max_lat],lon[ind_max_lon],lon[ind_min_lat]]
    ys = [lat[ind_min_lat],lat[ind_min_lon],lat[ind_max_lat],lat[ind_max_lon],lat[ind_min_lat]]
    plt.plot(xs, ys,linewidth=linewidth,color=color,alpha=alpha,zorder=5)       
#-----------------------------------------------------------------------------------------
def distance_latlon(slat,elat,slon,elon):
    slat = radians(float(slat))
    slon = radians(float(slon))
    elat = radians(float(elat))
    elon = radians(float(elon))
    dist = 6371.01 * acos(sin(slat)*sin(elat) + cos(slat)*cos(elat)*cos(slon - elon))
    return dist
#-----------------------------------------------------------------------------------------
def square_from_center(lon,lat,side_km):
    dist_one_lon = distance_latlon(lat,lat,lon,lon+1)
    dist_one_lat = distance_latlon(lat,lat+1,lon,lon)
    lonmin = lon - 0.5*side_km/dist_one_lon
    lonmax = lon + 0.5*side_km/dist_one_lon
    latmin = lat - 0.5*side_km/dist_one_lat
    latmax = lat + 0.5*side_km/dist_one_lat
    check_dist = distance_latlon(latmin,latmax,lon,lon)
    return lonmin,lonmax,latmin,latmax    
#-----------------------------------------------------------------------------------------
def rectangle_from_center(lon,lat,largeur_km,longueur_km):
    dist_one_lon = distance_latlon(lat,lat,lon,lon+1)
    dist_one_lat = distance_latlon(lat,lat+1,lon,lon)
    lonmin = lon - 0.5*longueur_km/dist_one_lon
    lonmax = lon + 0.5*longueur_km/dist_one_lon
    latmin = lat - 0.5*largeur_km/dist_one_lat
    latmax = lat + 0.5*largeur_km/dist_one_lat
    return lonmin,lonmax,latmin,latmax    
#-----------------------------------------------------------------------------------------
def latlon_from_dist_and_cap(lat_i,lon_i,d, cap):
    #lat_i, lon_i : initial point (in degree)
    # d : distance in km
    # cap : cap in degree
    R = 6378.1 #Radius of the Earth
    brng = np.pi*cap/180. 
    #
    lat1 = math.radians(lat_i) #Current lat point converted to radians
    lon1 = math.radians(lon_i) #Current long point converted to radians
    #
    lat2 = math.asin( math.sin(lat1)*math.cos(d/R) +
     math.cos(lat1)*math.sin(d/R)*math.cos(brng))
    lon2 = lon1 + math.atan2(math.sin(brng)*math.sin(d/R)*math.cos(lat1),
             math.cos(d/R)-math.sin(lat1)*math.sin(lat2))
    lat_f = math.degrees(lat2)
    lon_f = math.degrees(lon2)
    return lon_f,lat_f
#-----------------------------------------------------------------------------------------







def reverse_colourmap(cmap, name = 'my_cmap_r'):     
    reverse = []
    k = []   

    for key in cmap._segmentdata:   
        k.append(key)
        channel = cmap._segmentdata[key]
        data = []
        for t in channel:                    
            data.append((1-t[0],t[2],t[1]))            
        reverse.append(sorted(data))    
    LinearL = dict(zip(k,reverse))
    my_cmap_r = mpl.colors.LinearSegmentedColormap(name, LinearL) 
    return my_cmap_r   
cm1 = cmx.get_cmap('seismic')
cmap_r = reverse_colourmap(cm1)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def inv_cmap(data,cmap0):
    idx=np.sign(np.nanmin(data))-np.sign(np.nanmax(data))
    if idx != 0:
        cmap0='RdBu'
    return cmap0


# Suppression des axes a droite et en haut
# Adaptation des scripts de Florent Brient
# ax = fig.add_subplot(111) par exemple
# spines = 'left', 'bottom', etc
def adjust_spines(ax, spines,width_spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 0))  # outward by 10 points
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
    elif 'top' in spines:
        ax.xaxis.set_ticks_position('top')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])
    for axis in spines:
        ax.spines[axis].set_linewidth(width_spines)

def legend_fig(xstring,ystring,size_leg,size_ticks):
    ax = plt.gca()
    label = ax.set_xlabel(xstring,fontsize = size_leg,fontweight='bold')
    label = ax.set_ylabel(ystring,fontsize = size_leg,fontweight='bold')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(size_ticks)
        label.set_fontweight('bold')
    ax.tick_params('both', length=10, width=2, which='major', direction='out')
    ax.tick_params('both', length=5, width=1, which='minor', direction='out')
 
     
def list_of_hex_colours(N, base_cmap): # ! A deplacer dans plot_tools !
    """
    Return a list of colors from a colourmap as hex codes
        Arguments:
            cmap: colormap instance, eg. cm.jet.
            N: number of colors.
        Author: FJC
    """
    cmap = cmx.get_cmap(base_cmap, N)

    hex_codes = []
    for i in range(cmap.N):
        rgb = cmap(i)[:3] # will return rgba, we take only first 3 so we get rgb
        hex_codes.append(colors.rgb2hex(rgb))
    return hex_codes



#------------------------------------------------------------------------------------       
# parametre de figure avec plotly
#------------------------------------------------------------------------------------    

def figure_layout(fig,fig_width,fig_height):
    fig.update_layout(autosize=False,
                      width=fig_width,
                      height=fig_height,
                      margin=dict(autoexpand=True,l=20,r=20,t=20,b=20),
                      showlegend=False,
                      plot_bgcolor='white',
                     )
    fig.update_yaxes(dict(showgrid=False,zeroline=False,showline=True,showticklabels=True,linecolor='black',linewidth=2,
                     ticks='outside',tickfont=dict(size=15),tickwidth=2,ticklen=10)
                    )
    fig.update_xaxes(dict(showline=True,showgrid=False,showticklabels=True,tickformat="%H:%M:%S",linecolor='black',linewidth=2,
                     ticks='outside',tickfont=dict(size=15))
                    )
#------------------------------------------------------------------------------------    
# TAILLE POLICE AXES ET LEGENDES PLUS GRANDE POUR ARTICLE
def figure_layout_V2(fig,fig_width,fig_height):
    fig.update_layout(autosize=False,
                      width=fig_width,
                      height=fig_height,
                      margin=dict(autoexpand=True,l=20,r=20,t=20,b=20),
                      showlegend=False,
                      plot_bgcolor='white',
                     )
    fig.update_yaxes(dict(showgrid=False,zeroline=False,showline=True,showticklabels=True,linecolor='black',linewidth=2,
                     ticks='outside',tickfont=dict(family="Computer Modern Typewriter",size=20),tickwidth=2,ticklen=10)
                    )
    fig.update_xaxes(dict(showline=True,showgrid=False,showticklabels=True,tickformat="%H:%M:%S",linecolor='black',linewidth=2,
                     ticks='outside',tickfont=dict(family="Computer Modern Typewriter",size=20),tickwidth=2,ticklen=10)
                    )
#------------------------------------------------------------------------------------    

    
def figure_legend(fig):
    fig.update_layout(showlegend=True)
    
def figure_layout_spectrum(fig,fig_width,fig_height):
    fig.update_layout(autosize=False,
                      width=fig_width,
                      height=fig_height,
                      margin=dict(autoexpand=True,l=20,r=20,t=20,b=20),
                      showlegend=False,
                      plot_bgcolor='white',
                     )
    fig.update_yaxes(dict(showgrid=True,zeroline=False,showline=True,showticklabels=True,linecolor='black',linewidth=2,
                     ticks='outside',tickfont=dict(size=15),tickwidth=2,ticklen=10,type="log",gridwidth=1, gridcolor='Lightgray')
                    )
    fig.update_xaxes(dict(showline=True,showgrid=True,showticklabels=True,linecolor='black',linewidth=2,
                     ticks='outside',tickfont=dict(size=15),tickwidth=2,ticklen=10,type="log",gridwidth=1, gridcolor='Lightgray')
                    )

def figure_layout_histogramm(fig,fig_width,fig_height):
    fig.update_layout(autosize=False,
                      width=fig_width,
                      height=fig_height,
                      margin=dict(autoexpand=True,l=20,r=20,t=20,b=20),
                      showlegend=False,
                      plot_bgcolor='white',
                     )
    fig.update_yaxes(dict(showgrid=False,zeroline=False,showline=True,showticklabels=True,linecolor='black',linewidth=2,
                     ticks='outside',tickfont=dict(size=15),tickwidth=2,ticklen=10)
                    )
    fig.update_xaxes(dict(showline=True,showgrid=False,showticklabels=True,linecolor='black',linewidth=2,
                     ticks='outside',tickfont=dict(size=15))
                    )