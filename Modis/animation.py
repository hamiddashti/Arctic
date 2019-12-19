# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 08:48:27 2019

@author: hamiddashti
"""

import os
import xarray as xr
import pandas as pd
from matplotlib import pyplot as plt
import geopandas as gpd
import imageio

import salem

os.chdir('P:\\nasa_above\\working\\modis_analyses')
    # Read the nc file as xarray DataSet
ds = xr.open_dataset('GIMMS3g_LAI_Bimonthly_2000_2010_60min.nc')

 
dates=pd.date_range(start='1/1/2000', end='12/31/2010',periods=264)
ds['time']=dates
ds=ds*0.1
ds=ds.sel(lat=slice(75,45),lon=slice(-162,-51))


ds_ymax = ds.resample(time='AS').max()

ds_mean = ds_ymax.mean(dim='time')
ds_std = ds_ymax.std(dim='time')

test=(ds_ymax-ds_mean)/ds_std

test = test.rename({'LAI':'z_score'})

year = test.time.dt.year
year = year.values


n=test.time.size
for i in range(n):
    im = test.z_score[i,:,:].plot(cmap ='PiYG' ,figsize=(20,10),add_colorbar=False)
    cb = plt.colorbar(im, orientation="vertical", pad=0.05)
    cb.set_label(label='Z Score', size='large', weight='bold')
    cb.ax.tick_params(labelsize='large')
    plt.rc('xtick',labelsize=18)
    plt.rc('ytick',labelsize=18)
    plt.title('Year: '+str(year[i]),fontsize=18)
    plt.xlabel('Longitude [degree_east]', fontsize=18)
    plt.ylabel('Latitude [degree_north]', fontsize=16)
    #plt.show()
    plt.savefig(f"animation_lai2/LAI_frame_{i}.png")
    plt.close()
 
    

images = []
for i in range(n):
    images.append(imageio.imread('animation_lai2/LAI_frame_'+str(i)+'.png'))


imageio.mimsave('animation_lai2/LAI_animation.gif', images,duration=4)













































