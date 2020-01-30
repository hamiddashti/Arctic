# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 13:23:46 2020

@author: hamiddashti
"""

import glob
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fiona
import regionmask
from shapely.geometry import shape
from shapely.geometry import Polygon, MultiPolygon
import rioxarray



year1 = 2002
year2 = 2003
years = np.arange(year1,year2)

root_dir = '/data/ABOVE/MODIS/MCD15A3H/'
tif_dir = root_dir+'Tifs'

shpin = '/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/above_domain.shp'

tile = 'h11v02'


for year in years:
    textfile = root_dir+'filenames_'+str(year)+'.txt'
    f = open(textfile)
    content = f.readlines()
    f.close()
    indices = [i for i, s in enumerate(content) if tile in s]
    tile_content = [content[i] for i in indices]
    
    
    file_tmp = tile_content[0][32:-5]
    file_path =  root_dir+'Tifs/'+str(year)+'/'+'*'+file_tmp+'*.tif' 
    layernames = glob.glob(file_path)
    
    i_fpar = [i for i, s in enumerate(layernames) if 'Fpar_500m' in s]
    i_lai = [i for i, s in enumerate(layernames) if 'Lai_500m' in s]
    i_fpar_sdt = [i for i, s in enumerate(layernames) if 'FparStdDev_500m' in s]
    i_lai_std = [i for i, s in enumerate(layernames) if 'LaiStdDev_500m' in s]
    i_FparLai_QC = [i for i, s in enumerate(layernames) if 'FparLai_QC' in s]
    i_FparExtra_QC = [i for i, s in enumerate(layernames) if 'FparExtra_QC' in s]
    
    tmp_word = 'A'+str(year)
    doy = (int(layernames[0].split(tmp_word,1)[1][:3]))
    df = pd.DataFrame({"year":[year],"doy":[doy]})
    time_index = pd.to_datetime(df['year'] * 1000 + df['doy'], format='%Y%j')
    time = xr.Variable('time',time_index)
    
    ds_size = xr.open_rasterio(layernames[0])
    x_size=ds_size.sizes['x']
    y_size=ds_size.sizes['y']
    chunks = {'x': x_size, 'y': y_size, 'band': 1}
    
      
    
    Fpar_init =  xr.open_rasterio(layernames[i_fpar[0]],chunks=chunks)
    Fpar_init = Fpar_init.to_dataset(name='Fpar_500m')
        
    Lai_init =  xr.open_rasterio(layernames[i_lai[0]],chunks=chunks)
    Lai_init = Lai_init.to_dataset(name='Lai_500m')
        
    FparStd_init =  xr.open_rasterio(layernames[i_fpar_sdt[0]],chunks=chunks)
    FparStd_init = FparStd_init.to_dataset(name='FparStdDev_500m')
        
    LaiStd_init =  xr.open_rasterio(layernames[i_lai_std[0]],chunks=chunks)
    LaiStd_init = LaiStd_init.to_dataset(name='LaiStdDev_500m')
        
    FparLai_QC_init =  xr.open_rasterio(layernames[i_FparLai_QC[0]],chunks=chunks)
    FparLai_QC_init = FparLai_QC_init.to_dataset(name='FparLai_QC')
        
    FparExtra_QC_init =  xr.open_rasterio(layernames[i_FparExtra_QC[0]],chunks=chunks)
    FparExtra_QC_init = FparExtra_QC_init.to_dataset(name='FparExtra_QC')
        
    ds_init = xr.merge([Fpar_init,Lai_init,FparStd_init,LaiStd_init,FparLai_QC_init,FparExtra_QC_init])
    ds_init = ds_init.assign_coords({'time':time})
    
    
    for i in np.arange(1,len(tile_content)):
        print(i)
        file_tmp = tile_content[i][32:-5]
        file_path =  root_dir+'Tifs/'+str(year)+'/'+'*'+file_tmp+'*.tif' 
        layernames = glob.glob(file_path)
        
        i_fpar = [i for i, s in enumerate(layernames) if 'Fpar_500m' in s]
        i_lai = [i for i, s in enumerate(layernames) if 'Lai_500m' in s]
        i_fpar_sdt = [i for i, s in enumerate(layernames) if 'FparStdDev_500m' in s]
        i_lai_std = [i for i, s in enumerate(layernames) if 'LaiStdDev_500m' in s]
        i_FparLai_QC = [i for i, s in enumerate(layernames) if 'FparLai_QC' in s]
        i_FparExtra_QC = [i for i, s in enumerate(layernames) if 'FparExtra_QC' in s]
        
        tmp_word = 'A'+str(year)
        doy = (int(layernames[0].split(tmp_word,1)[1][:3]))
        df = pd.DataFrame({"year":[year],"doy":[doy]})
        time_index = pd.to_datetime(df['year'] * 1000 + df['doy'], format='%Y%j')
        time = xr.Variable('time',time_index)
        
        Fpar_tmp =  xr.open_rasterio(layernames[i_fpar[0]],chunks=chunks)
        Fpar_tmp = Fpar_tmp.to_dataset(name='Fpar_500m')
            
        Lai_tmp =  xr.open_rasterio(layernames[i_lai[0]],chunks=chunks)
        Lai_tmp = Lai_tmp.to_dataset(name='Lai_500m')
        
        FparStd_tmp =  xr.open_rasterio(layernames[i_fpar_sdt[0]],chunks=chunks)
        FparStd_tmp = FparStd_tmp.to_dataset(name='FparStdDev_500m')
            
        LaiStd_tmp =  xr.open_rasterio(layernames[i_lai_std[0]],chunks=chunks)
        LaiStd_tmp = LaiStd_tmp.to_dataset(name='LaiStdDev_500m')
            
        FparLai_QC_tmp =  xr.open_rasterio(layernames[i_FparLai_QC[0]],chunks=chunks)
        FparLai_QC_tmp = FparLai_QC_tmp.to_dataset(name='FparLai_QC')
            
        FparExtra_QC_tmp =  xr.open_rasterio(layernames[i_FparExtra_QC[0]],chunks=chunks)
        FparExtra_QC_tmp = FparExtra_QC_tmp.to_dataset(name='FparExtra_QC')
            
        ds_tmp = xr.merge([Fpar_init,Lai_init,FparStd_init,LaiStd_init,FparLai_QC_init,FparExtra_QC_init])
        ds_tmp = ds_tmp.assign_coords({'time':time})
        
        if (i==1):
            ds_final=xr.concat([ds_init,ds_tmp],dim='time')
        else:
            ds_final = xr.concat([ds_final,ds_tmp],dim='time')
            
            
    ds_final = ds_final.rename({'x': 'lon','y': 'lat'})
    ds_final=ds_final.squeeze(drop=True)
    
    lat=ds_final.lat
    lon=ds_final.lon
    
    shp_domain = fiona.open(shpin)
    first = shp_domain.next()
    shp_domain_geom = shape(first['geometry'])      # this is shapely 
    my_shp = MultiPolygon(shp_domain_geom)
    shp_poly = regionmask.Regions([my_shp])
    mask = shp_poly.mask(lon, lat)
    masked_ds = ds_final.where(mask == 0)
        
        
    