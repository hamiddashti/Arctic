import xarray as xr
import rioxarray
import geopandas
import numpy as np
from shapely.geometry import box, mapping
import matplotlib.pyplot as plt
import time 

t1=time.time
in_dir = '/data/ABOVE/MODIS/MYD21A2/'
out_dir = '/data/ABOVE/MODIS/MYD21A2/Clip/' 
geodf = geopandas.read_file("/data/home/hamiddashti/mnt/nasa_above/working/\
modis_analyses/Above_180km_Clip_Geographic.shp")

ds = xr.open_rasterio(in_dir+'LST_Day_1KM.nc')  
ds = ds.rio.set_crs(geodf.crs)
for i in np.arange(1,2):
    print(f'Cliping LST Day, Object id: {i}') 
    tmp = geodf[geodf['OBJECTID']==i]
    file_clipped = ds.rio.clip(tmp.geometry.apply(mapping), tmp.crs)
    outname = 'lst_day_tileID'+str(i)+'.nc'
    file_clipped.to_netcdf(out_dir+outname)

'''
ds = xr.open_rasterio(in_dir+'LST_Night_1KM.nc')  
ds = ds.rio.set_crs(geodf.crs)
for i in np.arange(1,2):
    print(f'Cliping LST Night, Object id: {i}') 
    tmp = geodf[geodf['OBJECTID']==i]
    file_clipped = ds.rio.clip(tmp.geometry.apply(mapping), tmp.crs)
    outname = 'LST_Night_'+str(i)
    file_clipped.rio.to_raster(out_dir+outname)

ds = xr.open_rasterio(in_dir+'QC_Day.nc')  
ds = ds.rio.set_crs(geodf.crs)
for i in np.arange(1,2):
    print(f'Cliping QC day , Object id: {i}') 
    tmp = geodf[geodf['OBJECTID']==i]
    file_clipped = ds.rio.clip(tmp.geometry.apply(mapping), tmp.crs)
    outname = 'QC_Day_'+str(i)
    file_clipped.rio.to_raster(out_dir+outname)

ds = xr.open_rasterio(in_dir+'QC_Night.nc')  
ds = ds.rio.set_crs(geodf.crs)
for i in np.arange(1,2):
    print(f'Cliping QC night , Object id: {i}') 
    tmp = geodf[geodf['OBJECTID']==i]
    file_clipped = ds.rio.clip(tmp.geometry.apply(mapping), tmp.crs)
    outname = 'QC_night_'+str(i)
    file_clipped.rio.to_raster(out_dir+outname)
'''
t2=time.time   

print(f'########## All done in:{(t2-t1)/60}') 

