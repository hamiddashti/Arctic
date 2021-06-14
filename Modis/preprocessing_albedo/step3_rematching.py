# The main goal of this script is to rematch the albedo product (500m) to Modis LST (1000)
# The rematch function provided by rioxarray takes care of this process.

import xarray as xr
import rioxarray
import pandas as pd

fname = 'Albedo_annual.tif'
out_dir = (
    "/data/ABOVE/MODIS/blue_sky_albedo/orders/df99d6e7e26fb666143dbf2de2de4707/"
    "Albedo_Boreal_North_America/Albedo_processed/mosaic/matched/")
ds = xr.open_rasterio(fname)
lst = xr.open_rasterio('Test_lst.tif')
lst = lst.rio.set_crs(4326)
ds = ds.rio.reproject_match(lst)
date = pd.date_range('2002', '2015', freq='AS')
ds.reset_index(['band'], drop=True)
ds = ds.rename({'band': 'year'})
ds = ds.assign_coords({'year': date.year})
ds.rio.to_raster(out_dir + 'Albedo_annual.nc')

fname = 'Albedo_growing.tif'
out_dir = '/data/ABOVE/MODIS/blue_sky_albedo/orders/df99d6e7e26fb666143dbf2de2de4707/Albedo_Boreal_North_America/Albedo_processed/mosaic/matched/'
ds = xr.open_rasterio(fname)
lst = xr.open_rasterio('Test_lst.tif')
lst = lst.rio.set_crs(4326)
ds = ds.rio.reproject_match(lst)
date = pd.date_range('2002', '2015', freq='AS')
ds.reset_index(['band'], drop=True)
ds = ds.rename({'band': 'year'})
ds = ds.assign_coords({'year': date.year})
ds.rio.to_raster(out_dir + 'Albedo_growing.nc')

fname = 'Albedo_season_group.tif'
out_dir = '/data/ABOVE/MODIS/blue_sky_albedo/orders/df99d6e7e26fb666143dbf2de2de4707/Albedo_Boreal_North_America/Albedo_processed/mosaic/matched/'
ds = xr.open_rasterio(fname)
lst = xr.open_rasterio('Test_lst.tif')
lst = lst.rio.set_crs(4326)
ds = ds.rio.reproject_match(lst)
ds.reset_index(['band'], drop=True)
ds = ds.rename({'band': 'season'})
ds.rio.to_raster(out_dir + 'Albedo_season_group.nc')

fname = 'Albedo_month_group.tif'
out_dir = '/data/ABOVE/MODIS/blue_sky_albedo/orders/df99d6e7e26fb666143dbf2de2de4707/Albedo_Boreal_North_America/Albedo_processed/mosaic/matched/'
ds = xr.open_rasterio(fname)
lst = xr.open_rasterio('Test_lst.tif')
lst = lst.rio.set_crs(4326)
ds = ds.rio.reproject_match(lst)
ds.reset_index(['band'], drop=True)
ds = ds.rename({'band': 'month'})
ds.rio.to_raster(out_dir + 'Albedo_month_group.nc')

fname = 'Albedo_month_resample.tif'
out_dir = '/data/ABOVE/MODIS/blue_sky_albedo/orders/df99d6e7e26fb666143dbf2de2de4707/Albedo_Boreal_North_America/Albedo_processed/mosaic/matched/'
ds = xr.open_rasterio(fname)
lst = xr.open_rasterio('Test_lst.tif')
lst = lst.rio.set_crs(4326)
ds = ds.rio.reproject_match(lst)
dir = '/data/ABOVE/MODIS/blue_sky_albedo/orders/df99d6e7e26fb666143dbf2de2de4707/Albedo_Boreal_North_America/Albedo_processed/'
tmp = xr.open_dataarray(dir + 'Albedo_month_resample_75.nc')
time = tmp.time.values
ds.reset_index(['band'], drop=True)
ds = ds.rename({'band': 'time'})
ds = ds.assign_coords({'time': time})
ds.rio.to_raster(out_dir + 'Albedo_month_resample.nc')

fname = 'Albedo_season_resample.tif'
out_dir = '/data/ABOVE/MODIS/blue_sky_albedo/orders/df99d6e7e26fb666143dbf2de2de4707/Albedo_Boreal_North_America/Albedo_processed/mosaic/matched/'
ds = xr.open_rasterio(fname)
lst = xr.open_rasterio('Test_lst.tif')
lst = lst.rio.set_crs(4326)
ds = ds.rio.reproject_match(lst)
dir = '/data/ABOVE/MODIS/blue_sky_albedo/orders/df99d6e7e26fb666143dbf2de2de4707/Albedo_Boreal_North_America/Albedo_processed/'
tmp = xr.open_dataarray(dir + 'Albedo_season_resample_75.nc')
time = tmp.time.values
ds.reset_index(['band'], drop=True)
ds = ds.rename({'band': 'time'})
ds = ds.assign_coords({'time': time})
ds.rio.to_raster(out_dir + 'Albedo_season_resample.nc')
