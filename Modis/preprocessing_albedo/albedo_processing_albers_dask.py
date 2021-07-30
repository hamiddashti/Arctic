import xarray as xr
import matplotlib.pylab as plt
import numpy as np
import xesmf as xe
from dask.diagnostics import ProgressBar
import matplotlib.pylab as plt
import glob
from dask.distributed import Client, LocalCluster
import os
import rioxarray
import pandas as pd
import dask
import modis_functions
from rasterio.enums import Resampling
import geopandas as gpd

tif_dir = ("/data/ABOVE/MODIS/ALBEDO2/orders/8d2930e665cfdf9b1f358ba0fc39f38d/"
           "Albedo_Boreal_North_America/data/")
out_dir = (
    "/data/home/hamiddashti/nasa_above/outputs/albedo_processed/new_filter/")

# out_dir = (
#     "/data/home/hamiddashti/nasa_above/outputs/albedo_processed/")

shp_dir = ("/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/"
           "Study_area/")

fnames_albedo_all = glob.glob(tif_dir + "*_albedo.tif")
fnames_quality_all = []
for f in fnames_albedo_all:
    b = f.rsplit("albedo", 1)
    fnames_quality_all.append("quality".join(b))

summer_flag = [0, 1, 2, 4, 5, 6, 16, 17, 18, 20, 21, 22]
# winter_flag = [
# In case wanted to include more pixels
#     0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22, 23,
#     24, 25, 26, 27, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 48, 49, 50,
#     51, 52, 53, 54, 55, 56, 57, 58, 59
# ]
winter_flag = [
    0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35,
    36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55
]


def filtering_fun(i):
    # for i in range(len(fname_albedo_2003)):
    basename = os.path.basename(fnames_albedo_all[i])
    print(basename)
    albedo_tmp = xr.open_rasterio(fnames_albedo_all[i])
    quality_tmp = xr.open_rasterio(fnames_quality_all[i])
    tmp_date = pd.to_datetime(os.path.basename(fnames_albedo_all[i])[15:19] +
                              os.path.basename(fnames_albedo_all[i])[20:23],
                              format='%Y%j')
    if 5 <= tmp_date.month <= 9:
        # Summer months
        da_albedo_clean = albedo_tmp.where(quality_tmp.isin(summer_flag))
    else:
        # Winter months
        da_albedo_clean = albedo_tmp.where(quality_tmp.isin(winter_flag))

    da_albedo_clean.rio.to_raster(out_dir + "filtered_data/dask/filtered_" +
                                  basename)


lazy_results = []
for i in range(len(fnames_albedo_all)):
    mycal = dask.delayed(filtering_fun)(i)
    lazy_results.append(mycal)
results = dask.compute(lazy_results)
# -----------------------------------------------------------
from dask.distributed import Client, LocalCluster

client = Client(processes=False)

filtered_dir = (
    "/data/home/hamiddashti/nasa_above/outputs/albedo_processed/filtered_data/dask/"
)
fnames_albedo_all = glob.glob(filtered_dir + "*_albedo.tif")
mydate = []
for f in fnames_albedo_all:
    mydate.append(
        pd.to_datetime(os.path.basename(f)[24:28] + os.path.basename(f)[29:32],
                       format='%Y%j'))
date_xr = xr.Variable("time", mydate)

chunks = {'x': 15355, 'y': 9352, 'band': 1}
da_albedo = xr.concat(
    [xr.open_rasterio(f, chunks=chunks) for f in fnames_albedo_all],
    dim=date_xr)
da_albedo = da_albedo.squeeze()
da_albedo = da_albedo.sortby("time")
chunks = {'x': -1, 'y': -1, 'time': "auto"}
da_albedo_rechunk = da_albedo.chunk(chunks=chunks)
# da_albedo_rechunk = da_albedo_rechunk.squeeze()
# da_albedo_rechunk_sorted = da_albedo_rechunk.sortby("time")
# da_albedo_seasonal = modis_functions.weighted_season_resmaple(
#     da_albedo_rechunk)
da_albedo_growing_season = modis_functions.growing_season(
    da_albedo_rechunk)
da_albedo_growing_season.to_netcdf(out_dir +
                             "growing_season_dask/albedo_growing_season.nc")
# --------------------------------------------------------------
da = xr.open_dataarray(out_dir + "seasonal_dask/albedo_season_resample.nc",
                       decode_coords="all")
domain = gpd.read_file(shp_dir + "CoreDomain.shp")
da_domain = da.rio.clip(domain.geometry)
da_domain.to_netcdf(out_dir + "seasonal_dask/albedo_season_resample_cliped.nc")
# ------------------------------------------------------
lst_ref = xr.open_dataset(
    "/data/home/hamiddashti/nasa_above/outputs/"
    "lst_processed/albers_proj_lst_mean_Annual.nc",
    decode_coords="all")
lst_ref = lst_ref["lst_mean_Annual"].isel(year=0)
lst_ref = lst_ref.rename({"x": "lon", "y": "lat"})
# crs = xr.open_rasterio(fnames_albedo_all[0]).rio.crs

da = xr.open_dataarray(out_dir +
                       "seasonal_dask/albedo_season_resample_cliped.nc",
                       decode_coords="all")
da = da.rename({"x": "lon", "y": "lat"})

# Remove "spatial_ref" coordinate, if not it messes up with reprojecting
da = da.reset_coords("spatial_ref", drop=True)
da = da.rio.write_crs(lst_ref.rio.crs)

da_reproj = da.rio.reproject_match(lst_ref,
                                   resampling=Resampling.bilinear) * 0.001
da_reproj.to_netcdf(out_dir +
                    "final_albedo/seasonal/albers/final_seasonal_albedo.nc")

# ds_out = xr.Dataset({
#     'lat': (['lat'], lst_ref['lat'].values),
#     'lon': (['lon'], lst_ref['lon'].values)
# })
# ds_in = xr.Dataset({
#     'lat': (['lat'], da['lat'].values),
#     'lon': (['lon'], da['lon'].values)
# })
# # Create the regridder
# regridder = xe.Regridder(ds_in, ds_out, 'bilinear')
# regridder.to_netcdf(out_dir + "seasonal_dask/regridder.nc")
