from os.path import basename
from dask import base
from numpy import datetime_as_string
import xarray as xr
import pandas as pd
import os
import glob
import rioxarray
from rasterio.enums import Resampling
import dask
import matplotlib.pylab as plt

SCALING_FACTOR = 0.001
summer_flag = [0, 1, 2, 4, 5, 6, 16, 17, 18, 20, 21, 22]
winter_flag = [
    0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35,
    36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55
]


def myfun(i):
    print(time[i])
    da_albedo = xr.open_rasterio(filenames_albedo[i]).squeeze()
    da_quality = xr.open_rasterio(filenames_quality[i]).squeeze()
    if 5 <= time[i].month <= 9:
        da_tmp = da_albedo.where(da_quality.isin(summer_flag))
    else:
        da_tmp = da_albedo.where(da_quality.isin(winter_flag))
    da_reproject = da_tmp.rio.reproject_match(lst_ref,
                                              resampling=Resampling.bilinear)
    da_reproject = da_reproject.where(da_reproject != -9999)
    da_reproject = da_reproject * SCALING_FACTOR
    basename = os.path.splitext(os.path.basename(filenames_albedo[i]))[0]
    da_reproject.rio.to_raster(out_dir + "reproject_" + basename + ".tif")


# To convert it to LST resolution
lst_ref = xr.open_dataset(
    "/data/home/hamiddashti/nasa_above/outputs/"
    "lst_processed/albers_proj_lst_mean_Annual.nc",
    decode_coords="all")
lst_ref = lst_ref["lst_mean_Annual"].isel(year=0)

in_dir = ("/data/ABOVE/MODIS/ALBEDO2/orders/8d2930e665cfdf9b1f358ba0fc39f38d/"
          "Albedo_Boreal_North_America/data/")
out_dir = ("/data/home/hamiddashti/nasa_above/outputs/albedo_processed/"
           "filtered_reproject/tmp/")

filenames_albedo = glob.glob(in_dir + "*_albedo.tif")
filenames_quality = []
for f in filenames_albedo:
    b = f.rsplit("albedo", 1)
    filenames_quality.append("quality".join(b))

tileID = 75






# filenames_quality = glob.glob(in_dir + "*_quality.tif")
date = []
for f in filenames_albedo:
    date.append(
        pd.to_datetime(os.path.basename(f)[20:24] + os.path.basename(f)[25:28],
                       format='%Y%j'))
time = pd.DatetimeIndex(date)

delayed_results = []
for i in range(len(filenames_albedo)):
    # for i in range(1):
    mycals = dask.delayed(myfun)(i)
    delayed_results.append(mycals)
results = dask.compute(*delayed_results)
"""


# ------------------ Xarray way donw there------------------------
time = xr.Variable("time", pd.DatetimeIndex(date))
chunks = {'x': 6536, 'y': 6473, 'band': 1}
da_albedo = xr.concat(
    [xr.open_rasterio(f, chunks=chunks) for f in filenames_albedo], dim=time)
da_albedo = da_albedo.sortby("time").squeeze()
da_quality = xr.concat(
    [xr.open_rasterio(f, chunks=chunks) for f in filenames_quality], dim=time)
da_quality = da_quality.sortby("time").squeeze()

da_albedo.to_netcdf(files_dir + "stacked_clip_albedo.nc")
da_quality.to_netcdf(files_dir + "stacked_clip_quality.nc")
"""