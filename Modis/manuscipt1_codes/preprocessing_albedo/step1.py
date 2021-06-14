"""
Step 1 of preprocessing albedo data. This step has been done on UA hpc 
due to the heavy processing. 
"""
import geopandas as gpd
import xarray as xr
import numpy as np
import rasterio
import pandas as pd
import os
from rasterio.mask import mask
import pycrs
import dask
import modis_functions
from pathlib import Path
import glob
from rasterio.enums import Resampling
import rioxarray

tif_dir = (
    "/xdisk/davidjpmoore/hamiddashti/data/ALBEDO2/orders/"
    "8d2930e665cfdf9b1f358ba0fc39f38d/Albedo_Boreal_North_America/data/")
out_dir = (
    "/xdisk/davidjpmoore/hamiddashti/nasa_above_outputs/albedo_processed/")

shp_dir = ("/xdisk/davidjpmoore/hamiddashti/data/study_area/")
# ----------------------------------------------------------------------
# Step 1: filter original albedo data based on quality flags and clip
# to the extent of above bigger domain.
# ----------------------------------------------------------------------
fnames_albedo = glob.glob(tif_dir + "*_albedo.tif")
fnames_quality = []
for f in fnames_albedo:
    b = f.rsplit("albedo", 1)
    fnames_quality.append("quality".join(b))

geodf = gpd.read_file(shp_dir + "Above_180km_clip.shp")
summer_flag = [0, 1, 2, 4, 5, 6, 16, 17, 18, 20, 21, 22]
winter_flag = [
    0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35,
    36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55
]


def filtering_fun(i):
    """filtering original 500m albedo data 
    """
    basename = os.path.basename(fnames_albedo[i])
    print(basename, flush=True)
    albedo_tmp = xr.open_rasterio(fnames_albedo[i])
    quality_tmp = xr.open_rasterio(fnames_quality[i])
    tmp_date = pd.to_datetime(os.path.basename(fnames_albedo[i])[15:19] +
                              os.path.basename(fnames_albedo[i])[20:23],
                              format='%Y%j')
    if 5 <= tmp_date.month <= 9:
        # Summer months
        da_albedo_clean = albedo_tmp.where(quality_tmp.isin(summer_flag))
    else:
        # Winter months
        da_albedo_clean = albedo_tmp.where(quality_tmp.isin(winter_flag))
    # Clip to the extended core domain
    clipped = da_albedo_clean.rio.clip(geodf.geometry)
    clipped.rio.to_raster(out_dir + "step1_filtered_data/cliped_" + basename)


lazy_results = []
for i in range(len(fnames_albedo)):
    mycal = dask.delayed(filtering_fun)(i)
    lazy_results.append(mycal)
results = dask.compute(lazy_results)