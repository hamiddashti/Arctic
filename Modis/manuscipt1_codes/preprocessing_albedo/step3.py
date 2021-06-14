# Temporal resampling
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


# ---------- Step 3: resample the cliped file into annual ---------------------
def resampling_fun(tileID):
    # for tileID in np.arange(1, len(geodf) + 1):
    print("step3 ---> " + str(tileID))
    tmp_dir = out_dir + "step2_filtered_data_cliped" + "/tmp_" + str(tileID)
    fnames = glob.glob(tmp_dir + "/*.tif")
    mydate = []
    for f in fnames:
        mydate.append(
            pd.to_datetime(os.path.basename(f)[22:26] +
                           os.path.basename(f)[27:30],
                           format='%Y%j'))
    date_xr = xr.Variable("time", mydate)
    da = xr.concat([xr.open_rasterio(f) for f in fnames], dim=date_xr)
    da = da.sortby("time").squeeze()
    da.to_netcdf(out_dir + "step3_daily_tiles/da_concat_" + str(tileID) +
                 ".nc")


lazy_results = []
for tileID in np.arange(1, 176):
    mycal = dask.delayed(resampling_fun)(tileID)
    lazy_results.append(mycal)
results = dask.compute(lazy_results)
print("Step3_1 Done!")
