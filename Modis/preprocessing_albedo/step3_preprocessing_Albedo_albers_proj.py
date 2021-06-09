
#----------------------------------------------------------------------
#		Step2: take the growing season, monthly, and annual means
#----------------------------------------------------------------------

import xarray as xr
import rioxarray
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import modis_functions
import glob
import os

in_dir = ("/data/home/hamiddashti/nasa_above/outputs/albedo_processed/"
          "filtered_reproject/")
out_dir = ("/data/home/hamiddashti/nasa_above/outputs/albedo_processed/"
           "final_resampled/")

filenames_albedo = glob.glob(in_dir + "*_albedo.nc")
date = []
for f in filenames_albedo:
    date.append(
        pd.to_datetime(os.path.basename(f)[30:34] + os.path.basename(f)[35:38],
                       format='%Y%j'))
time = pd.DatetimeIndex(date)
time = xr.Variable("time", pd.DatetimeIndex(date))
chunks = {'x': 4172, 'y': 4343}

print("concatenating")
da_albedo = xr.concat(
    [xr.open_dataarray(f, chunks=chunks) for f in filenames_albedo], dim=time)

# da_albedo = da_albedo.chunk({'x': 4172, 'y': 6536, 'time': 1})
# da_albedo = da_albedo.sortby("time").squeeze()

print(" Group by year\n")
albedo_annual = da_albedo.groupby('time.year').mean(dim='time')
albedo_annual = albedo_annual.chunk({'x': 1000, 'y': 1000, 'year': 1})
albedo_annual.to_netcdf(out_dir + "albedo_annual.nc")

print(
    '---------- Growing season (April-October)------------------------------\n'
)
albedo_growing = modis_functions.growing_season(da_albedo)
albedo_growing.to_netcdf(out_dir + "albedo_growing.nc")

print(
    '---------- Monthly mean ------------------------------------------------\n'
)
albedo_monthly = da_albedo.resample(time="1MS").mean()
albedo_monthly.to_netcdf(out_dir + "albedo_monthly.nc")

print(
    '------------ Seasonal mean --------------------------------------------------\n'
)
albedo_seasonal = modis_functions.weighted_season_resmaple(da_albedo)
albedo_seasonal.to_netcdf(out_dir + "albedo_seasonal")
print(f"All Done!\n")
