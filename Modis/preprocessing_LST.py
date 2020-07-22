import xarray as xr
import time
# import rioxarray
#import numpy as np
#import pandas as pd
# import matplotlib.pyplot as plt
import modis_functions

#init_time = time.time()
#from dask.distributed import Client
#client = Client(n_workers=2, threads_per_worker=2, memory_limit='1GB')
#client
t1 = time.time()
in_dir = '/xdisk/davidjpmoore/hamiddashti/data/LST/'
out_dir = '/xdisk/davidjpmoore/hamiddashti/data/LST/'

#print('--------- Filtering the LST data based on the quality flag) ----------\n')

chunks=({'time':10,'lat': 2692, 'lon': 8089})
"""
lst_day_ds = xr.open_dataset(in_dir+'MYD_LST_Day.nc',chunks=chunks)
lst_day = lst_day_ds['LST_Day_1KM']
lst_day_qc = lst_day_ds['QC_Day']
lst_night_ds = xr.open_dataset(in_dir+'MYD_LST_Night.nc',chunks=chunks)
lst_night = lst_night_ds['LST_Night_1KM']
lst_night_qc = lst_night_ds['QC_Night']

# These numbers comes from the quality flag file that comes with MYD21A3
lst_day_night_flag = [160, 161, 164, 165, 176, 177, 180, 181, 224, 225, 240, 245]

# Filtering
print('Filtering day LST and saving')
lst_day_filtered = lst_day.where(lst_day_qc.isin(lst_day_night_flag))
lst_day_filtered.to_netcdf(out_dir+'lst_day_filtered.nc')
t2 = time.time()
print(f'total time={t2-t1}')
print('Filtering night LST')
lst_night_filtered = lst_night.where(lst_night_qc.isin(lst_day_night_flag))
lst_night_filtered.to_netcdf(out_dir+'lst_night_filtered.nc')
print("All Done!")

lst_day_filtered = xr.open_dataarray(in_dir+'lst_day_filtered.nc',chunks=chunks)
lst_night_filtered = xr.open_dataarray(in_dir+'lst_night_filtered.nc',chunks=chunks)

#lst_day_tmp = lst_day_filtered.fillna(0)
#lst_night_tmp = lst_night_filtered.fillna(0)

lst_mean_filtered = (lst_day_filtered+lst_night_filtered)/2
print('###### Saving the mean #####')
lst_mean_filtered.to_netcdf(out_dir+'lst_mean_filtered.nc')

t2 = time.time()
print(f'--------- Taking the mean is done: {(t2-t1)/60} minutes ---------------------\n\n')

"""
print('------------- Group by year ------------------------------------------------\n')
t1 = time.time()

lst_mean_filtered = xr.open_dataarray(in_dir+'lst_mean_filtered.nc',chunks=chunks)
lst_day_filtered = xr.open_dataarray(in_dir+'lst_day_filtered.nc',chunks=chunks)
lst_night_filtered = xr.open_dataarray(in_dir+'lst_night_filtered.nc',chunks=chunks)

lst_day_annual = lst_day_filtered.groupby('time.year').mean(dim='time')
lst_night_annual = lst_night_filtered.groupby('time.year').mean(dim='time')
lst_mean_annual = lst_mean_filtered.groupby('time.year').mean(dim='time')

lst_day_annual.to_netcdf(out_dir+'lst_day_annual.nc')
lst_night_annual.to_netcdf(out_dir+'lst_night_annual.nc')
lst_mean_annual.to_netcdf(out_dir+'lst_mean_annual.nc')

t2 = time.time()
print(f"------------ Finished annual mean: {t2-t1} --------------------------------\n")

print('---------- Growing season (April-October)------------------------------\n')
# Taking the mean of the LST data from April to October. Selection of the month is just beacuse
# initital investigation of the Landsat NDVI data showed the satrt and end of the season. 
t1 = time.time()

lst_day_growing = modis_functions.growing_season(lst_day_filtered)
lst_night_growing = modis_functions.growing_season(lst_night_filtered)
# Below we could take the mean of lst_day_growing and lst_night_growing (which is probably faster).
# Hoewver beacuse there are NA values in both lst_day_filtered and lst_night_filtered, it is more 
# precise if we use the lst_mean_filtered, beacuse we count for all NAs in day and night original LST
lst_mean_growing = modis_functions.growing_season(lst_mean_filtered)

lst_day_growing.to_netcdf(out_dir+'lst_day_growing.nc')
lst_night_growing.to_netcdf(out_dir+'lst_night_growing.nc')
lst_mean_growing.to_netcdf(out_dir+'lst_mean_growing.nc')

t2 = time.time()
print(f'--------- Growing season Done: {(t2-t1)/60} minutes\n\n')

print('---------- Monthly mean ------------------------------------------------\n')
# taking the monthly mean of the LST data 
t1 = time.time()
lst_day_month_resample = lst_day_filtered.resample(time="1MS").mean()
lst_night_month_resample = lst_night_filtered.resample(time="1MS").mean()
lst_mean_month_resample = lst_mean_filtered.resample(time="1MS").mean()

#Saving
lst_day_month_resample.to_netcdf(out_dir + "lst_day_month_resample.nc")
lst_night_month_resample.to_netcdf(out_dir + "lst_night_month_resample.nc")
lst_mean_month_resample.to_netcdf(out_dir + "lst_mean_month_resample.nc")

t2 = time.time()
print(f'----------- Monthly mean Done: {(t2-t1)/60} minutes----------------------\n\n')

print("------------ Groupby month ---------------------------------------------------\n")
# Grouping the whole time series by month
t1 = time.time()
lst_day_month_group = lst_day_filtered.groupby("time.month").mean()
lst_night_month_group = lst_night_filtered.groupby("time.month").mean()
lst_mean_month_group = lst_mean_filtered.groupby("time.month").mean()

lst_day_month_group.to_netcdf(out_dir + "lst_day_month_group.nc")
lst_night_month_group.to_netcdf(out_dir + "lst_night_month_group.nc")
lst_mean_month_group.to_netcdf(out_dir + "lst_mean_month_group.nc")

t2 = time.time()
print(f'----------- Group month Done: {(t2-t1)/60} minutes -----------------------\n\n')

print('------------ Seasonal mean --------------------------------------------------\n')
# Taking the seasonal mean of the LST data
t1 = time.time()
lst_day_season_resample = modis_functions.weighted_season_resmaple(lst_day_filtered)
lst_night_season_resample = modis_functions.weighted_season_resmaple(lst_night_filtered)
lst_mean_season_resample = modis_functions.weighted_season_resmaple(lst_mean_filtered)

lst_day_season_resample.to_netcdf(out_dir + "lst_day_season_resample.nc")
lst_night_season_resample.to_netcdf(out_dir + "lst_night_season_resample.nc")
lst_mean_season_resample.to_netcdf(out_dir + "lst_mean_season_resample.nc")
t2 = time.time()

print(f' ----------- Seasonal mean Done: {(t2-t1)/60} minutes ---------------------\n\n')

print('------------- Group by season ------------------------------------------------\n')
t1 = time.time()
lst_day_season_group = modis_functions.weighted_season_group(lst_day_filtered)
lst_night_season_group = modis_functions.weighted_season_group(lst_night_filtered)
lst_mean_season_group = modis_functions.weighted_season_group(lst_mean_filtered)

lst_day_season_group.to_netcdf(out_dir + "lst_day_season_group.nc")
lst_night_season_group.to_netcdf(out_dir + "lst_night_season_group.nc")
lst_mean_season_group.to_netcdf(out_dir + "lst_mean_season_group.nc")

t2 = time.time()
print(f"------------ Finished seasonal group: {t2-t1}------------------------------\n")


print(f"############ ALL DONE!----> {(end_time-init_time)/60} minutes ###############")
