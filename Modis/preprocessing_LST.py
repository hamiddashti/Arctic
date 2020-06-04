import xarray as xr
import rioxarray
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import modis_functions

init_time = time.time()


in_dir = 'F:\\MYD21A2\\'
out_dir = 'F:\\MYD21A2\\outputs\\LST\\'


print('--------- Filtering the LST data based on the quality flag) ----------\n')
t1 = time.time()
lst_day = xr.open_dataarray(in_dir+'LST_Day_0804.nc',chunks=({'y': 272, 'x': 731}))
lst_day_qc = xr.open_dataarray(in_dir+'QC_Day_0804.nc',chunks=({'y': 272, 'x': 731}))
lst_night = xr.open_dataarray(in_dir+'LST_Night_0804.nc',chunks=({'y': 272, 'x': 731}))
lst_night_qc = xr.open_dataarray(in_dir+'QC_Night_0804.nc',chunks=({'y': 272, 'x': 731}))

# These numbers comes from the quality flag file that comes with MYD21A3
lst_day_night_flag = [160, 161, 164, 165, 176, 177, 180, 181, 224, 225, 240, 245]

# Filtering
lst_day_filtered = lst_day.where(lst_day_qc.isin(lst_day_night_flag))*0.02
lst_night_filtered = lst_night.where(lst_night_qc.isin(lst_day_night_flag))*0.02
tmp_df = xr.concat([lst_day_filtered, lst_night_filtered],'Average')

# Note here that we DO NOT skip the NA values. So we set the mean as NA if either of day or night is NA
lst_mean_filtered = tmp_df.mean('Average',skipna = False) 
# saving the results to the disk
lst_day_filtered.to_netcdf(out_dir+'LST_Day_filtered.nc')
lst_night_filtered.to_netcdf(out_dir+'LST_Night_filtered.nc')
lst_mean_filtered.to_netcdf(out_dir+'lst_mean_filtered.nc')

t2 = time.time()
print(f'--------- Filtering Done: {(t2-t1)/60} minutes ---------------------\n\n')


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

#Saving 
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

print('------------- Group by year ------------------------------------------------\n')
t1 = time.time()
lst_day_annual = lst_day_filtered.groupby('time.year').mean(dim='time')
lst_night_annual = lst_night_filtered.groupby('time.year').mean(dim='time')
lst_mean_annual = lst_mean_filtered.groupby('time.year').mean(dim='time')

lst_day_annual.to_netcdf(out_dir+'lst_day_annual.nc')
lst_night_annual.to_netcdf(out_dir+'lst_night_annual.nc')
lst_mean_annual.to_netcdf(out_dir+'lst_mean_annual.nc')

t2 = time.time()
print(f"------------ Finished annual mean: {t2-t1} --------------------------------\n")

end_time = time.time()
print(f"############ ALL DONE!----> {(end_time-init_time)/60} minutes ###############")
