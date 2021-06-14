import xarray as xr
import modis_functions

in_dir = '/xdisk/davidjpmoore/hamiddashti/data/LST/'
out_dir = '/xdisk/davidjpmoore/hamiddashti/data/LST/'

# We ran the analyses in Above original projection as well. However later
# decided to drop this analyses. In case to rerun them we can uncomment the 
# following path. 
# in_dir = '/xdisk/davidjpmoore/hamiddashti/data/LST_Albert/'
# out_dir = '/xdisk/davidjpmoore/hamiddashti/data/LST_Albert/processed/'
# in_dir = "/data/ABOVE/ABoVE_Final_Data/LST/"
# out_dir = "/data/ABOVE/ABoVE_Final_Data/LST/processed/"
#print('--------- Filtering the LST data based on the quality flag) ----------\n')

chunks = ({'time': 10, 'ydim': 2692, 'xdim': 8089})

# Use the MYD21A2.006_1km_aid0001.nc to extract day and night data
LST = xr.open_dataset(in_dir + "MYD21A2.006_1km_aid0001.nc", chunks=chunks)
# lst_day_ds = xr.open_dataset(in_dir + 'MYD_LST_Day.nc', chunks=chunks)
# lst_night_ds = xr.open_dataset(in_dir + 'MYD_LST_Night.nc', chunks=chunks)

lst_day = LST['LST_Day_1KM']
lst_day_qc = LST['QC_Day']
lst_night = LST['LST_Night_1KM']
lst_night_qc = LST['QC_Night']

# These numbers comes from the quality flag file that comes with MYD21A3
lst_day_night_flag = [
    144, 145, 160, 161, 164, 165, 176, 177, 180, 181, 208, 209, 224, 225, 240,
    244, 245
]

# Filtering
print('Filtering day LST and saving')
lst_day_filtered = lst_day.where(lst_day_qc.isin(lst_day_night_flag))
lst_day_filtered.to_netcdf(out_dir + 'lst_day_filtered.nc')
print('Filtering night LST')
lst_night_filtered = lst_night.where(lst_night_qc.isin(lst_day_night_flag))
lst_night_filtered.to_netcdf(out_dir + 'lst_night_filtered.nc')
print("All Done!")

lst_day_filtered = xr.open_dataarray(out_dir + 'lst_day_filtered.nc',
                                     chunks=chunks)
lst_night_filtered = xr.open_dataarray(out_dir + 'lst_night_filtered.nc',
                                       chunks=chunks)

lst_mean_filtered = (lst_day_filtered + lst_night_filtered) / 2
print('###### Saving the mean #####')
lst_mean_filtered.to_netcdf(out_dir + 'lst_mean_filtered.nc')

print(
    '--------- Group by year ------------------------------------------------\n'
)

lst_mean_filtered = xr.open_dataarray(out_dir + 'lst_mean_filtered.nc',
                                      chunks=chunks)
lst_day_filtered = xr.open_dataarray(out_dir + 'lst_day_filtered.nc',
                                     chunks=chunks)
lst_night_filtered = xr.open_dataarray(out_dir + 'lst_night_filtered.nc',
                                       chunks=chunks)

lst_day_annual = lst_day_filtered.groupby('time.year').mean(dim='time')
lst_night_annual = lst_night_filtered.groupby('time.year').mean(dim='time')
lst_mean_annual = lst_mean_filtered.groupby('time.year').mean(dim='time')

lst_day_annual.to_netcdf(out_dir + 'lst_day_Annual.nc')
lst_night_annual.to_netcdf(out_dir + 'lst_night_Annual.nc')
lst_mean_annual.to_netcdf(out_dir + 'lst_mean_Annual.nc')

print(
    '---------- Growing season (April-October)------------------------------\n'
)
# Taking the mean of the LST data from April to October. Selection of the month is just beacuse
# initital investigation of the Landsat NDVI data showed the satrt and end of the season.

lst_day_growing = modis_functions.growing_season(lst_day_filtered)
lst_night_growing = modis_functions.growing_season(lst_night_filtered)
# Below we could take the mean of lst_day_growing and lst_night_growing (which is probably faster).
# Hoewver beacuse there are NA values in both lst_day_filtered and lst_night_filtered, it is more
# precise if we use the lst_mean_filtered, beacuse we count for all NAs in day and night original LST
lst_mean_growing = modis_functions.growing_season(lst_mean_filtered)

lst_day_growing.to_netcdf(out_dir + 'lst_day_Growing.nc')
lst_night_growing.to_netcdf(out_dir + 'lst_night_Growing.nc')
lst_mean_growing.to_netcdf(out_dir + 'lst_mean_Growing.nc')

print(
    '---------- Monthly mean ------------------------------------------------\n'
)
# taking the monthly mean of the LST data
lst_day_month_resample = lst_day_filtered.resample(time="1MS").mean()
lst_night_month_resample = lst_night_filtered.resample(time="1MS").mean()
lst_mean_month_resample = lst_mean_filtered.resample(time="1MS").mean()

#Saving
lst_day_month_resample.to_netcdf(out_dir + "lst_day_month_resample.nc")
lst_night_month_resample.to_netcdf(out_dir + "lst_night_month_resample.nc")
lst_mean_month_resample.to_netcdf(out_dir + "lst_mean_month_resample.nc")

print(
    "------------ Groupby month ---------------------------------------------------\n"
)
# Grouping the whole time series by month

lst_day_month_group = lst_day_filtered.groupby("time.month").mean()
lst_night_month_group = lst_night_filtered.groupby("time.month").mean()
lst_mean_month_group = lst_mean_filtered.groupby("time.month").mean()

lst_day_month_group.to_netcdf(out_dir + "lst_day_month_group.nc")
lst_night_month_group.to_netcdf(out_dir + "lst_night_month_group.nc")
lst_mean_month_group.to_netcdf(out_dir + "lst_mean_month_group.nc")

print(
    '------------ Seasonal mean --------------------------------------------------\n'
)
# Taking the seasonal mean of the LST data

lst_day_season_resample = modis_functions.weighted_season_resmaple(
    lst_day_filtered)
lst_night_season_resample = modis_functions.weighted_season_resmaple(
    lst_night_filtered)
lst_mean_season_resample = modis_functions.weighted_season_resmaple(
    lst_mean_filtered)

lst_day_season_resample.to_netcdf(out_dir + "lst_day_season_resample.nc")
lst_night_season_resample.to_netcdf(out_dir + "lst_night_season_resample.nc")
lst_mean_season_resample.to_netcdf(out_dir + "lst_mean_season_resample.nc")

print(
    '------------- Group by season ------------------------------------------------\n'
)

lst_day_season_group = modis_functions.weighted_season_group(lst_day_filtered)
lst_night_season_group = modis_functions.weighted_season_group(
    lst_night_filtered)
lst_mean_season_group = modis_functions.weighted_season_group(
    lst_mean_filtered)

lst_day_season_group.to_netcdf(out_dir + "lst_day_season_group.nc")
lst_night_season_group.to_netcdf(out_dir + "lst_night_season_group.nc")
lst_mean_season_group.to_netcdf(out_dir + "lst_mean_season_group.nc")

print(f"All Done!\n")
