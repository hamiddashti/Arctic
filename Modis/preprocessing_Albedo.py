import xarray as xr
import rioxarray
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import modis_functions

t1 = time.time()
in_dir = '/xdisk/davidjpmoore/hamiddashti/data/Albedo/NetCDF2/'
out_dir = '/xdisk/davidjpmoore/hamiddashti/data/Albedo/Albedo_processed/'
for i in np.arange(138, 139):
	
	fname = "Albedo_Tile_" + str(i)+".nc"
	print('Processing --> '+fname)
	ds = xr.open_dataset(in_dir+fname)
	#scaling factor
	ds = ds * 0.001
	
	# ------------ Growing season ------------------------------
	albedo_growing = modis_functions.growing_season(ds)
	albedo_growing.to_netcdf(out_dir + "Albedo_growing_"+str(i)+".nc")
	# ---------------------------------------------------------

	# ------------ Monthly resampled ----------------------------
	albedo_month_resample = ds.resample(time="1MS").mean()
	albedo_month_resample.to_netcdf(out_dir + "Albedo_month_resample_"+str(i)+".nc")
	# ----------------------------------------------------------

	albedo_month_group = ds.groupby("time.month").mean()
	albedo_month_group.to_netcdf(out_dir + "Albedo_month_group_"+str(i)+".nc")
	# ----------------------------------------------------------

	# ----------- Seasonal mean --------------------------------
	albedo_season_resample = modis_functions.weighted_season_resmaple(ds)
	albedo_season_resample.to_netcdf(out_dir + "Albedo_season_resample"+str(i)+".nc")
	# ----------------------------------------------------------

	# ----------- Group by the season --------------------------
	albedo_season_group = modis_functions.weighted_season_group(ds)
	albedo_season_group = albedo_season_group.where(albedo_season_group != 0)
	albedo_season_group.to_netcdf(out_dir + "Albedo_season_group_"+str(i)+".nc")
	# ----------------------------------------------------------

	# ----------- Group by year --------------------------------
	albedo_annual = ds.groupby("time.year").mean(dim="time")
	albedo_annual.to_netcdf(out_dir + "Albedo_annual_"+str(i)+".nc")

t2= time.time()
print(f"\n It is done! in {(t2-t1)/60} minutes")

ds = xr.open_dataset('Albedo_annual_138.nc')
ds['Albedo'].isel(year=3,x=100,y=100)


