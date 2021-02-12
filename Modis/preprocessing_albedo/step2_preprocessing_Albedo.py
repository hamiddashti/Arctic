#
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

t1 = time.time()
in_dir = '/data/ABOVE/MODIS/blue_sky_albedo/orders/df99d6e7e26fb666143dbf2de2de4707/Albedo_Boreal_North_America/NetCDF2/'
out_dir = '/data/ABOVE/MODIS/blue_sky_albedo/orders/df99d6e7e26fb666143dbf2de2de4707/Albedo_Boreal_North_America/Albedo_processed/'
for i in np.arange(1, 176):
	
	fname = "Albedo_Tile_" + str(i)+".nc"
	print('Processing --> '+fname)
	ds = xr.open_dataset(in_dir+fname)
	#scaling factor
	ds = ds * 0.001
	
	# ------------ Growing season ------------------------------
	albedo_growing = modis_functions.growing_season(ds)
	albedo_growing.rio.to_raster(out_dir + "Albedo_growing_"+str(i)+".tif")
	# ---------------------------------------------------------

	# ------------ Monthly resampled ----------------------------
	albedo_month_resample = ds.resample(time="1MS").mean()
	albedo_month_resample.rio.to_raster(out_dir + "Albedo_month_resample_"+str(i)+".tif")
	# ----------------------------------------------------------

	albedo_month_group = ds.groupby("time.month").mean()
	albedo_month_group.rio.to_raster(out_dir + "Albedo_month_group_"+str(i)+".tif")
	# ----------------------------------------------------------

	# ----------- Seasonal mean --------------------------------
	albedo_season_resample = modis_functions.weighted_season_resmaple(ds)
	albedo_season_resample.rio.to_raster(out_dir + "Albedo_season_resample"+str(i)+".tif")
	# ----------------------------------------------------------

	# ----------- Group by the season --------------------------
	albedo_season_group = modis_functions.weighted_season_group(ds)
	albedo_season_group = albedo_season_group.where(albedo_season_group != 0)
	albedo_season_group.rio.to_raster(out_dir + "Albedo_season_group_"+str(i)+".tif")
	# ----------------------------------------------------------

	# ----------- Group by year --------------------------------
	albedo_annual = ds.groupby("time.year").mean(dim="time")
	albedo_annual.rio.to_raster(out_dir + "Albedo_annual_"+str(i)+".tif")

t2= time.time()
print(f"\n It is done! in {(t2-t1)/60} minutes")

