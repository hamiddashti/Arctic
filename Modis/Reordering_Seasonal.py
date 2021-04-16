import xarray as xr
import rioxarray
import pandas as pd
import numpy as np

# Puting all specific month of each year in one file
# For example all the januaries from 2003 to 2015 in one file

# ----------------- LST DAY -------------------------------------
in_dir = "/data/ABOVE/Final_data/LST_Final/LST/Seasonal_Mean/"
out_dir = "/data/ABOVE/Final_data/LST_Final/LST/Seasonal_Mean/"
LST = xr.open_dataarray(in_dir + "lst_day_season_resample.nc")
seasons = LST.time.dt.season
LST[seasons=='DJF'].to_netcdf(out_dir + "LST_Day_DJF.nc")
LST[seasons=='MAM'].to_netcdf(out_dir + "LST_Day_MAM.nc")
LST[seasons=='JJA'].to_netcdf(out_dir + "LST_Day_JJA.nc")
LST[seasons=='SON'].to_netcdf(out_dir + "LST_Day_SON.nc")

# ----------------- LST NIGHT -------------------------------------
in_dir = "/data/ABOVE/Final_data/LST_Final/LST/Seasonal_Mean/"
out_dir = "/data/ABOVE/Final_data/LST_Final/LST/Seasonal_Mean/"
LST = xr.open_dataarray(in_dir + "lst_night_season_resample.nc")
seasons = LST.time.dt.season
LST[seasons=='DJF'].to_netcdf(out_dir + "LST_Night_DJF.nc")
LST[seasons=='MAM'].to_netcdf(out_dir + "LST_Night_MAM.nc")
LST[seasons=='JJA'].to_netcdf(out_dir + "LST_Night_JJA.nc")
LST[seasons=='SON'].to_netcdf(out_dir + "LST_Night_SON.nc")


# ----------------- LST MEAN -------------------------------------
in_dir = "/data/ABOVE/Final_data/LST_Final/LST/Seasonal_Mean/"
out_dir = "/data/ABOVE/Final_data/LST_Final/LST/Seasonal_Mean/"
LST = xr.open_dataarray(in_dir + "lst_mean_season_resample.nc")
seasons = LST.time.dt.season
LST[seasons=='DJF'].to_netcdf(out_dir + "LST_Mean_DJF.nc")
LST[seasons=='MAM'].to_netcdf(out_dir + "LST_Mean_MAM.nc")
LST[seasons=='JJA'].to_netcdf(out_dir + "LST_Mean_JJA.nc")
LST[seasons=='SON'].to_netcdf(out_dir + "LST_Mean_SON.nc")


# ----------------- ET -------------------------------------
in_dir = "/data/ABOVE/Final_data/ET_Final/Seasonal_ET/"
out_dir = "/data/ABOVE/Final_data/ET_Final/Seasonal_ET/"
Et_comp = ["EC","EI","ET", "ES", "EW", "ET"]
for k in Et_comp:
	print(k)
	da = xr.open_dataarray(in_dir + k + "_Season.nc")
	seasons = da.time.dt.season
	da[seasons=='DJF'].to_netcdf(out_dir+k+'_Mean_'+'DJF'+'.nc')
	da[seasons=='MAM'].to_netcdf(out_dir+k+'_Mean_'+'MAM'+'.nc')
	da[seasons=='JJA'].to_netcdf(out_dir+k+'_Mean_'+'JJA'+'.nc')
	da[seasons=='SON'].to_netcdf(out_dir+k+'_Mean_'+'SON'+'.nc')

# ----------------- Albedo -------------------------------------
in_dir = "/data/ABOVE/Final_data/ALBEDO_Final/"
out_dir = "/data/ABOVE/Final_data/ALBEDO_Final/Seasonal_Albedo/"
albedo = xr.open_dataarray(in_dir + "Albedo_season_resample.nc")
season = albedo.time.dt.season

albedo[seasons=='DJF'].to_netcdf(out_dir + "Albedo_Mean_DJF.nc")
albedo[seasons=='MAM'].to_netcdf(out_dir + "Albedo_Mean_MAM.nc")
albedo[seasons=='JJA'].to_netcdf(out_dir + "Albedo_Mean_JJA.nc")
albedo[seasons=='SON'].to_netcdf(out_dir + "Albedo_Mean_SON.nc")
