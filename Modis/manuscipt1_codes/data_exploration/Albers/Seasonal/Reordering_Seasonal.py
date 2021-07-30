import xarray as xr
import rioxarray
import pandas as pd
import numpy as np

# Puting all specific month of each year in one file
# For example all the januaries from 2003 to 2015 in one file

# ----------------- LST DAY -------------------------------------
in_dir = "/data/ABOVE/Final_data/LST_Final/LST/Seasonal_Mean/albers/"
out_dir = "/data/ABOVE/Final_data/LST_Final/LST/Seasonal_Mean/albers/"
LST = xr.open_dataarray(in_dir + "albers_proj_lst_day_season_resample.nc",
                        decode_coords="all")
seasons = LST.time.dt.season
LST[seasons == 'DJF'].to_netcdf(out_dir + "LST_Day_DJF.nc")
LST[seasons == 'MAM'].to_netcdf(out_dir + "LST_Day_MAM.nc")
LST[seasons == 'JJA'].to_netcdf(out_dir + "LST_Day_JJA.nc")
LST[seasons == 'SON'].to_netcdf(out_dir + "LST_Day_SON.nc")

# ----------------- LST NIGHT -------------------------------------
in_dir = "/data/ABOVE/Final_data/LST_Final/LST/Seasonal_Mean/albers/"
out_dir = "/data/ABOVE/Final_data/LST_Final/LST/Seasonal_Mean/albers/"
LST = xr.open_dataarray(in_dir + "albers_proj_lst_night_season_resample.nc",
                        decode_coords="all")
seasons = LST.time.dt.season
LST[seasons == 'DJF'].to_netcdf(out_dir + "LST_Night_DJF.nc")
LST[seasons == 'MAM'].to_netcdf(out_dir + "LST_Night_MAM.nc")
LST[seasons == 'JJA'].to_netcdf(out_dir + "LST_Night_JJA.nc")
LST[seasons == 'SON'].to_netcdf(out_dir + "LST_Night_SON.nc")

# ----------------- LST MEAN -------------------------------------
in_dir = "/data/ABOVE/Final_data/LST_Final/LST/Seasonal_Mean/albers/"
out_dir = "/data/ABOVE/Final_data/LST_Final/LST/Seasonal_Mean/albers/"
LST = xr.open_dataarray(in_dir + "albers_proj_lst_mean_season_resample.nc",
                        decode_coords="all")
seasons = LST.time.dt.season
LST[seasons == 'DJF'].to_netcdf(out_dir + "LST_Mean_DJF.nc")
LST[seasons == 'MAM'].to_netcdf(out_dir + "LST_Mean_MAM.nc")
LST[seasons == 'JJA'].to_netcdf(out_dir + "LST_Mean_JJA.nc")
LST[seasons == 'SON'].to_netcdf(out_dir + "LST_Mean_SON.nc")

# ----------------- ET -------------------------------------
in_dir = "/data/ABOVE/Final_data/ET_Final/Seasonal_ET/albers/"
out_dir = "/data/ABOVE/Final_data/ET_Final/Seasonal_ET/albers/"
Et_comp = ["EC", "EI", "ET", "ES", "EW"]
for k in Et_comp:
    print(k)
    da = xr.open_dataarray(in_dir + "albers_proj_" + k + "_Season.nc",
                           decode_coords="all")
    seasons = da.time.dt.season
    da[seasons == 'DJF'].to_netcdf(out_dir + "albers_proj_" + k + '_DJF' +
                                   '.nc')
    da[seasons == 'MAM'].to_netcdf(out_dir + "albers_proj_" + k + '_MAM' +
                                   '.nc')
    da[seasons == 'JJA'].to_netcdf(out_dir + "albers_proj_" + k + '_JJA' +
                                   '.nc')
    da[seasons == 'SON'].to_netcdf(out_dir + "albers_proj_" + k + '_SON' +
                                   '.nc')

# ----------------- Albedo -------------------------------------
in_dir = ("/data/ABOVE/Final_data/ALBEDO_Final/Seasonal_Albedo/albers/")
out_dir = ("/data/ABOVE/Final_data/ALBEDO_Final/Seasonal_Albedo/albers/")
albedo = xr.open_dataarray(in_dir + "final_seasonal_albedo.nc")
seasons = albedo.time.dt.season
albedo[seasons == 'DJF'].to_netcdf(out_dir + "Albedo_Mean_DJF.nc")
albedo[seasons == 'MAM'].to_netcdf(out_dir + "Albedo_Mean_MAM.nc")
albedo[seasons == 'JJA'].to_netcdf(out_dir + "Albedo_Mean_JJA.nc")
albedo[seasons == 'SON'].to_netcdf(out_dir + "Albedo_Mean_SON.nc")
