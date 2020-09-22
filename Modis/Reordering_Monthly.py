import xarray as xr
import rioxarray
import pandas as pd
import numpy as np

# Puting all specific month of each year in one file
# For example all the januaries from 2003 to 2015 in one file

Months = ["Jan", "Feb", "Mar", "Apr","May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
# ----------------- LST -------------------------------------
in_dir = "/data/ABOVE/Final_data/LST_Final/LST/Monthly_Mean/"
out_dir = "/data/ABOVE/Final_data/LST_Final/LST/Monthly_Mean/"
LST = xr.open_dataarray(in_dir + "lst_mean_month_resample.nc")
month = LST.time.dt.month.values
for i in np.arange(1, 13):
	print(f"converting month {Months[i-1]}")
	a = np.where(month == i)[0]
	tmp = LST.isel(time=a)
	tmp = tmp.rename({"time":"year"})
	tmp.to_netcdf(out_dir + "/" + "LST_Mean_" + Months[i - 1] + ".nc")

# ----------------- ET -------------------------------------
in_dir = "/data/ABOVE/Final_data/ET_Final/Monthly_ET/"
out_dir = "/data/ABOVE/Final_data/ET_Final/Monthly_ET/"
Et_comp = ["EC","EI","ET", "ES", "EW", "ET"]
for k in Et_comp:
	print(k)
	da = xr.open_dataarray(in_dir + k + "_Monthly.nc")
	date = pd.to_datetime(da.time.values)
	da = da.assign_coords({"time": date})
	month = da.time.dt.month.values
	for i in np.arange(1, 13):
		print(f"converting month {Months[i-1]}")
		a = np.where(month == i)[0]
		tmp = da.isel(time=a)
		tmp = tmp.rename({"time":"year"})
		tmp = tmp.assign_coords({"year":tmp.year.dt.year})
		tmp.to_netcdf(out_dir + "/" + k + "_Mean_" + Months[i - 1] + ".nc")

# ----------------- Albedo -------------------------------------
in_dir = "/data/ABOVE/Final_data/ALBEDO_Final/Monthly_Albedo/"
out_dir = "/data/ABOVE/Final_data/ALBEDO_Final/Monthly_Albedo/"
albedo = xr.open_dataarray(in_dir + "Albedo_month_resample.nc")
month = albedo.time.dt.month.values
for i in np.arange(1, 13):
	print(f"converting month {Months[i-1]}")
	a = np.where(month == i)[0]
	tmp = albedo.isel(time=a)
	tmp = tmp.rename({"time":"year"})
	tmp = tmp.assign_coords({"year":tmp.year.dt.year})
	tmp.to_netcdf(out_dir + "/" + "Albedo_" + Months[i - 1] + ".nc")
