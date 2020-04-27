import os
import xarray as xr
import rioxarray
import time


in_dir = "/data/home/hamiddashti/mnt/nasa_above/working/landsat/time_series/key_images/"
out_dir = (
    "/data/home/hamiddashti/mnt/nasa_above/working/landsat/time_series/key_images/LST/"
)

"""
print("Kelvin to centigrade")
t1 = time.time()
lst = xr.open_dataarray(in_dir + "LST_original.nc", chunks={"x": 5000, "y": 5000})
lst = lst - 273.15
lst = lst.where((lst > -100) & (lst < 100))
t2 = time.time()
lst.to_netcdf(out_dir + "lst_centigrade.nc")
lst.rio.to_raster(out_dir + "lst_centigrade_tif.tif")
print(f"time to finish unit conversion:{t2-t1}")


print("####### Growing season (4-10)")
t1 = time.time()
lst_grouped = lst.where(
    lst.time.dt.month.isin([4, 5, 6, 7, 8, 9, 10])
)  # This line set other months than numbered to nan

lst_growing = lst.groupby("time.year").mean()
lst_growing.to_netcdf(out_dir + "lst_growing.nc")
lst_growing.rio.to_raster(out_dir + "lst_growing_tif.tif")
t2 = time.time()
print(f"finished growing season {t2-t1}")


print("####### Monthly")
t1 = time.time()
lst_monthly = lst.resample(time="1MS").mean()
lst_monthly.to_netcdf(out_dir + "lst_monthly.nc")
lst_monthly.rio.to_raster(out_dir + "lst_monthly_tif.tif")
t2 = time.time()
print(f"finished monthly in: {t2-t1}")

print("####### Groupby month")
t1 = time.time()
lst_monthly_group = lst.groupby("time.month").mean()
lst_monthly_group.to_netcdf(out_dir + "lst_monthly_group.nc")
lst_monthly_group.rio.to_raster(out_dir + "lst_monthly_group_tif.tif")
t2 = time.time()
print(f"finished monthly group in: {t2-t1}")

print("####### Seasonal")
t1 = time.time()
lst_season = lst.resample(time="QS-DEC").mean()
lst_season.to_netcdf(out_dir + "lst_season.nc")
lst_season.rio.to_raster(out_dir + "lst_season_tif.tif")
t2 = time.time()
print(f"finished season in: {t2-t1}")
"""
lst = xr.open_dataarray(out_dir + "LST_centigrade.nc", chunks={"x": 5000, "y": 5000})
print("######## Seasonal group ")
t1 = time.time()
lst_season_group = lst.groupby("time.season").mean()
lst_season_group.to_netcdf(out_dir + "lst_season_group.nc")
lst_season_group.rio.to_raster(out_dir + "lst_season_group_tif.tif")
t2 = time.time()
print(f"Finished seasonal group: {t2-t1}")

print("##### ALL DONE! ##########")
