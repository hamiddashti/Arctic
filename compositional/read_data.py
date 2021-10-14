import xarray as xr
import matplotlib.pylab as plt
import rioxarray
import numpy as np

in_dir = ("/data/home/hamiddashti/nasa_above/outputs/percent_cover/"
          "percent_cover_albert/")
out_dir = (
    "/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/test/")

lc = xr.open_dataarray(
    in_dir + "LULC_10_2003_2014.nc",
    decode_coords="all",
)

lst = xr.open_dataset(
    ("/data/home/hamiddashti/nasa_above/outputs/lst_processed/"
     "albers_proj_lst_mean_Annual.nc"),
    decode_coords="all",
)
lst = lst.rename({"x": "lon", "y": "lat"})
lst = lst["lst_mean_Annual"]
lc = lc.assign_coords({"lat": lc.lat, "lon": lc.lon})

lc_2003_2013 = lc.loc[[2003, 2013]]
# lc_2003_2013.to_netcdf(out_dir+"LC_2003_2013.nc")
lst_2003_2013 = lst.loc[[2003, 2013]]
# lst_2003_2013.to_netcdf(out_dir+"LST_2003_2013.nc")

lc_2003_2013_test = lc_2003_2013.isel(lat=np.arange(1382, 1482),
                                      lon=np.arange(1015, 1115))

lst_2003_2013_test = lst_2003_2013.isel(lat=np.arange(1382, 1482),
                                        lon=np.arange(1015, 1115))
# lst_2013_test_region = LST.isel(lat=np.arange(1382,1482),lon=np.arange(1015,1115),year=11)

lc_2003_2013_test.to_netcdf(out_dir + "lc_2003_2013_test.nc")
lst_2003_2013_test.to_netcdf(out_dir + "lst_2003_2013_test.nc")

lc_2003_2013_stacked = lc_2003_2013_test.stack(stacked=("lon", "lat"))
lc_2003_2013_stacked = lc_2003_2013_stacked.rename({"time": "year"})
lc_2003_2013_stacked = lc_2003_2013_stacked.reset_index("stacked")

lst_2003_2013_stacked = lst_2003_2013_test.stack(stacked=("lon", "lat"))
lst_2003_2013_stacked = lst_2003_2013_stacked.reset_index("stacked")

lc_2003_2013_stacked.to_netcdf(out_dir + "lc_2003_2013_stacked.nc")
lst_2003_2013_stacked.to_netcdf(out_dir + "lst_2003_2013_stacked.nc")

ds = xr.Dataset(data_vars={
    "lst_2003_2013": (("year", "obs"), lst_2003_2013_stacked),
    "lc_2003_2013": (("year", "band", "obs"), lc_2003_2013_stacked)
},
                coords={
                    "year": [2003, 2013],
                    "band": range(1, 11),
                    "obs": range(0, lst_2003_2013_stacked.shape[1])
                })
ds.to_netcdf(out_dir + "data.nc")
