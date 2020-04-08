# -*- coding: utf-8 -*-
# in this code we do some simple statistics for each landsat ARD tile


import numpy as np
import xarray as xr
import landsat_functions
from timeit import default_timer as timer

# in_path = "P:\\nasa_above\\working\\landsat\\time_series\\"
# out_path = "P:\\nasa_above\\working\\landsat\\time_series\\tmp_figures\\"

start = timer()
in_path = "/data/home/hamiddashti/mnt/nasa_above/working/landsat/time_series/"
out_path = (
    "/data/home/hamiddashti/mnt/nasa_above/working/landsat/time_series/tmp_figures/"
)


f_tmp = "NDVI_processed.nc"
fname = in_path + f_tmp
tile = "008003"
nrow = 500
ncol = 500
print("######### start processing NDVI ####################################")
ndvi = xr.open_dataset(fname, chunks={"x": ncol, "y": nrow})[
    "__xarray_dataarray_variable__"
]  #  #,chunks={'x': ncol, 'y': nrow}
# tmp_ndvi = ndvi.isel(time=np.arange(550, 601))
# tmp_ndvi = tmp_ndvi.isel(x=np.arange(0, 5), y=np.arange(0, 5))

landsat_functions.month_stat(ndvi, "NDVI", out_path)
landsat_functions.season_stat(ndvi, "NDVI", out_path)

print("######### start processing LST ####################################")
f_tmp = "LST_processed.nc"
fname = in_path + f_tmp
tile = "008003"
nrow = 500
ncol = 500
lst = xr.open_dataset(fname, chunks={"x": ncol, "y": nrow})[
    "__xarray_dataarray_variable__"
]  #  #,chunks={'x': ncol, 'y': nrow}
lst = lst.where(lst > 0)
lst = lst - 273.15
# tmp_lst = lst.isel(time=np.arange(0, 601))
# tmp_lst = tmp_lst.isel(x=np.arange(0, 50), y=np.arange(0, 50))

landsat_functions.season_stat(lst, "LST", out_path)
landsat_functions.month_stat(lst, "LST", out_path)

end = timer()
elapsed = end - start
print("elapsed time for chunk size 5000*5000 is:" + str(elapsed) + " second")
