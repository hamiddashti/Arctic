# -*- coding: utf-8 -*-
# in this code we do some simple statistics for each landsat ARD tile


import numpy as np
import xarray as xr
import landsat_functions

# in_path = "P:\\nasa_above\\working\\landsat\\time_series\\"
# out_path = "P:\\nasa_above\\working\\landsat\\time_series\\tmp_figures\\"
in_path = "/data/home/hamiddashti/mnt/nasa_above/working/landsat/time_series/"
out_path = (
    "/data/home/hamiddashti/mnt/nasa_above/working/landsat/time_series/tmp_figures/"
)


f_tmp = "NDVI_processed.nc"
fname = in_path + f_tmp
tile = "008003"
nrow = 50
ncol = 50
# start = timer()
ndvi = xr.open_dataset(fname)[
    "__xarray_dataarray_variable__"
]  #  #,chunks={'x': ncol, 'y': nrow}
tmp_ndvi = ndvi.isel(time=np.arange(550, 601))
tmp_ndvi = tmp_ndvi.isel(x=np.arange(0, 5), y=np.arange(0, 5))

landsat_functions.month_stat(tmp_ndvi, "NDVI", out_path)
landsat_functions.season_stat(tmp_ndvi, "NDVI", out_path)

f_tmp = "LST_processed.nc"
fname = in_path + f_tmp
tile = "008003"
nrow = 50
ncol = 50
# start = timer()
lst = xr.open_dataset(fname, chunks={"x": ncol, "y": nrow})[
    "__xarray_dataarray_variable__"
]  #  #,chunks={'x': ncol, 'y': nrow}
lst = lst.where(lst > 0)
lst = lst - 273.15
tmp_lst = lst.isel(time=np.arange(0, 601))
tmp_lst = tmp_lst.isel(x=np.arange(0, 50), y=np.arange(0, 50))

landsat_functions.season_stat(tmp_lst, "LST", out_path)
landsat_functions.month_stat(tmp_lst, "LST", out_path)
