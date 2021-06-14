import xarray as xr
import matplotlib.pylab as plt
import numpy as np
from pylab import savefig as save

da = xr.open_dataarray("da_concat_81.nc")
annual = xr.open_rasterio("annual_albedo_tile_81.tif")
seasonal = xr.open_rasterio("seasonal_albedo_tile_81.tif")
da.loc["2003-4":"2003-6"].isel(x=100,y=100).mean() 
seasonal.isel(band=3)


da_annual = da.groupby('time.year').mean(dim='time').squeeze()
da_annual.loc[2003].isel(x=100,y=100).mean()
annual.isel(band=3,x=100,y=100) 
