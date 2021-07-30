# ----------------------------------------------------------
#  The main goal is to regrid RS products to CLM resolution
# ----------------------------------------------------------

# Import libraries
# -----------------
import xarray as xr
import matplotlib.pylab as plt
import numpy as np
import xesmf as xe
from dask.diagnostics import ProgressBar
import matplotlib.pylab as plt
import glob
from dask.distributed import Client, LocalCluster

# cluster = LocalCluster()
client = Client()

# Define functions
#-----------------

# Filter LAI based on QC
#-----------------------
# This should be done before regriding

# Open the files using xarray

in_dir = "/data/ABOVE/MODIS/LAI_CLM_DOMAIN/"
out_dir = ("/data/ABOVE/MODIS/LAI_CLM_DOMAIN/processed/")

lai_ds = xr.open_dataset(in_dir + 'LAI_500m_CLM_Domain.nc',
                         chunks={
                             "lat": -1,
                             "lon": -1,
                             "time": "auto"
                         })

lai = lai_ds['Lai_500m']
std = lai_ds['LaiStdDev_500m']
qc = lai_ds['FparLai_QC']
lai_filtered_lazy = lai.where(qc <= 50)
lai_filtered_lazy.to_netcdf(out_dir + "lai_filtered.nc")

std_filtered_lazy = std.where(qc <= 50)
std_filtered_lazy.to_netcdf(out_dir + "std_filtered.nc")

# Regriding filtered data to the CLM domain
clm_image = xr.open_dataset(in_dir + 'PPE.spin81.h0.2017-02-01-00000.nc')
lai_filtered = xr.open_dataarray(out_dir + "lai_filtered.nc",
                                 chunks={
                                     "lat": -1,
                                     "lon": -1,
                                     "time": "auto"
                                 })
std_filtered = xr.open_dataarray(out_dir + "std_filtered.nc",
                                 chunks={
                                     "lat": -1,
                                     "lon": -1,
                                     "time": "auto"
                                 })

ds_out = xr.Dataset({
    'lat': (['lat'], clm_image['lat'].values),
    'lon': (['lon'], clm_image['lon'].values - 360)
})
ds_in = xr.Dataset({
    'lat': (['lat'], lai_filtered['lat'].values),
    'lon': (['lon'], lai_filtered['lon'].values)
})
# Create the regridder 
regridder = xe.Regridder(ds_in, ds_out, 'bilinear')
regridder.to_netcdf(out_dir+"regridder_entire_clm_domain.nc")

lai_regrided = regridder(lai_filtered)
std_regrided = regridder(std_filtered)

ds_lai_final = lai_regrided.where(lai_regrided.notnull(), -9999.0)
ds_std_final = std_regrided.where(std_regrided.notnull(), -9999.0)

ds_lai_final.to_netcdf(out_dir + "final_resampled/lai_025deg_2011_2020.nc")
ds_std_final.to_netcdf(out_dir + "final_resampled/lai_std_025deg_2011_2020.nc")
print("all done!")

