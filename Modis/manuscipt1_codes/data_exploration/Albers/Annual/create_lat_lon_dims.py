import xarray as xr
import numpy as np

changed = xr.open_dataarray(
    ("/data/home/hamiddashti/nasa_above/outputs/Natural_Variability/"
     "Natural_Variability_Annual_outputs/Albers/changed.nc"))
x = np.arange(0, changed.shape[0])
xx = np.transpose(np.array([x] * changed.shape[1]))
changed.copy(data=xx).to_netcdf(
    ("/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/test/"
     "lat_index.nc"))
changed.to_netcdf(
    ("/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/test/"
     "changed.nc"))
y = np.arange(0, changed.shape[1])
yy = np.array([y] * changed.shape[0])
changed.copy(data=yy).to_netcdf(
    ("/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/test/"
     "lon_index.nc"))
