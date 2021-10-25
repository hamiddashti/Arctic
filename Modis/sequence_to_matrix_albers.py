import xarray as xr
import matplotlib.pylab as plt
import numpy as np

da = xr.open_dataarray(
    ("/data/home/hamiddashti/nasa_above/outputs/Natural_Variability/"
     "Natural_Variability_Seasonal_Outputs/albers/DJF/dlst_total_DJF.nc"))
x = da.shape[0]
y = da.shape[1]
a = np.arange(0, (x * y)).reshape((x, y), order="F")
b = da.copy(data=a)
b.to_netcdf(
    "/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/test/indices.nc"
)
