import xarray

import xarray as xr

out_dir = ("/data/home/hamiddashti/nasa_above/outputs/Natural_Variability/"
           "Natural_Variability_Annual_outputs/geographic/02_percent/")
dlst = xr.open_dataarray(out_dir + "dlst_mean_lcc.nc")
dlst.notnull().sum()
changed = xr.open_dataarray(out_dir + "changed_pixels_01percent.nc")
changed.sum()