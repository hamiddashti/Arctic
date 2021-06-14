import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

in_dir = ("/data/home/hamiddashti/nasa_above/outputs/Natural_Variability/"
          "Natural_Variability_Annual_outputs/geographic/02_percent/")
out_dir = "/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/test/"


def outliers_index(data, m=2):
    """
    Returns true if a value is outlier
    https://www.itl.nist.gov/div898/handbook/eda/section3/eda356.htm#MAD
    :param int data: numpy array
    :param int m: # of std to include data 
    """
    import numpy as np
    d = np.abs(data - np.nanmedian(data))
    mdev = np.nanmedian(d)
    s = d / mdev if mdev else 0.
    return ~(s < m)


dlst_nv = xr.open_dataarray(in_dir + "dlst_mean_nv.nc")
dlst_lcc = xr.open_dataarray(in_dir + "dlst_mean_lcc.nc")
dlst_total = xr.open_dataarray(in_dir + "dlst_mean_changed.nc")
ct = xr.open_dataset(
    "/data/home/hamiddashti/nasa_above/outputs/Sensitivity/EndPoints/Annual/Geographic/"
    "02_percent/Confusion_Table_final_02precent.nc")

I_dlst_nv = outliers_index(dlst_nv, 2)
I_dlst_lcc = outliers_index(dlst_lcc, 2)
I_dlst_total = outliers_index(dlst_total, 2)

dlst_nv_clean = dlst_nv.where((I_dlst_nv == False) & (I_dlst_lcc == False)
                              & (I_dlst_total == False))
dlst_lcc_clean = dlst_lcc.where((I_dlst_nv == False) & (I_dlst_lcc == False)
                                & (I_dlst_total == False))
dlst_total_clean = dlst_total.where((I_dlst_nv == False)
                                    & (I_dlst_lcc == False)
                                    & (I_dlst_total == False))




dlst_nv_cooling = dlst_nv_clean.where(dlst_nv_clean < 0)
dlst_lcc_nv_cooling = dlst_lcc_clean.where(dlst_nv_clean < 0)
dlst_total_nv_cooling = dlst_total_clean.where(dlst_nv_clean < 0)
dlst_warmed_nv_cooled = dlst_lcc_nv_cooling.where(dlst_lcc_nv_cooling > 0)
lcc_warmed = xr.ufuncs.isfinite(dlst_warmed_nv_cooled).sum()
nv_cooled = xr.ufuncs.isfinite(dlst_nv_cooling).sum()
percent_warmed = (lcc_warmed / nv_cooled) * 100

dlst_total_nv_cooling

dlst_nv_clean.to_netcdf(out_dir + "dlst_nv_test.nc")
dlst_lcc_clean.to_netcdf(out_dir + "dlst_lcc_test.nc")
dlst_total_clean.to_netcdf(out_dir + "dlst_total_test.nc")
dlst_nv_cooling.to_netcdf(out_dir + "dlst_nv_cooling.nc")
tmp.to_netcdf(out_dir + "dlcc_cooling.nc")
