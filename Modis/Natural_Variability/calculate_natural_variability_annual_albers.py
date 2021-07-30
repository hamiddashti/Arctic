import xarray as xr
import numpy as np
import dask
from dask.diagnostics import ProgressBar
import rioxarray
import geopandas as gpd


def outliers_index(data, m=3):
    """
    Returns true if a value is outlier
    
    :param int data: numpy array
    :param int m: # of std to include data 
    """
    import numpy as np
    d = np.abs(data - np.nanmedian(data))
    mdev = np.nanmedian(d)
    s = d / mdev if mdev else 0.
    return ~(s < m)


# Directoiries where annual ET, albedo and LST are located
in_dir = ("/data/home/hamiddashti/nasa_above/outputs/percent_cover/"
          "percent_cover_albert/")
out_dir = ("/data/home/hamiddashti/nasa_above/outputs/Natural_Variability/"
           "Natural_Variability_Annual_outputs/Albers/")
lst_dir = "/data/home/hamiddashti/nasa_above/outputs/lst_processed/"
et_dir = "/data/home/hamiddashti/nasa_above/outputs/et_processed/Annual/"

WINSIZE = 101  # Window size

# The land cover data
LC = xr.open_dataarray(
    in_dir + "LULC_10_2003_2014.nc",
    decode_coords="all",
)
LST = xr.open_dataset(
    ("/data/home/hamiddashti/nasa_above/outputs/lst_processed/"
     "albers_proj_lst_mean_Annual.nc"),
    decode_coords="all",
)
LST = LST.rename({"x": "lon", "y": "lat"})
LST = LST["lst_mean_Annual"]

ET = xr.open_dataset(
    et_dir + "albers_proj_ET_Annual.nc",
    decode_coords="all",
)
ET = ET.rename({"x": "lon", "y": "lat"})
ET = ET["ET_Annual"]

Albedo = xr.open_dataarray(
    "/data/home/hamiddashti/nasa_above/outputs/"
    "albedo_processed/final_albedo/annual/final_annual_albedo.nc")
Albedo = Albedo.rename({"x": "lon", "y": "lat"})

# This is beacuse the coordinates are not exactly the same due to rounding issu
LC = LC.assign_coords({"lat": LST.lat, "lon": LST.lon})
ET = ET.assign_coords({"lat": LST.lat, "lon": LST.lon})
Albedo = Albedo.assign_coords({"lat": LST.lat, "lon": LST.lon})

# Take the difference between year 2003 and 2013
dlcc = LC.loc[2013] - LC.loc[2003]
dlst = LST.loc[2013] - LST.loc[2003]
det = ET.loc[2013] - ET.loc[2003]
dalbedo = Albedo.loc[2013] - Albedo.loc[2003]

# set nan for land cover that are zero in both years
dlcc = dlcc.where((LC.loc[2013] != 0) & (LC.loc[2003] != 0))
dlcc_abs = abs(dlcc)
changed = (dlcc_abs > 0.02).any(dim="band") * 1  #its one if changed else zero


changed_roll = changed.rolling({
    "lat": WINSIZE,
    "lon": WINSIZE
}, center=True).construct({
    "lat": "lat_dim",
    "lon": "lon_dim"
})

dlcc_abs_roll = dlcc_abs.rolling({
    "lat": WINSIZE,
    "lon": WINSIZE
}, center=True).construct({
    "lat": "lat_dim",
    "lon": "lon_dim"
})

dlst_roll = dlst.rolling({
    "lat": WINSIZE,
    "lon": WINSIZE
}, center=True).construct({
    "lat": "lat_dim",
    "lon": "lon_dim"
})

det_roll = det.rolling({
    "lat": WINSIZE,
    "lon": WINSIZE
}, center=True).construct({
    "lat": "lat_dim",
    "lon": "lon_dim"
})

dalbedo_roll = dalbedo.rolling({
    "lat": WINSIZE,
    "lon": WINSIZE
}, center=True).construct({
    "lat": "lat_dim",
    "lon": "lon_dim"
})

# -----------------------------------------------------------------------
#       Writing a loop to go over each windows and perform analyses
# -----------------------------------------------------------------------
# create an empty dataset to save the calculated NV
dlst_nv_no_outlier = xr.full_like(changed, fill_value=np.nan, dtype=float)
det_nv_no_outlier = xr.full_like(changed, fill_value=np.nan, dtype=float)
dalbedo_nv_no_outlier = xr.full_like(changed, fill_value=np.nan, dtype=float)

half_win = int((WINSIZE - 1) / 2)  # half window size
counter = 0

# Looping over windows
for i in range(0, dlst_nv_no_outlier.shape[0]):
    for j in range(0, dlst_nv_no_outlier.shape[1]):
        counter += 1
        print(counter)
        changed_tmp = changed_roll.isel(lat=i, lon=j)
        dlst_tmp = dlst_roll.isel(lat=i, lon=j)
        det_tmp = det_roll.isel(lat=i, lon=j)
        dalbedo_tmp = dalbedo_roll.isel(lat=i, lon=j)
        if (changed_tmp[half_win, half_win]
                == 0) | (dlst_tmp[half_win, half_win].isnull()
                         ):  # if the pixels has not been changed skip the loop
            continue

        dlcc_tmp = dlcc_abs_roll.isel(lat=i, lon=j)

        # Create a mask of pixels that have not changed (changed==0)
        dlcc_all_mask = np.isfinite(
            dlcc_tmp.where(changed_tmp == 0)).any(dim="band")
        # Masking surronding variables that are not changed
        dlst_not_changed_all = dlst_tmp.where(dlcc_all_mask)
        det_not_changed_all = det_tmp.where(dlcc_all_mask)
        dalbedo_not_changed_all = dalbedo_tmp.where(dlcc_all_mask)

        # Exclude the value of central pixel (the changed pixel) and take mean
        dlst_not_changed_all[half_win, half_win] = np.nan
        det_not_changed_all[half_win, half_win] = np.nan
        dalbedo_not_changed_all[half_win, half_win] = np.nan
        # Removing outliers in data
        I_outliers_lst = outliers_index(dlst_not_changed_all, 3)
        dlst_not_changed_clean = dlst_not_changed_all.where(
            I_outliers_lst == False)
        I_outliers_et = outliers_index(det_not_changed_all, 3)
        det_not_changed_clean = det_not_changed_all.where(
            I_outliers_et == False)
        I_outliers_albedo = outliers_index(dalbedo_not_changed_all, 3)
        dalbedo_not_changed_clean = dalbedo_not_changed_all.where(
            I_outliers_albedo == False)

        dlst_nv_no_outlier[i, j] = dlst_not_changed_clean.mean()
        det_nv_no_outlier[i, j] = det_not_changed_clean.mean()
        dalbedo_nv_no_outlier[i, j] = dalbedo_not_changed_clean.mean()

dlst_lcc = dlst - dlst_nv_no_outlier
# det = det.assign_coords({"lat": LST.lat, "lon": LST.lon})
det_nv_no_outlier = det_nv_no_outlier.assign_coords({
    "lat": LST.lat,
    "lon": LST.lon
})
# dalbedo = dalbedo.assign_coords({"lat": LST.lat, "lon": LST.lon})
# dalbedo_nv_no_outlier = dalbedo_nv_no_outlier.assign_coords({
#     "lat": LST.lat,
#     "lon": LST.lon
# })
det_lcc = det - det_nv_no_outlier
dalbedo_lcc = dalbedo - dalbedo_nv_no_outlier

dlcc.to_netcdf(out_dir + "dlcc.nc")
changed.to_netcdf(out_dir + "changed.nc")

dlst.to_netcdf(out_dir + "dlst_total.nc")
dlst_nv_no_outlier.to_netcdf(out_dir + "dlst_nv.nc")
dlst_lcc.to_netcdf(out_dir + "dlst_lcc.nc")

det.to_netcdf(out_dir + "det_total.nc")
det_nv_no_outlier.to_netcdf(out_dir + "det_nv.nc")
det_lcc.to_netcdf(out_dir + "det_lcc.nc")

dalbedo.to_netcdf(out_dir + "dalbedo_total.nc")
dalbedo_nv_no_outlier.to_netcdf(out_dir + "dalbedo_nv.nc")
dalbedo_lcc.to_netcdf(out_dir + "dalbedo_lcc.nc")
"""
# ----------------------------------------------------------------------
#           Doing the same thing but using xarray and dask
# ----------------------------------------------------------------------
dask.config.set({"array.slicing.split_large_chunks": True})
dlcc_all_mask = np.isfinite(dlcc_abs_roll.where(changed == 0)).any(dim="band")
dlst_not_changed_all = dlst_roll.where(dlcc_all_mask)
dlst_not_changed_all.load()
dlst_not_changed_all[:, :, 50, 50] = np.nan
I_outliers = xr.apply_ufunc(outliers_index,
                            dlst_not_changed_all,
                            input_core_dims=[['lat_dim', 'lon_dim']],
                            output_core_dims=[["lat_dim", "lon_dim"]],
                            vectorize=True)
dlst_not_changed_all_clean = dlst_not_changed_all.where(I_outliers == False)
dlst_nv_all_no_outlier = dlst_not_changed_all_clean.mean(
    dim=["lat_dim", "lon_dim"])
dlst_nv_all_no_outlier.to_netcdf(out_dir + "dlst_nv_all_no_outlier.nc")
"""