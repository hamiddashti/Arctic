import xarray as xr
import rioxarray
import numpy as np
import modis_functions
from numpy.lib.stride_tricks import as_strided
import matplotlib.pyplot as plt

######################################################################

# ----------- Prameteres and paths -----------------------------------

in_dir = "F:\\MYD21A2\\outputs\\"
out_dir = "F:\\MYD21A2\\outputs\\DeltaLST\\Natural_vs_LULC_LST\\"

nband = 10  # number of LUC classes in the analysis
year1 = 2003
year2 = 2013
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
#                              Functions
# ---------------------------------------------------------------------

def check_finite(x):
    # This fuction checks if there is any finite values in an array
    # nan means that there are changes in the LULC
    import numpy as np

    if np.isfinite(x).any():
        # return nan if there is nan (it has been changed)
        return True
    else:
        # return 1 if there ls no nan which means no change in LULC
        return False


def no_change(xrd, dim):
    # This function uses the check_finite and highlights the pixels where pixels
    # LULC chaged.
    return xr.apply_ufunc(
        check_finite, xrd, input_core_dims=[[dim]], dask="allowed", vectorize=True,
    )


def dist_matrix(x_size, y_size):
    a1 = np.floor(x_size / 2)
    a2 = np.floor(y_size / 2)
    x_arr, y_arr = np.mgrid[0:x_size, 0:y_size]
    cell = (a1, a2)
    dists = np.sqrt((x_arr - cell[0]) ** 2 + (y_arr - cell[1]) ** 2)
    dists[int(a1), int(a2)] = 0
    return dists


# ------------------------------------------------------------------
#                           Preprocessing
# ------------------------------------------------------------------

# open LST and LUC tiles
annual_lst = xr.open_dataarray(in_dir + "LST\\lst_mean_annual.nc")
luc2003 = xr.open_rasterio(
    in_dir
    + "LULC\\PercentCover\\2003_PercentCover_ABoVE_LandCover_Simplified_Bh08v04.tif"
)
luc2003 = luc2003.where(luc2003 != -9999)

luc2013 = xr.open_rasterio(
    in_dir
    + "LULC\\PercentCover\\2013_PercentCover_ABoVE_LandCover_Simplified_Bh08v04.tif"
)
luc2013 = luc2013.where(luc2013 != -9999)

# --------- just to make them align for plotting
annual_lst.rio.set_crs(4326)
luc2003.rio.set_crs(4326)
luc2013.rio.set_crs(4326)
luc2003 = luc2003.rio.reproject_match(annual_lst)
luc2013 = luc2013.rio.reproject_match(annual_lst)

# luc2003 = luc2003.isel(x=np.arange(100,105),y=np.arange(100,105))
# luc2013 = luc2013.isel(x=np.arange(100,105),y=np.arange(100,105))
# annual_lst = annual_lst.isel(x=np.arange(100,105),y=np.arange(100,105))


# Taking the difference in LST and LUC
delta_luc = abs(luc2013 - luc2003)
delta_lst_total = annual_lst.sel(year=year2) - annual_lst.sel(year=year1)

# In the original LUC dataset, when there is no class present the pixels where assigned 0. To avoid confusion
# with pixels that that actually there was a class but it hasn't been changed (luc13-luc03 = 0)
# we set the pixles that are zero in both years (non existance classes) to nan.
tmp = xr.ufuncs.isnan(delta_luc.where((luc2003 == 0) & (luc2013 == 0)))
mask = tmp.where(tmp == True)
luc_masked = delta_luc * mask

# If any of 10 classes has been changed more than 1 percent we call that a changed pixels
# so we don't use them to find the natural variability
changed_pixels = luc_masked.where(luc_masked > 1)
# changed_pixels_all_classes = (xr.ufuncs.isnan(changed_pixels)).astype('int')

# Extracting pixels that have been changed
Changed_pixels_mask = no_change(changed_pixels, "band")
delta_lst_not_changed = delta_lst_total.where(Changed_pixels_mask == False)
delta_lst_changed = delta_lst_total.where(Changed_pixels_mask == True)
delta_luc_not_changed = luc_masked.where(Changed_pixels_mask == False)
delta_luc_changed = luc_masked.where(Changed_pixels_mask == True)

""" -----------------------------------------------------------------------
                 Extracting the natural variability of LST

The method is based on the following paper: 
Alkama, R., Cescatti, A., 2016. Biophysical climate impacts of recent changes
in global forest cover. Science (80-. ). 351, 600 LP – 604.
https://doi.org/10.1126/science.aac8083

* Here we use the concept of numpy stride_trick to create moving windows. 

!!!!! Be very CAREFUL about using strides as also advised by numpy!!!!! 
Best way to check it is to constantly checking the shape of arrays and see if 
they are correct in every step of the work. 
------------------------------------------------------------------------ """

# To run the search window we pad the actual matrix with nan to count for 
# the pixels at the edge

win_size = (
    51  # The size of the search window (e.g. 51*51 pixels or searching within 51 km)
)
win_size_half = int(np.floor(win_size / 2))

# Stridind up the LST and LUC at changed and not changed areas
# --------------------------------------
lst_val_not_changed = delta_lst_not_changed.values
lst_val_not_changed = np.pad(
    lst_val_not_changed,
    (win_size_half, win_size_half),
    "constant",
    constant_values=(np.nan),
)
sub_shape = (win_size, win_size)
view_shape = tuple(np.subtract(lst_val_not_changed.shape, sub_shape) + 1) + sub_shape
lst_val_not_changed_view = as_strided(
    lst_val_not_changed, view_shape, lst_val_not_changed.strides * 2
)
lst_val_not_changed_view = lst_val_not_changed_view.reshape((-1,) + sub_shape)
# --------------------------------------

# --------------------------------------
lst_val_changed = delta_lst_changed.values
lst_val_changed = np.pad(
    lst_val_changed,
    (win_size_half, win_size_half),
    "constant",
    constant_values=(np.nan),
)
sub_shape = (win_size, win_size)
view_shape = tuple(np.subtract(lst_val_changed.shape, sub_shape) + 1) + sub_shape
lst_val_changed_view = as_strided(
    lst_val_changed, view_shape, lst_val_changed.strides * 2
)
lst_val_changed_view = lst_val_changed_view.reshape((-1,) + sub_shape)
# --------------------------------------

# --------------------------------------
luc_val_not_changed = delta_luc_not_changed.values
luc_val_not_changed = np.pad(
    luc_val_not_changed,
    ((0, 0), (win_size_half, win_size_half), (win_size_half, win_size_half)),
    "constant",
    constant_values=np.nan,
)
sub_shape = (10, win_size, win_size)
view_shape = tuple(np.subtract(luc_val_not_changed.shape, sub_shape) + 1) + sub_shape
luc_val_not_changed_view = as_strided(
    luc_val_not_changed, view_shape, luc_val_not_changed.strides * 2
)
luc_val_not_changed_view = luc_val_not_changed_view.reshape((-1,) + sub_shape)
# --------------------------------------

# --------------------------------------
luc_val_changed = delta_luc_changed.values
luc_val_changed = np.pad(
    luc_val_changed,
    ((0, 0), (win_size_half, win_size_half), (win_size_half, win_size_half)),
    "constant",
    constant_values=np.nan,
)
sub_shape = (10, win_size, win_size)
view_shape = tuple(np.subtract(luc_val_changed.shape, sub_shape) + 1) + sub_shape
luc_val_changed_view = as_strided(
    luc_val_changed, view_shape, luc_val_changed.strides * 2
)
luc_val_changed_view = luc_val_changed_view.reshape((-1,) + sub_shape)
# --------------------------------------

# Matrix of distance of each pixel from the central pixel used in inverse 
# distance weighting in the next step
dist_m = dist_matrix(win_size, win_size)


delta_natural_variability = np.empty(len(lst_val_changed_view))
delta_natural_variability[:] = np.nan
for i in np.arange(0, len(lst_val_changed_view) - 1):
    print(i)
    # Each loops goes through each window
    # Read the lst and LUC value of changed and not changed pixels
    lst_changed_tmp = lst_val_changed_view[i]
    luc_changed_tmp = luc_val_changed_view[i]
    luc_not_changed_tmp = luc_val_not_changed_view[i]
    lst_not_changed_tmp = lst_val_not_changed_view[i]
    # If the center pixel of the window is nan (meaning there is no LULC change in that pixel) skip it
    if np.isnan(lst_changed_tmp[win_size_half, win_size_half]):
        continue
       

    # if nan returns False, else returns True: This line tell us what classes exist (True values) in that central pixel
    center_luc = (np.isfinite(luc_changed_tmp[:, win_size_half, win_size_half])).reshape(
        nband, 1, 1
    )

    # This is all pixels where classes havent been changed and surrond the target pixel wirh classes changed
    other_luc = np.isfinite(luc_not_changed_tmp)
    mask = (center_luc == other_luc).all(
        axis=0
    # True if the center center pixel have exact same classes  as the other classes in unchanged surronding areas
    # False otherwise
    )  # This mask is all pixels that have same class as the central pixel

    
    lst_not_changed_tmp_masked = np.where(mask == True, lst_not_changed_tmp, np.nan)
    dist_mask = np.where(mask == True, dist_m, np.nan)
    dist_mask[win_size_half, win_size_half] = np.nan
    weighted_lst = lst_not_changed_tmp_masked / dist_mask

    delta_natural_variability[i] = np.nansum(weighted_lst) / np.nansum(1 / dist_mask)

delta_natural_variability = delta_natural_variability.reshape(
    lst_val_changed.shape[0] - (win_size - 1), lst_val_changed.shape[1] - (win_size - 1)
)


delta_lst_nv = delta_lst_total.copy(data=delta_natural_variability)

# ------------------------------------------------------------
# Calulating the delta LST casusd by changes in LUC
delta_lst_lulc = delta_lst_changed-delta_lst_nv
# ------------------------------------------------------------

# Saving the results
delta_lst_total.to_netcdf(out_dir+"delta_lst_total.nc")
delta_lst_changed.to_netcdf(out_dir+"delta_lst_changed.nc")
delta_lst_not_changed.to_netcdf(out_dir+"delta_lst_not_changed.nc")
delta_luc_changed.to_netcdf(out_dir+"delta_luc_changed.nc")
delta_luc_not_changed.to_netcdf(out_dir+"delta_luc_not_changed.nc")
delta_lst_nv.to_netcdf(out_dir+"delta_lst_nv.nc")
delta_lst_lulc.to_netcdf(out_dir+"delta_lst_lulc.nc")


###################     All Done !    #########################