import xarray as xr
import rioxarray
import numpy as np

# import modis_functions
from numpy.lib.stride_tricks import as_strided
import matplotlib.pyplot as plt


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

def window_view(data, win_size, type):
	# This is for creating moving windows
    import numpy as np
    from numpy.lib.stride_tricks import as_strided

    win_size = win_size
    win_size_half = int(np.floor(win_size / 2))

    # pad with nan to get correct window for the edges
    if type == "LST":
        data = np.pad(
            data, (win_size_half, win_size_half), "constant", constant_values=(np.nan),
        )
        sub_shape = (win_size, win_size)
        view_shape = tuple(np.subtract(data.shape, sub_shape) + 1) + sub_shape
        data_view = as_strided(data, view_shape, data.strides * 2)

    elif type == "LULC":
        nband = 7  # number of classes
        data = np.pad(
            data,
            ((0, 0), (win_size_half, win_size_half), (win_size_half, win_size_half)),
            "constant",
            constant_values=np.nan,
        )
        sub_shape = (nband, win_size, win_size)
        view_shape = tuple(np.subtract(data.shape, sub_shape) + 1) + sub_shape
        data_view = as_strided(data, view_shape, data.strides * 2)
        # luc_val_not_changed_view = luc_val_not_changed_view.reshape((-1,) + sub_shape)
        data_view = data_view.squeeze()

    return data_view


# -----------------------------------------------------------------
# 						Prameteres and paths 
# -----------------------------------------------------------------
#in_dir = "F:\\working\\LUC\\"
in_dir = '/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/LULC_Maximum_Change_Analyses/'
out_dir = '/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/LULC_Maximum_Change_Analyses/Natural_Variability_outputs/'
#out_dir = "F:\\working\\LUC\\Final_Natural_Variability_Analysis\\"
nband = 7  # number of LUC classes in the analysis
years = range(2004,2015)
win_size = (
		51  # The size of the search window (e.g. 51*51 pixels or searching within 51 km)
	)
win_size_half = int(np.floor(win_size / 2))

	# Matrix of distance of each pixel from the central pixel used in inverse
	# distance weighting in the next step
dist_m = dist_matrix(win_size, win_size)


# ------------------------------------------------------------------
#                           Preprocessing
# Calculate the delta LST and LUC
# ------------------------------------------------------------------

for k in range(0,len(years)-1):
	year1 = years[k] 
	year2 = years[k+1]
	print(year2)
	# open LST and LUC tiles
	annual_lst = xr.open_dataarray(in_dir + "Annual_LST/lst_mean_annual.nc")
	annual_lst = annual_lst.rename({"lat": "y", "lon": "x"})
	annual_lst = annual_lst - 273.15

	luc = xr.open_dataarray(in_dir + "LULC_2003_2014.nc")

	luc_year1 = luc.sel(year=year1)
	luc_year2 = luc.sel(year=year2)

	# luc_year1 = luc_year1.isel(y=range(1495, 1506), x=range(3995, 4006))
	# luc_year2 = luc_year2.isel(y=range(1495, 1506), x=range(3995, 4006))
	# annual_lst = annual_lst.isel(y=range(1495, 1506), x=range(3995, 4006))


	# Taking the difference in LST and LUC
	delta_abs_luc = abs(luc_year2 - luc_year1)
	delta_luc_loss_gain = luc_year2 - luc_year1
	delta_lst_total = annual_lst.sel(year=year2) - annual_lst.sel(year=year1)


	# In the original LUC dataset, when there is no class present the pixels where assigned 0. To avoid confusion
	# with pixels that that actually there was a class but it hasn't been changed (luc13-luc03 = 0)
	# we set the pixles that are zero in both years (non existance classes) to nan.
	tmp = xr.ufuncs.isnan(delta_abs_luc.where((luc_year1 == 0) & (luc_year2 == 0)))
	mask = tmp.where(tmp == True)
	delta_abs_luc = delta_abs_luc * mask
	delta_luc_loss_gain = delta_luc_loss_gain * mask

	# If any of 10 classes has been changed more than 1 percent we call that a changed pixels
	# so we don't use them to find the natural variability
	# True --> not changed; False --> changed
	not_changed_pixels = delta_abs_luc.where(delta_abs_luc > 1)
	# changed_pixels_all_classes = (xr.ufuncs.isnan(changed_pixels)).astype('int')

	# Extracting pixels that have been changed
	not_changed_pixels_mask = no_change(not_changed_pixels, "band")


	delta_lst_not_changed = delta_lst_total.where(not_changed_pixels_mask == False)
	delta_lst_changed = delta_lst_total.where(not_changed_pixels_mask == True)
	delta_abs_luc_not_changed = delta_abs_luc.where(not_changed_pixels_mask == False)
	delta_abs_luc_changed = delta_abs_luc.where(not_changed_pixels_mask == True)
	delta_luc_loss_gain_changed = delta_luc_loss_gain.where(not_changed_pixels_mask == True)
	delta_luc_loss_gain_not_changed = delta_luc_loss_gain.where(
		not_changed_pixels_mask == False
	)

	
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

	# Stridind up the LST and LUC at changed and not changed areas
	# -------------------------------------------------------------
	lst_val_not_changed = delta_lst_not_changed.values
	lst_val_not_changed_view = window_view(lst_val_not_changed, win_size, type="LST")
	lst_val_not_changed_view.shape

	lst_val_changed = delta_lst_changed.values
	lst_val_changed_view = window_view(lst_val_changed, win_size, type="LST")
	lst_val_changed_view.shape

	luc_val_not_changed = delta_abs_luc_not_changed.values
	luc_val_not_changed_view  = window_view(luc_val_not_changed, win_size, type="LULC")
	luc_val_not_changed_view.shape

	luc_val_changed = delta_abs_luc_changed.values
	luc_val_changed_view = window_view(luc_val_changed,win_size,type="LULC")
	luc_val_changed_view.shape


	# Calculate the natural variability
	delta_natural_variability = np.empty(
		(lst_val_changed_view.shape[0], lst_val_changed_view.shape[1])
	)
	delta_natural_variability[:] = np.nan

	counter = 0
	for i in range(0, lst_val_changed_view.shape[0]):
		for j in range(0, lst_val_changed_view.shape[1]):
			
			# Each loops goes through each window
			# Read the lst and LUC value of changed and not changed pixels
			lst_changed_tmp = lst_val_changed_view[i, j]
			luc_changed_tmp = luc_val_changed_view[i, j]
			luc_not_changed_tmp = luc_val_not_changed_view[i, j]
			lst_not_changed_tmp = lst_val_not_changed_view[i, j]

			# If the center pixel of the window is nan (meaning there is no LULC change in that pixel) skip it
			if np.isnan(lst_changed_tmp[win_size_half, win_size_half]):
				continue

			# if nan returns False, else returns True: This line tell us what classes exist (True values) in that central pixel
			win_size_half = int(np.floor(win_size / 2))
			center_luc = (
				np.isfinite(luc_changed_tmp[:, win_size_half, win_size_half])
			).reshape(nband, 1, 1)

			# This is all pixels where classes havent been changed and surrond the target pixel with classes changed
			other_luc = np.isfinite(luc_not_changed_tmp)
			mask = (center_luc == other_luc).all(
				axis=0
				# True if the center center pixel have exact same classes  as the other classes in unchanged surronding areas
				# False otherwise
			)  # This mask is all pixels that have same class as the central pixel

			lst_not_changed_tmp_masked = np.where(
				mask == True, lst_not_changed_tmp, np.nan
			)
			dist_mask = np.where(mask == True, dist_m, np.nan)
			dist_mask[win_size_half, win_size_half] = np.nan
			weighted_lst = lst_not_changed_tmp_masked / dist_mask

			delta_natural_variability[i, j] = np.nansum(weighted_lst) / np.nansum(
				1 / dist_mask
			)

	
	# Converting a numpy array to xarray dataframe
	delta_lst_nv = delta_lst_total.copy(data=delta_natural_variability)

	# ------------------------------------------------------------
	# Calulating the delta LST casusd by changes in LUC
	delta_lst_lulc = delta_lst_changed - delta_lst_nv
	# ------------------------------------------------------------

	# Savinng the results
	not_changed_pixels_mask.to_netcdf(out_dir + "changed_pixels_mask_"+str(year2)+".nc")
	delta_lst_total.to_netcdf(out_dir+ "delta_lst_total_"+str(year2)+".nc")
	delta_lst_changed.to_netcdf(out_dir + "delta_lst_changed_"+str(year2)+".nc")
	delta_lst_not_changed.to_netcdf(out_dir + "delta_lst_not_changed_"+str(year2)+".nc")
	delta_lst_nv.to_netcdf(out_dir + "delta_lst_changed_nv_component_"+str(year2)+".nc")
	delta_lst_lulc.to_netcdf(out_dir + "delta_lst_changed_lulc_component_"+str(year2)+".nc")

	delta_abs_luc_changed.to_netcdf(out_dir + "delta_abs_luc_changed_"+str(year2)+".nc")
	delta_abs_luc_not_changed.to_netcdf(out_dir + "delta_abs_luc_not_changed_"+str(year2)+".nc")
	delta_luc_loss_gain_changed.to_netcdf(out_dir + "delta_luc_loss_gain_changed_"+str(year2)+".nc")
	delta_luc_loss_gain_not_changed.to_netcdf(
		out_dir + "delta_luc_loss_gain_not_changed_"+str(year2)+".nc"
	)

###################     All Done !    #########################
