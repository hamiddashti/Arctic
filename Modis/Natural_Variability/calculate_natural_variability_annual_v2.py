import xarray as xr
import matplotlib.pylab as plt
from matplotlib.pylab import savefig as save
import warnings
warnings.filterwarnings("ignore")


def check_finite(x):
    # This fuction checks if there is any finite values in an array
    # nan means that there are changes in the LULC
    import numpy as np

    if np.isfinite(x).any():
        # return nan if there is nan (it has been changed)
        return True
    else:
        # return 1 if there is no nan which means no change in LULC
        return False


def no_change(xrd, dim):
    # This function uses the check_finite and highlights the pixels where pixels
    # LULC changed.
    return xr.apply_ufunc(
        check_finite,
        xrd,
        input_core_dims=[[dim]],
        dask="allowed",
        vectorize=True,
    )


def dist_matrix(x_size, y_size):
    import numpy as np
    a1 = np.floor(x_size / 2)
    a2 = np.floor(y_size / 2)
    x_arr, y_arr = np.mgrid[0:x_size, 0:y_size]
    cell = (a1, a2)
    dists = np.sqrt((x_arr - cell[0])**2 + (y_arr - cell[1])**2)
    dists[int(a1), int(a2)] = 0
    return dists


def produce_change_mask(luc, years, thresh):
    import xarray as xr
    lc_year1 = luc.sel(year=years[0])
    lc_year2 = luc.sel(year=years[1])

    lc_year1[:, 1600, 4300].values
    lc_year2[:, 1600, 4300].values
    dlcc = lc_year2 - lc_year1
    dlcc[:, 1600, 4300]
    dlcc_abs = abs(dlcc)
    # In the original LUC dataset, when there is no class present the
    # pixels where assigned 0. To avoid confusion with pixels that that
    # actually there was a class but it hasn't been
    # changed (e.g.luc2006-luc2005 = 0). we set the pixle values that are
    # zero in both years (non existance classes) to nan.
    tmp = xr.ufuncs.isnan(dlcc_abs.where((lc_year1 == 0) & (lc_year2 == 0)))
    # To convert tmp from True/False to one/zero
    mask = tmp.where(tmp == True)
    dlcc = dlcc * mask
    dlcc_abs = dlcc_abs * mask
    # If any of 7 classes has been changed more than 1 percent we call
    # that a changed pixels so we don't use them to find the natural variability
    changed_pixels = dlcc_abs.where(dlcc_abs > thresh)
    # Extracting pixels that have been changed
    # True --> changed; False --> not changed
    # changed_pixels_mask = xr.ufuncs.isfinite(changed_pixels).any("band")
    changed_pixels_mask = no_change(changed_pixels, "band")
    return changed_pixels_mask, dlcc, dlcc_abs


def window_view(data, win_size, type):
    # This is for creating moving windows
    import numpy as np
    from numpy.lib.stride_tricks import as_strided
    win_size = win_size
    win_size_half = int(np.floor(win_size / 2))
    # pad with nan to get correct window for the edges
    if type == "OTHER":
        data = np.pad(
            data,
            (win_size_half, win_size_half),
            "constant",
            constant_values=(np.nan),
        )
        sub_shape = (win_size, win_size)
        view_shape = tuple(np.subtract(data.shape, sub_shape) + 1) + sub_shape
        data_view = as_strided(data, view_shape, data.strides * 2)
    elif type == "LCC":
        nband = 7  # number of classes
        data = np.pad(
            data,
            (
                (0, 0),
                (win_size_half, win_size_half),
                (win_size_half, win_size_half),
            ),
            "constant",
            constant_values=np.nan,
        )
        sub_shape = (nband, win_size, win_size)
        view_shape = tuple(np.subtract(data.shape, sub_shape) + 1) + sub_shape
        data_view = as_strided(data, view_shape, data.strides * 2)
        data_view = data_view.squeeze()
    return data_view


def calculate_nv(data, var_name, changed_pixels, years, win_size, dist_m,
                 out_dir, nband):
    import numpy as np
    import xarray as xr
    """ -----------------------------------------------------------------------
                    Extracting the natural variability of LST

    The method is based on the following paper: 
    Alkama, R., Cescatti, A., 2016. Biophysical climate impacts of recent 
    changes in global forest cover. Science (80-. ). 351, 600 LP – 604.
    https://doi.org/10.1126/science.aac8083

    * Here we use the concept of numpy stride_trick to create moving windows. 

    !!!!! Be very CAREFUL about using strides as also advised by numpy!!!!! 
    Best way to check it is to constantly checking the shape of arrays and see
    if they are correct in every step of the work. An alternative is using 
    xarray n-dimensional rolling. But somehow it is slower than the following 
    code. 
    ------------------------------------------------------------------------"""
    changed_pixels = changed_pixels.values
    ddata = data.sel(year=years[1]) - data.sel(year=years[0])

    ddata_changed = ddata.where(changed_pixels == True)
    ddata_not_changed = ddata.where(changed_pixels == False)
    dlcc_abs_changed = dlcc_abs.where(changed_pixels == True)
    dlcc_abs_not_changed = dlcc_abs.where(changed_pixels == False)

    # Stridind up the LST and LUC at changed and not changed areas
    # -------------------------------------------------------------
    ddata_changed_view = window_view(ddata_changed.values,
                                     win_size,
                                     type="OTHER")

    ddata_not_changed_view = window_view(ddata_not_changed.values,
                                         win_size,
                                         type="OTHER")
    dlcc_abs_changed_view = window_view(dlcc_abs_changed.values,
                                        win_size,
                                        type="LCC")
    dlcc_abs_not_changed_view = window_view(dlcc_abs_not_changed.values,
                                            win_size,
                                            type="LCC")

    ddata_natural_variability = np.empty(
        (ddata_changed_view.shape[0], ddata_changed_view.shape[1]))

    for i in range(0, ddata_not_changed_view.shape[0]):
        for j in range(0, ddata_not_changed_view.shape[1]):

            # Each loops goes through each window
            # Read the lst and LUC value of changed and not changed pixels
            ddata_changed_tmp = ddata_changed_view[i, j]
            ddata_not_changed_tmp = ddata_not_changed_view[i, j]
            lc_changed_tmp = dlcc_abs_changed_view[i, j]
            lc_not_changed_tmp = dlcc_abs_not_changed_view[i, j]

            # If the center pixel of the window is nan
            # (meaning there is no LULC change in that pixel) skip it
            win_size_half = int(np.floor(win_size / 2))
            if np.isnan(ddata_changed_tmp[win_size_half, win_size_half]):
                continue

            # if nan returns False, else returns True:
            # This line tell us what classes exist (True values) in that
            # central pixel
            center_luc = (np.isfinite(lc_changed_tmp[:, win_size_half,
                                                     win_size_half])).reshape(
                                                         nband, 1, 1)

            # This is all pixels where classes havent been changed
            # and surrond the target changed pixel
            other_luc = np.isfinite(lc_not_changed_tmp)

            # True if the center center pixel have exact same classes
            # as the other classes in unchanged surronding areas False otherwise
            # This mask is all pixels that have same class as the central pixel
            mask = (center_luc == other_luc).all(axis=0)
            ddata_not_changed_tmp_masked = np.where(mask == True,
                                                    ddata_not_changed_tmp,
                                                    np.nan)

            dist_mask = np.where(mask == True, dist_m, np.nan)
            # Set the center of distance matrix nan
            dist_mask[win_size_half, win_size_half] = np.nan
            weighted_ddata = ddata_not_changed_tmp_masked / dist_mask
            ddata_natural_variability[
                i, j] = np.nansum(weighted_ddata) / np.nansum(1 / dist_mask)

    ddata_nv = data.sel(year=years[1]).copy(data=ddata_natural_variability)
    ddata_lcc = ddata - ddata_nv
    return [ddata_nv, ddata_lcc, ddata_changed, ddata_not_changed]


in_dir = "/data/ABOVE/Final_data/"
out_dir = ("/data/home/hamiddashti/nasa_above/outputs/Natural_Variability/"
           "Natural_Variability_Annual_outputs/EndPoints/all_bands/")

luc = xr.open_dataarray(in_dir + "LUC/LUC_10/LULC_10_2003_2014.nc")
lst_mean = xr.open_dataarray(in_dir +
                             "LST_Final/LST/Annual_Mean/lst_mean_annual.nc")
lst_day = xr.open_dataarray(in_dir +
                            "LST_Final/LST/Annual_Mean/lst_day_annual.nc")
lst_night = xr.open_dataarray(in_dir +
                              "LST_Final/LST/Annual_Mean/lst_night_annual.nc")
albedo = xr.open_dataarray(in_dir +
                           "ALBEDO_Final/Annual_Albedo/Albedo_annual.nc")
et = xr.open_dataarray(in_dir + "ET_Final/Annual_ET/ET_Annual.nc")

# luc = luc.isel(y=range(1400, 1600), x=range(4400, 4600))
# lst_mean = lst_mean.isel(lat=range(1400, 1600), lon=range(4400, 4600))
# lst_day = lst_day.isel(lat=range(1400, 1600), lon=range(4400, 4600))
# lst_night = lst_night.isel(lat=range(1400, 1600), lon=range(4400, 4600))
# albedo = albedo.isel(y=range(1400, 1600), x=range(4400, 4600))
# et = et.isel(y=range(1400, 1600), x=range(4400, 4600))

# Calculating the natural variability for LST, Albedo and ET
years = [2003, 2013]
win_size = 51
dist_m = dist_matrix(win_size, win_size)
thresh = 0.01
# Detecting the changed pixels
changed_pixels, dlcc, dlcc_abs = produce_change_mask(luc=luc,
                                                     years=years,
                                                     thresh=thresh)

changed_pixels.to_netcdf(out_dir + "changed_pixels.nc")
dlcc.to_netcdf(out_dir + "dlcc.nc")
dlcc_abs.to_netcdf(out_dir + "dlcc_abs.nc")

var_names = ["dlst_mean", "dlst_day", "dlst_night", "albedo", "et"]
datasets = [lst_mean, lst_day, lst_night, albedo, et]

for i in range(len(var_names)):
    var_name = var_names[i]
    print(var_name)
    data = datasets[i]
    [nv, lcc, var_changed,
     var_not_changed] = calculate_nv(data=data,
                                     var_name=var_name,
                                     changed_pixels=changed_pixels,
                                     years=years,
                                     win_size=win_size,
                                     dist_m=dist_m,
                                     nband=7,
                                     out_dir=out_dir)

    nv.to_netcdf(out_dir + var_name + "_nv.nc")
    lcc.to_netcdf(out_dir + var_name + "_lcc.nc")
    var_changed.to_netcdf(out_dir + var_name + "_changed.nc")
    var_not_changed.to_netcdf(out_dir + var_name + "_not_changed.nc")

da = xr.open_dataarray(out_dir + "changed_pixels.nc")
da.plot()
plt.savefig(
    "/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/test/changed.png"
)

da.sum()
100 * da.sum() / (da.shape[0] * da.shape[1])
