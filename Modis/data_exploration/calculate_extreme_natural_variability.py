import xarray as xr
import numpy as np
import matplotlib.pylab as plt
import dask


def no_change(x):
    return (abs(x) < PERCENT_CHANGE).all()


def no_change_mask(xrd, dim):
    # This function uses the check_finite and highlights the pixels where pixels
    # LULC changed.
    return xr.apply_ufunc(
        no_change,
        xrd,
        input_core_dims=[[dim]],
        dask="allowed",
        vectorize=True,
    )


def dist_matrix(x_size, y_size):
    a1 = np.floor(x_size / 2)
    a2 = np.floor(y_size / 2)
    x_arr, y_arr = np.mgrid[0:x_size, 0:y_size]
    cell = (a1, a2)
    dists = np.sqrt((x_arr - cell[0])**2 + (y_arr - cell[1])**2)
    dists[int(a1), int(a2)] = np.nan
    return dists


def calculate_nv(i, extreme_index, lst_diff, luc_diff, weights):
    x_cord = extreme_index[i][0]
    y_cord = extreme_index[i][1]
    lst_tmp = lst_diff_windows[x_cord, y_cord, :, :]
    luc_tmp = luc_diff_windows[:, x_cord, y_cord, :, :]
    mask = np.apply_along_axis(no_change, axis=0, arr=luc_tmp)
    if np.sum(mask) < 30:
        print("not enough pixels")
        lst_nv_tmp = np.nan
        return lst_nv_tmp
    lst_nv_tmp = np.nansum(lst_tmp[mask] / weights[mask]) / np.nansum(
        1 / weights[mask])
    return lst_nv_tmp


in_dir = "/data/ABOVE/Final_data/"
out_dir = "/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/test/"
PERCENT_CHANGE = 5
WIN_SIZE = 51

luc = xr.open_dataarray(in_dir + "LUC/LULC_2003_2014.nc")
luc = luc.rename({"x": "lon", "y": "lat"})
lst = xr.open_dataarray(in_dir +
                        "/LST_Final/LST/Annual_Mean/lst_mean_annual.nc")

lst_diff = lst.loc[2013] - lst.loc[2003]
luc_diff = luc.loc[2013] - luc.loc[2003]

et = xr.open_dataarray(
    in_dir +
    "ET_Final/Annual_ET/ET_Annual.nc")  # Vegetation evapotranspiration
et = et.fillna(0)
et_diff = et.loc[2013] - et.loc[2003]

albedo = xr.open_dataarray(in_dir + "ALBEDO_Final/Albedo_annual.nc")
albedo_diff = albedo.loc[2013] - albedo.loc[2003]

luc_extreme = luc_diff.where((luc_diff > 50) | (luc_diff < -50))

lst_diff_windows = lst_diff.rolling({
    "lat": WIN_SIZE,
    "lon": WIN_SIZE
},
                                    center=True).construct({
                                        "lat": "lat_dim",
                                        "lon": "lon_dim"
                                    }).values
luc_diff_windows = luc_diff.rolling({
    "lat": WIN_SIZE,
    "lon": WIN_SIZE
},
                                    center=True).construct({
                                        "lat": "lat_dim",
                                        "lon": "lon_dim"
                                    }).values

lst_nv_extreme = xr.full_like(luc_diff, np.nan).values
lst_total_extreme = xr.full_like(luc_diff, np.nan).values
lst_lcc_extreme = xr.full_like(luc_diff, np.nan).values
et_extreme = xr.full_like(luc_diff, np.nan).values
albedo_extreme = xr.full_like(luc_diff, np.nan).values

for k in range(0, 7):
    print(k)
    band = luc_extreme.isel(band=k).values
    extreme_index = np.argwhere(~np.isnan(band))
    weights = dist_matrix(WIN_SIZE, WIN_SIZE)
    delayed_results = []
    for i in range(0, len(extreme_index)):

        delayed_results_tmp = dask.delayed(calculate_nv)(i, extreme_index,
                                                         lst_diff, luc_diff,
                                                         weights)
        delayed_results.append(delayed_results_tmp)
    results = dask.compute(*delayed_results)

    rows, cols = zip(*extreme_index)
    lst_nv_extreme[k, rows, cols] = results
    lst_total_extreme[k, rows, cols] = lst_diff.values[rows, cols]
    et_extreme[k, rows, cols] = et_diff.values[rows, cols]
    albedo_extreme[k, rows, cols] = albedo_diff.values[rows, cols]
    lst_lcc_extreme[k, rows,
                    cols] = lst_total_extreme[k, rows,
                                              cols] - lst_nv_extreme[k, rows,
                                                                     cols]

ds = xr.Dataset(
    {
        "LST_NV": (["band", "lat", "lon"], lst_nv_extreme),
        "LST_LCC": (["band", "lat", "lon"], lst_lcc_extreme),
        "LST_TOTAL": (["band", "lat", "lon"], lst_total_extreme),
        "LUC_EXTREME": (["band", "lat", "lon"], luc_extreme),
        "ET": (["band", "lat", "lon"], et_extreme),
        "ALBEDO": (["band", "lat", "lon"], albedo_extreme)
    },
    coords={
        "lat": luc_diff['lat'],
        "lon": luc_diff['lon']
    })
ds.to_netcdf(out_dir+"latest_nc.nc")