import xarray as xr
import numpy as np


def reject_outliers(data, m):
    # m is number of std
    import numpy as np

    data = data.astype(float)
    data[abs(data - np.nanmean(data)) > m * np.nanstd(data)] = np.nan
    return data


lst_lulc = xr.open_dataarray("/data/home/hamiddashti/mnt/nasa_above/working/"
                             "modis_analyses/outputs/Natural_Variability/"
                             "Natural_Variability_Annual_outputs/EndPoints/"
                             "delta_lst_changed_lulc_component_2013.nc")

changed = xr.open_dataarray("/data/home/hamiddashti/mnt/nasa_above/working/"
                            "modis_analyses/outputs/Natural_Variability/"
                            "Natural_Variability_Annual_outputs/EndPoints/"
                            "changed_pixels_mask_2013.nc")

albedo = xr.open_dataarray("/data/ABOVE/Final_data/ALBEDO_Final/"
                           "Albedo_annual.nc")
albedo_diff = albedo.isel({"year": 11}) - albedo.isel({"year": 1})
albedo_diff_lulc = albedo_diff.where(changed)

et = xr.open_dataarray(
    '/data/ABOVE/Final_data/ET_Final/Annual_ET/ET_Annual.nc')
et_diff = et.isel({"year": 11}) - et.isel({"year": 1})
et_diff_lulc = et_diff.where(changed)

mean_changed = 100 * changed.sum() / (changed.shape[0] * changed.shape[1])
lst_val = lst_lulc.values.ravel()
lst_outliers = ~np.isnan(reject_outliers(lst_val, 2))
lst = lst_val[lst_outliers]
lst_lulc_max = lst.max()
lst_lulc_min = lst.min()

albedo_val = albedo_diff_lulc.values.ravel()
albedo_outliers = ~np.isnan(reject_outliers(albedo_val, 2))
albedo = albedo_val[albedo_outliers]
albedo_lulc_max = albedo.max()
albedo_lulc_min = albedo.min()

et_val = et_diff_lulc.values.ravel()
et_outliers = ~np.isnan(reject_outliers(et_val, 2))
et = et_val[et_outliers]
et_lulc_max = et.max()
et_lulc_min = et.min()
