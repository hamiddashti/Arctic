# This script calculate the confusion table and associated LST, ET and albedo
# and save it in a netcdf file

import numpy as np
import rasterio
import fiona
import pandas as pd
import xarray as xr
from rasterio import features
from rasterio.mask import mask
import dask
from dask.diagnostics import ProgressBar
from xarray.core.duck_array_ops import count


def mymask(tif, shp):
    # To mask landsat LUC pixels included in each MODIS pixel
    out_image, out_transform = rasterio.mask.mask(tif,
                                                  shp,
                                                  all_touched=False,
                                                  crop=True)
    # out_meta = tif.meta
    # return out_image,out_meta,out_transform
    return out_image, out_transform


def confusionmatrix(actual, predicted, unique, imap):
    """
    Generate a confusion matrix for multiple classification
    @params:
        actual      - a list of integers or strings for known classes
        predicted   - a list of integers or strings for predicted classes
        # normalize   - optional boolean for matrix normalization
        unique		- is the unique numbers assigned to each class
        imap		- mapping of classes 

    @return:
        matrix      - a 2-dimensional list of pairwise counts
    """

    matrix = [[0 for _ in unique] for _ in unique]
    # Generate Confusion Matrix
    for p, a in list(zip(actual, predicted)):
        if ((p > len(unique)) or (a > len(unique))):
            continue
        matrix[imap[p]][imap[a]] += 1
    # Matrix Normalization
    # if normalize:
    sigma = sum([sum(matrix[imap[i]]) for i in unique])
    matrix_normalized = [
        row for row in map(lambda i: list(map(lambda j: j / sigma, i)), matrix)
    ]
    return matrix, matrix_normalized


NUMBER_OF_CLASSES = 10  #[DF,DF,shrub,herb,sparse,wetland, water]
class_names = [
    "EF", "DF", "Shrub", "Herb", "Sparse", "Barren", "Fen", "Bog", "SL",
    "water"
]
conversion_type = []
for i in range(0, NUMBER_OF_CLASSES):
    for j in range(0, NUMBER_OF_CLASSES):
        # if (i==j):
        # 	continue
        tmp = class_names[i] + "_" + class_names[j]
        conversion_type.append(tmp)

in_dir = "/data/ABOVE/Final_data/"
out_dir = ("/data/home/hamiddashti/nasa_above/outputs/")

luc2003 = rasterio.open(in_dir + 'LUC/LUC_10/mosaic_reproject_2003.tif')
luc2013 = rasterio.open(in_dir + 'LUC/LUC_10/mosaic_reproject_2013.tif')

changed_pixels_mask = xr.open_dataarray(
    out_dir + "Natural_Variability/"
    "Natural_Variability_Annual_outputs/geographic/changed_pixels.nc")

lst_mean = xr.open_dataarray(
    in_dir + "LST_Final/LST/albers/Annual_Mean/albers_proj_lst_mean_Annual.nc")
lst_day = xr.open_dataarray(in_dir +
                            "LST_Final/LST/Annual_Mean/lst_day_annual.nc")
lst_night = xr.open_dataarray(in_dir +
                              "LST_Final/LST/Annual_Mean/lst_night_annual.nc")
et = xr.open_dataarray(in_dir + "ET_Final/Annual_ET/ET_Annual.nc")
et = et.rename({"x": "lon", "y": "lat"})
et = et.where(lst_mean.loc[2003:2015].notnull())

# albedo = xr.open_dataarray(in_dir +
#                            "ALBEDO_Final/Annual_Albedo/Albedo_annual.nc")
albedo = xr.open_dataarray(
    "/data/home/hamiddashti/nasa_above/outputs/"
    "albedo_processed/step4_resampling/annual/resampled/annual_albedo.nc")
lst_mean_2003 = lst_mean.loc[2003]
lst_mean_2013 = lst_mean.loc[2013]
lst_day_2003 = lst_day.loc[2003]
lst_day_2013 = lst_day.loc[2013]
lst_night_2003 = lst_night.loc[2003]
lst_night_2013 = lst_night.loc[2013]
et_2003 = et.loc[2003]
et_2013 = et.loc[2013]
albedo_2003 = albedo.loc[2003]
albedo_2013 = albedo.loc[2013]
dlst_mean_total = xr.open_dataarray(
    out_dir + "Natural_Variability/"
    "Natural_Variability_Annual_outputs/geographic/02_percent/dlst_mean_changed.nc"
)
dlst_mean_lcc = xr.open_dataarray(
    out_dir + "Natural_Variability/"
    "Natural_Variability_Annual_outputs/geographic/02_percent/dlst_mean_lcc.nc"
)
dlst_mean_nv = xr.open_dataarray(
    out_dir + "Natural_Variability/"
    "Natural_Variability_Annual_outputs/geographic/02_percent/dlst_mean_nv.nc")

dlst_day_total = xr.open_dataarray(
    out_dir + "Natural_Variability/"
    "Natural_Variability_Annual_outputs/geographic/02_percent/dlst_day_changed.nc"
)
dlst_day_lcc = xr.open_dataarray(
    out_dir + "Natural_Variability/"
    "Natural_Variability_Annual_outputs/geographic/02_percent/dlst_day_lcc.nc")
dlst_day_nv = xr.open_dataarray(
    out_dir + "Natural_Variability/"
    "Natural_Variability_Annual_outputs/geographic/02_percent/dlst_day_nv.nc")
dlst_night_total = xr.open_dataarray(
    out_dir + "Natural_Variability/"
    "Natural_Variability_Annual_outputs/geographic/02_percent/dlst_night_changed.nc"
)
dlst_night_lcc = xr.open_dataarray(
    out_dir + "Natural_Variability/"
    "Natural_Variability_Annual_outputs/geographic/02_percent/dlst_night_lcc.nc"
)
dlst_night_nv = xr.open_dataarray(
    out_dir + "Natural_Variability/"
    "Natural_Variability_Annual_outputs/geographic/02_percent/dlst_night_nv.nc"
)
det_total = xr.open_dataarray(
    out_dir + "Natural_Variability/"
    "Natural_Variability_Annual_outputs/geographic/02_percent/et_changed.nc")
det_lcc = xr.open_dataarray(
    out_dir + "Natural_Variability/"
    "Natural_Variability_Annual_outputs/geographic/02_percent/et_lcc.nc")
det_nv = xr.open_dataarray(
    out_dir + "Natural_Variability/"
    "Natural_Variability_Annual_outputs/geographic/02_percent/et_nv.nc")
dalbedo_total = xr.open_dataarray(
    out_dir + "Natural_Variability/"
    "Natural_Variability_Annual_outputs/geographic/02_percent/albedo_changed.nc"
)
dalbedo_lcc = xr.open_dataarray(
    out_dir + "Natural_Variability/"
    "Natural_Variability_Annual_outputs/geographic/02_percent/albedo_lcc.nc")

dalbedo_nv = xr.open_dataarray(
    out_dir + "Natural_Variability/"
    "Natural_Variability_Annual_outputs/geographic/02_percent/albedo_nv.nc")
dlcc = xr.open_dataarray(
    out_dir + "Natural_Variability/"
    "Natural_Variability_Annual_outputs/geographic/02_percent/dlcc.nc")

# Calculate the area weights based on the latitude
weights = np.cos(np.deg2rad(dlst_mean_total.lat))
weights = np.transpose([weights.values] * dlst_mean_total.shape[1])

shape_file = in_dir + "shp_files/ABoVE_1km_Grid_4326_area.shp"
# shape_file = ("/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/"
# "test/test.shp")
print('reading the shapefile')
with fiona.open(shape_file, "r") as shapefile:
    shapes = [feature["geometry"] for feature in shapefile]
    area = [feature["properties"]["area"] for feature in shapefile]

changed_pixels_mask_val = np.ravel(changed_pixels_mask.values, order="F")
lst_mean_2003_val = np.ravel(lst_mean_2003.values, order="F")
lst_mean_2013_val = np.ravel(lst_mean_2013.values, order="F")
lst_day_2003_val = np.ravel(lst_day_2003.values, order="F")
lst_day_2013_val = np.ravel(lst_day_2013.values, order="F")
lst_night_2003_val = np.ravel(lst_night_2003.values, order="F")
lst_night_2013_val = np.ravel(lst_night_2013.values, order="F")
et_2003_val = np.ravel(et_2003.values, order="F")
et_2013_val = np.ravel(et_2013.values, order="F")
albedo_2003_val = np.ravel(albedo_2003.values, order="F")
albedo_2013_val = np.ravel(albedo_2013.values, order="F")
dlst_mean_total_val = np.ravel(dlst_mean_total.values, order="F")
dlst_mean_lcc_val = np.ravel(dlst_mean_lcc.values, order="F")
dlst_mean_nv_val = np.ravel(dlst_mean_nv.values, order="F")
dlst_day_total_val = np.ravel(dlst_day_total.values, order="F")
dlst_day_lcc_val = np.ravel(dlst_day_lcc.values, order="F")
dlst_day_nv_val = np.ravel(dlst_day_nv.values, order="F")
dlst_night_total_val = np.ravel(dlst_night_total.values, order="F")
dlst_night_lcc_val = np.ravel(dlst_night_lcc.values, order="F")
dlst_night_nv_val = np.ravel(dlst_night_nv.values, order="F")
det_total_val = np.ravel(det_total.values, order="F")
det_lcc_val = np.ravel(det_lcc.values, order="F")
det_nv_val = np.ravel(det_nv.values, order="F")
dalbedo_total_val = np.ravel(dalbedo_total.values, order="F")
dalbedo_lcc_val = np.ravel(dalbedo_lcc.values, order="F")
dalbedo_nv_val = np.ravel(dalbedo_nv.values, order="F")
dlcc_val = dlcc.values.reshape(dlcc.shape[0],
                               dlcc.shape[1] * dlcc.shape[2],
                               order="F")
weights_val = np.ravel(weights, order="F")

pix_index = []
final_confusion = []
final_normal_confusion = []
final_lst_mean_2003 = []
final_lst_mean_2013 = []
final_lst_day_2003 = []
final_lst_day_2013 = []
final_lst_night_2003 = []
final_lst_night_2013 = []
final_et_2003 = []
final_et_2013 = []
final_albedo_2003 = []
final_albedo_2013 = []
final_dlst_mean_total = []
final_dlst_mean_lcc = []
final_dlst_mean_nv = []
final_dlst_day_total = []
final_dlst_day_lcc = []
final_dlst_day_nv = []
final_dlst_night_total = []
final_dlst_night_lcc = []
final_dlst_night_nv = []
final_det_total = []
final_det_lcc = []
final_det_nv = []
final_dalbedo_total = []
final_dalbedo_lcc = []
final_dalbedo_nv = []
final_area = []
# final_percent_2003 = []
# final_percent_2013 = []
final_dlcc = []
Err = []
final_weights = []
# unique and imap are input to the confusionmatrix function
unique = np.arange(1, NUMBER_OF_CLASSES + 1)
imap = {key: i for i, key in enumerate(unique)}
# def cal_confus(i):
# for i in range(len(shapes)):

# for i in range(6588237, 6698651):
for i in range(len(shapes)):
    print(i)
    if changed_pixels_mask_val[i] == 0:
        continue
    luc2003_masked = mymask(tif=luc2003, shp=[shapes[i]])[0]
    luc2013_masked = mymask(tif=luc2013, shp=[shapes[i]])[0]
    try:
        conf_tmp, conf_normal_tmp = np.asarray(
            confusionmatrix(luc2003_masked.ravel(), luc2013_masked.ravel(),
                            unique, imap))
    except ZeroDivisionError:
        # This error mostly happens at the border of the study area,
        # where after clipping it with shapefile only left values are
        # 255 and 254 (i.e. nan values)
        print("ZeroDivisionError")
        Err.append(i)
        continue
    # count_2003 = []
    # count_2013 = []
    # for j in np.arange(1, 11):
    #     count_2003_tmp = (luc2003_masked == j).sum()
    #     count_2003.append(count_2003_tmp)
    #     count_2013_tmp = (luc2013_masked == j).sum()
    #     count_2013.append(count_2013_tmp)
    # for k in range(len(count_2013)):
    #     if (count_2003[k] == 0) & (count_2013[k] == 0):
    #         count_2003[k] = np.nan
    #         count_2013[k] = np.nan

    # percent_2003 = count_2003 / (np.nansum(count_2003))
    # percent_2013 = count_2013 / (np.nansum(count_2013))
    # dlcc_val = percent_2013 - percent_2003
    # conf_tmp2 = np.ravel(conf_tmp, order="C")
    # conf_normal_tmp2 = np.ravel(conf_normal_tmp, order="C")
    final_confusion.append(conf_tmp)
    final_normal_confusion.append(conf_normal_tmp)

    pix_index.append(i)
    final_lst_mean_2003.append(lst_mean_2003_val[i])
    final_lst_mean_2013.append(lst_mean_2013_val[i])
    final_lst_day_2003.append(lst_day_2003_val[i])
    final_lst_day_2013.append(lst_day_2013_val[i])
    final_lst_night_2003.append(lst_night_2003_val[i])
    final_lst_night_2013.append(lst_night_2013_val[i])
    final_et_2003.append(et_2003_val[i])
    final_et_2013.append(et_2013_val[i])
    final_albedo_2003.append(albedo_2003_val[i])
    final_albedo_2013.append(albedo_2013_val[i])
    final_dlst_mean_total.append(dlst_mean_total_val[i])
    final_dlst_mean_lcc.append(dlst_mean_lcc_val[i])
    final_dlst_mean_nv.append(dlst_mean_nv_val[i])
    final_dlst_day_total.append(dlst_day_total_val[i])
    final_dlst_day_lcc.append(dlst_day_lcc_val[i])
    final_dlst_day_nv.append(dlst_day_nv_val[i])
    final_dlst_night_total.append(dlst_night_total_val[i])
    final_dlst_night_lcc.append(dlst_night_lcc_val[i])
    final_dlst_night_nv.append(dlst_night_nv_val[i])
    final_det_total.append(det_total_val[i])
    final_det_lcc.append(det_lcc_val[i])
    final_det_nv.append(det_nv_val[i])
    final_dalbedo_total.append(dalbedo_total_val[i])
    final_dalbedo_lcc.append(dalbedo_lcc_val[i])
    final_dalbedo_nv.append(dalbedo_nv_val[i])
    final_area.append(area[i])
    # final_percent_2003.append(percent_2003)
    # final_percent_2013.append(percent_2013)
    final_dlcc.append(dlcc_val[:, i])
    final_weights.append(weights_val[i])

pix_index = np.array(pix_index)
final_confusion = np.array(final_confusion)
final_normal_confusion = np.array(final_normal_confusion)
final_lst_mean_2003 = np.array(final_lst_mean_2003)
final_lst_mean_2013 = np.array(final_lst_mean_2013)
final_lst_day_2003 = np.array(final_lst_day_2003)
final_lst_day_2013 = np.array(final_lst_day_2013)
final_lst_night_2003 = np.array(final_lst_night_2003)
final_lst_night_2013 = np.array(final_lst_night_2013)
final_et_2003 = np.array(final_et_2003)
final_et_2013 = np.array(final_et_2013)
final_albedo_2003 = np.array(final_albedo_2003)
final_albedo_2013 = np.array(final_albedo_2013)
final_dlst_mean_total = np.array(final_dlst_mean_total)
final_dlst_mean_lcc = np.array(final_dlst_mean_lcc)
final_dlst_mean_nv = np.array(final_dlst_mean_nv)
final_dlst_day_total = np.array(final_dlst_day_total)
final_dlst_day_lcc = np.array(final_dlst_day_lcc)
final_dlst_day_nv = np.array(final_dlst_day_nv)
final_dlst_night_total = np.array(final_dlst_night_total)
final_dlst_night_lcc = np.array(final_dlst_night_lcc)
final_dlst_night_nv = np.array(final_dlst_night_nv)
final_det_total = np.array(final_det_total)
final_det_lcc = np.array(final_det_lcc)
final_det_nv = np.array(final_det_nv)
final_dalbedo_total = np.array(final_dalbedo_total)
final_dalbedo_lcc = np.array(final_dalbedo_lcc)
final_dalbedo_nv = np.array(final_dalbedo_nv)
final_area = np.array(final_area)
# final_percent_2003 = np.array(final_percent_2003)
# final_percent_2013 = np.array(final_percent_2013)
final_dlcc = np.array(final_dlcc)
final_weights = np.array(final_weights)

ds = xr.Dataset(
    data_vars={
        "CONFUSION": (("ID", "LC_t1", "LC_t2"), final_confusion),
        "NORMALIZED_CONFUSION":
        (("ID", "LC_t1", "LC_t2"), final_normal_confusion),
        "DLCC": (("ID", "LC"), final_dlcc),
        # "LC_2003": (("ID", "LC"), final_percent_2003),
        # "LC_2013": (("ID", "LC"), final_percent_2013),
        "PIX_INDEX": (("ID"), pix_index),
        "LST_MEAN_2003": (("ID"), final_lst_mean_2003),
        "LST_MEAN_2013": (("ID"), final_lst_mean_2013),
        "LST_DAY_2003": (("ID"), final_lst_day_2003),
        "LST_DAY_2013": (("ID"), final_lst_day_2013),
        "LST_NIGHT_2003": (("ID"), final_lst_night_2003),
        "LST_NIGHT_2013": (("ID"), final_lst_night_2013),
        "ET_2003": (("ID"), final_et_2003),
        "ET_2013": (("ID"), final_et_2013),
        "ALBEDO_2003": (("ID"), final_albedo_2003),
        "ALBEDO_2013": (("ID"), final_albedo_2013),
        "DLST_MEAN_TOTAL": (("ID"), final_dlst_mean_total),
        "DLST_MEAN_LCC": (("ID"), final_dlst_mean_lcc),
        "DLST_MEAN_NV": (("ID"), final_dlst_mean_nv),
        "DLST_DAY_TOTAL": (("ID"), final_dlst_day_total),
        "DLST_DAY_LCC": (("ID"), final_dlst_day_lcc),
        "DLST_DAY_NV": (("ID"), final_dlst_day_nv),
        "DLST_NIGHT_TOTAL": (("ID"), final_dlst_night_total),
        "DLST_NIGHT_LCC": (("ID"), final_dlst_night_lcc),
        "DLST_NIGHT_NV": (("ID"), final_dlst_night_nv),
        "DET_TOTAL": (("ID"), final_det_total),
        "DET_LCC": (("ID"), final_det_lcc),
        "DET_NV": (("ID"), final_det_nv),
        "DALBEDO_TOTAL": (("ID"), final_dalbedo_total),
        "DALBEDO_LCC": (("ID"), final_dalbedo_lcc),
        "DALBEDO_NV": (("ID"), final_dalbedo_nv),
        "Area": (("ID"), final_area),
        "WEIGHTS": (("ID"), final_weights)
    },
    coords={
        "ID": range(len(final_dlst_mean_total)),
        "LC_t1": range(1, 11),
        "LC_t2": range(1, 11),
        "LC": range(1, 11)
    })

ds.to_netcdf(
    out_dir +
    "Sensitivity/EndPoints/Annual/Geographic/02_percent/Confusion_Table_final_02precent_new_albedo.nc"
)

print("All done!")
