# Calculate the historical change in LST, albedo and ET due to LCC
import numpy as np
import rasterio
import fiona
import pandas as pd
import xarray as xr
from rasterio import features
from rasterio.mask import mask
import dask
from dask.diagnostics import ProgressBar


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


def my_fun(year1, year2, percent_cover, luc_dir, NUMBER_OF_CLASSES, lst_sens,
           albedo_sens, et_sens, shapes):
    """Calculate the historical change in biophysical variables due to LCC.
    """
    dlcc = percent_cover.loc[year2] - percent_cover.loc[year1]
    # set nan for land cover that are zero in both years
    dlcc = dlcc.where((percent_cover.loc[year2] != 0)
                      & (percent_cover.loc[year1] != 0))
    dlcc_abs = abs(dlcc)
    changed = (dlcc_abs > 0.02).any(
        dim="band") * 1  #its one if changed else zero
    changed_pixels_mask_val = np.ravel(changed.values, order="F")

    luc1 = rasterio.open(luc_dir + "mosaic_" + str(year1) + ".tif")
    luc2 = rasterio.open(luc_dir + "mosaic_" + str(year2) + ".tif")

    unique = np.arange(1, NUMBER_OF_CLASSES + 1)
    imap = {key: i for i, key in enumerate(unique)}

    dlst_matrix = np.empty((changed.shape[0], changed.shape[1]))
    dlst_matrix[:] = np.nan
    dlst_matrix = np.ravel(dlst_matrix, order="F")

    dalbedo_matrix = np.empty((changed.shape[0], changed.shape[1]))
    dalbedo_matrix[:] = np.nan
    dalbedo_matrix = np.ravel(dalbedo_matrix, order="F")

    det_matrix = np.empty((changed.shape[0], changed.shape[1]))
    det_matrix[:] = np.nan
    det_matrix = np.ravel(det_matrix, order="F")

    print(str(year1))
    for i in range(len(shapes)):
        # for i in range(10):
        # for i in range(10570000,10570500):
        print(str(year1) + "-->" + str(i))
        if changed_pixels_mask_val[i] == 0:
            continue
        luc1_masked = mymask(tif=luc1, shp=[shapes[i]])[0]
        luc2_masked = mymask(tif=luc2, shp=[shapes[i]])[0]
        try:
            conf_tmp, conf_normal_tmp = np.asarray(
                confusionmatrix(luc1_masked.ravel(), luc2_masked.ravel(),
                                unique, imap))
        except ZeroDivisionError:
            # This error mostly happens at the border of the study area,
            # where after clipping it with shapefile only left values are
            # 255 and 254 (i.e. nan values)
            # print("ZeroDivisionError")
            continue
        conf_normal_tmp = np.round(conf_normal_tmp, 3)
        # Set the diagonal of confuxion matix to zero
        np.fill_diagonal(conf_normal_tmp, 0)

        #skip a pixel if it include any transitions to/from bog, shallow/littoral
        # and water
        if np.count_nonzero(conf_normal_tmp[7:10, 7:10]) > 0:
            continue
        final_conf_normal = conf_normal_tmp[0:7, 0:7]
        dlst_matrix[i] = np.sum(final_conf_normal * lst_sens)
        dalbedo_matrix[i] = np.sum(final_conf_normal * albedo_sens)
        det_matrix[i] = np.sum(final_conf_normal * et_sens)

    dlst2 = dlst_matrix.reshape((changed.shape[0], changed.shape[1]),
                                order="F")
    final_dlst = changed.copy(data=dlst2)
    final_dlst.to_netcdf(out_dir + "dlst_hist_" + str(year1) + ".nc")

    dalbedo2 = dalbedo_matrix.reshape((changed.shape[0], changed.shape[1]),
                                      order="F")
    final_dalbedo = changed.copy(data=dalbedo2)
    final_dalbedo.to_netcdf(out_dir + "dalbedo_hist_" + str(year1) + ".nc")

    det2 = det_matrix.reshape((changed.shape[0], changed.shape[1]), order="F")
    final_det = changed.copy(data=det2)
    final_det.to_netcdf(out_dir + "det_hist_" + str(year1) + ".nc")


luc_dir = (
    "/data/ABOVE/LANDSAT/LANDCOVER/Annual_Landcover_ABoVE_1691/data/mosaic/")
lcc_dir = ("/data/home/hamiddashti/nasa_above/outputs/percent_cover/"
           "percent_cover_albert/")
shape_file = "/data/home/hamiddashti/nasa_above/outputs/grid/"
out_dir = "/data/home/hamiddashti/nasa_above/outputs/historical_change_bio/"

percent_cover = xr.open_dataarray(lcc_dir + "LULC_10.nc")
lst_sens = np.array([[0, 1.025, -1.016, -0.97, 0.397, 1.179, -0.919],
                     [-1.025, 0, -0.932, -0.519, 3.703, 5.566, 1.91],
                     [1.016, 0.932, 0, -0.171, 2.249, 4.378, 1.009],
                     [0.97, 0.519, 0.171, 0, 1.075, 3.186, 1.169],
                     [-0.397, -3.703, -2.249, -1.075, 0, 1.644, -2.902],
                     [-1.179, -5.566, -4.378, -3.186, -1.644, 0, -5.977],
                     [0.919, -1.91, -1.009, -1.169, 2.902, 5.977, 0]])
albedo_sens = np.array([[0, 0.018, 0.025, 0.04, -0.009, 0.064, 0.021],
                        [-0.018, 0, 0.003, 0.012, -0.04, -0.01, 0.009],
                        [-0.025, -0.003, 0, 0.005, -0.058, -0.017, -0.036],
                        [-0.04, -0.012, -0.005, 0, -0.012, -0.003, -0.019],
                        [0.009, 0.04, 0.058, 0.012, 0, -0.001, 0.082],
                        [-0.064, 0.01, 0.017, 0.003, 0.001, 0, 0.039],
                        [-0.021, -0.009, 0.036, 0.019, -0.082, -0.039, 0]])

et_sens = np.array(
    [[0, 81.26, -45.365, -64.02, -32.675, -86.608, -63.797],
     [-81.26, 0, -56.712, -39.329, -185.195, -175.382, -137.577],
     [45.365, 56.712, 0, -10.682, -38.298, -38.909, 10.666],
     [64.02, 39.329, 10.682, 0, -16.56, -4.562, 71.51],
     [32.675, 185.195, 38.298, 16.56, 0, -8.892, 51.73],
     [86.608, 175.382, 38.909, 4.562, 8.892, 0, 79.719],
     [63.797, 137.577, -10.666, -71.51, -51.73, -79.719, 0]])

with fiona.open(shape_file, "r") as shapefile:
    shapes = [feature["geometry"] for feature in shapefile]

years = np.arange(1985, 2014)
for i in range(len(years)):
    year1 = years[i]
    if year1 == 2013:
        continue
    year2 = years[i + 1]
    NUMBER_OF_CLASSES = 10
    var_sens = lst_sens
    my_fun(year1, year2, percent_cover, luc_dir, NUMBER_OF_CLASSES, lst_sens,
           albedo_sens, et_sens, shapes)

lst_names = []
albedo_names = []
et_names = []
for year in years[:-1]:
    lst_names.append(out_dir + "dlst_hist_" + str(year) + ".nc")
    albedo_names.append(out_dir + "dalbedo_hist_" + str(year) + ".nc")
    et_names.append(out_dir + "det_hist_" + str(year) + ".nc")
years = pd.date_range(start="1985", end="2013", freq="Y").year
da_lst = xr.concat([xr.open_dataarray(f) for f in lst_names], dim=years)
da_albedo = xr.concat([xr.open_dataarray(f) for f in albedo_names], dim=years)
da_et = xr.concat([xr.open_dataarray(f) for f in et_names], dim=years)
da_lst = da_lst.rename({"concat_dim":"time"})
da_albedo = da_albedo.rename({"concat_dim":"time"})
da_et = da_et.rename({"concat_dim":"time"})
da_lst.to_netcdf(out_dir + "final/dlst_hist.nc")
da_albedo.to_netcdf(out_dir + "final/dalbedo_hist.nc")
da_et.to_netcdf(out_dir + "final/det_hist.nc")
