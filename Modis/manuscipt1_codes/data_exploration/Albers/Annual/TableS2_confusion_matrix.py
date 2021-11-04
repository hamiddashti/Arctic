# This script calculate the confusion table and associated LST, ET and albedo
# and save it in a netcdf file

import numpy as np
from numpy.core.fromnumeric import squeeze
import rasterio
import fiona
import pandas as pd
import xarray as xr
from rasterio import features
from rasterio.mask import mask
from sklearn.metrics import confusion_matrix
import dask.array as da
# from dask.distributed import Client, LocalCluster
# client = Client()


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


in_dir = "/data/ABOVE/LANDSAT/LANDCOVER/Annual_Landcover_ABoVE_1691/data/mosaic/"
out_dir = ("/data/home/hamiddashti/nasa_above/outputs/data_analyses/Annual/"
           "Albers/Figures_MS1/")
chunks = {"x": 10000, "y": 10000}
# luc2003 = xr.open_rasterio(in_dir + 'LUC/LUC_10/mosaic_reproject_2003.tif',
#                            chunks=chunks)
# luc2013 = xr.open_rasterio(in_dir + 'LUC/LUC_10/mosaic_reproject_2013.tif',
#                            chunks=chunks)
NUMBER_OF_CLASSES = 10  #[DF,DF,shrub,herb,sparse,wetland, water]
class_names = [
    "EF", "DF", "Shrub", "Herb", "Sparse", "Barren", "Fen", "Bog",
    "Shallow/Litter", "water"
]
unique = np.arange(1, NUMBER_OF_CLASSES + 1)
imap = {key: i for i, key in enumerate(unique)}

shape_file = "/data/ABOVE/Final_data/shp_files/CoreDomain.shp"
with fiona.open(shape_file, "r") as shapefile:
    shapes = [feature["geometry"] for feature in shapefile]

years = np.array([2003, 2013])
for i in range(len(years)):
    year1 = years[i]
    print(year1)
    if year1 == 2014:
        continue
    year2 = years[i + 1]
    luc1 = rasterio.open(in_dir + "mosaic_" + str(year1) + ".tif")
    luc2 = rasterio.open(in_dir + "mosaic_" + str(year2) + ".tif")
    # luc1 = xr.open_rasterio(in_dir + "mosaic_" + str(year1) + ".tif", chunks=chunks)
    # luc2 = xr.open_rasterio(in_dir + "mosaic_" + str(year2) + ".tif", chunks=chunks)
    luc1_masked = mymask(tif=luc1, shp=[shapes[0]])[0]
    luc2_masked = mymask(tif=luc2, shp=[shapes[0]])[0]
    # conf_tmp, conf_normal_tmp = np.asarray(
    # f1=da.from_array(luc1_masked,chunks=chunks)
    # confusionmatrix(luc1.ravel(), luc2.ravel(), unique,
    #                 imap))

    conf_tmp = confusion_matrix(luc1_masked.ravel(),
                                luc2_masked.ravel(),
                                labels=unique)

    conf_tmp_normalized = confusion_matrix(luc1_masked.ravel(),
                                           luc2_masked.ravel(),
                                           labels=unique,
                                           normalize="true")

    df_not_normalized = pd.DataFrame(data=conf_tmp,
                                     index=class_names,
                                     columns=class_names)
    df_not_normalized.to_csv(out_dir + "confusion_table_not_normalized_" +
                             str(year1) + ".csv")

    df_normalized = pd.DataFrame(data=np.round(conf_tmp_normalized, 3),
                                 index=class_names,
                                 columns=class_names)
    df_normalized.to_csv(out_dir + "confusion_table_normalized_" + str(year1) +
                         ".csv")

# Combining the normalized and not normalized confusion tables
conf_combined_list = []
for i in range(10):
    for j in range(10):
        conf_combined_list.append(
            str("{:.1e}".format(conf_tmp[i, j])) + "(" +
            str(np.round(conf_tmp_normalized[i, j], 3)) + ")")

conf_combined = np.array(conf_combined_list).reshape(10, 10)
df_combined = pd.DataFrame(data=conf_combined,
                           index=class_names,
                           columns=class_names)
df_combined.to_csv(out_dir + "confusion_table_combined_" + str(year1) + ".csv")
