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


def my_fun(year1, year2, percent_cover, luc_dir, NUMBER_OF_CLASSES, shapes):
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

    conf_normal_final = []
    for i in range(len(shapes)):
        # for i in range(10):
        # for i in range(10570000,10570500):
        print(str(year1) + "-->" + str(i))
        if changed_pixels_mask_val[i] == 0:
            empty = np.empty((10, 10))
            empty[:] = np.nan
            conf_normal_final.append(empty)
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
        conf_normal_final.append(np.round(conf_normal_tmp, 3))

    a = np.array(conf_normal_final)
    b = a.reshape((changed.shape[0], changed.shape[1], 10, 10), order="F")
    da = xr.DataArray(
        data=b,
        dims=["lat", "lon", "year1", "year2"],
        coords={
            "lon": changed.lon,
            "lat": changed.lat,
            "time": pd.Timestamp(str(year1)).year
        },
        # lat=(["lon", "lat"], changed.lat.values),
        # time=pd.Timestamp(str(year1)).year
    )
    da.to_netcdf(out_dir + "confusion_tables_" + str(year1) + ".nc")


luc_dir = (
    "/data/ABOVE/LANDSAT/LANDCOVER/Annual_Landcover_ABoVE_1691/data/mosaic/")
lcc_dir = ("/data/home/hamiddashti/nasa_above/outputs/percent_cover/"
           "percent_cover_albert/")
shape_file = "/data/home/hamiddashti/nasa_above/outputs/grid/"
out_dir = "/data/home/hamiddashti/nasa_above/outputs/historical_change_bio/"

with fiona.open(shape_file, "r") as shapefile:
    shapes = [feature["geometry"] for feature in shapefile]

years = np.arange(1985, 2014)
percent_cover = xr.open_dataarray(lcc_dir + "LULC_10.nc")

for i in range(len(years)):
    year1 = years[i]
    if year1 == 2013:
        continue
    year2 = years[i + 1]
    NUMBER_OF_CLASSES = 10
    my_fun(year1, year2, percent_cover, luc_dir, NUMBER_OF_CLASSES, shapes)
