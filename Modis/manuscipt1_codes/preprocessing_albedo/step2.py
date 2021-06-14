"""
Step two: clip the filtered data based 175 Above tiles.

"""
import geopandas as gpd
import xarray as xr
import numpy as np
import rasterio
import pandas as pd
import os
from rasterio.mask import mask
import pycrs
import dask
import modis_functions
from pathlib import Path
import glob
from rasterio.enums import Resampling
import rioxarray

tif_dir = (
    "/xdisk/davidjpmoore/hamiddashti/data/ALBEDO2/orders/"
    "8d2930e665cfdf9b1f358ba0fc39f38d/Albedo_Boreal_North_America/data/")
out_dir = (
    "/xdisk/davidjpmoore/hamiddashti/nasa_above_outputs/albedo_processed/")

shp_dir = ("/xdisk/davidjpmoore/hamiddashti/data/study_area/")


def getFeatures(shp_file):
    """Function to parse features from GeoDataFrame in such a manner 
    that rasterio wants them"""
    import json
    return [json.loads(shp_file.to_json())["features"][0]["geometry"]]


def clip(tif_file, shp_file, outname):
    """clip based on tiles"""
    coords = getFeatures(shp_file)
    out_img, out_transform = mask(tif_file,
                                  shapes=coords,
                                  crop=True,
                                  nodata=np.nan)
    out_meta = tif_file.meta.copy()
    epsg_code = 102001  # projection system used by ABoVE

    out_meta.update({
        "driver": "GTiff",
        "height": out_img.shape[1],
        "width": out_img.shape[2],
        "transform": out_transform,
        "crs": pycrs.parse.from_epsg_code(epsg_code).to_proj4(),
    })
    # out_file = outname
    with rasterio.open(outname, "w", **out_meta) as dest:
        dest.write(out_img)


geodf = gpd.read_file(shp_dir + "Above_180km_clip.shp")


def clip_fun(f_name):
    albedo_file = rasterio.open(f_name)
    albedo_out_name = tmp_dir + "/" + Path(f_name).stem + ".tif"
    clip(albedo_file, tile_shp, albedo_out_name)
    albedo_file.close()


fnames = glob.glob(out_dir + "step1_filtered_data/" + "cliped*")
for tileID in np.arange(128, len(geodf) + 1):
    print(tileID, flush=True)
    tmp_dir = out_dir + "step2_filtered_data_cliped/" + "tmp_" + str(tileID)
    os.mkdir(tmp_dir)
    tile_shp = geodf[geodf["OBJECTID"] == tileID]
    lazy_results = []
    for f_name in fnames:
        mycal = dask.delayed(clip_fun)(f_name)
        lazy_results.append(mycal)
    results = dask.compute(lazy_results)
print("Cliping is done!", flush=True)
