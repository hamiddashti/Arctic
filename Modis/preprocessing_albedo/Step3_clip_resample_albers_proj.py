import geopandas as gpd
import xarray as xr
import rasterio
import rioxarray
import pandas as pd
import os
from rasterio.mask import mask
from pathlib import Path
import glob
import shutil
import numpy as np
import pycrs
import dask


def getFeatures(shp_file):
    """Function to parse features from GeoDataFrame in such a manner 
    that rasterio wants them"""
    import json
    return [json.loads(shp_file.to_json())["features"][0]["geometry"]]


def clip(tif_file, shp_file, outname):
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


in_dir = ("/data/home/hamiddashti/nasa_above/outputs/albedo_processed/"
          "filtered_reproject/")
out_dir = ("/data/home/hamiddashti/nasa_above/outputs/albedo_processed/")
shp_dir = ("/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/"
           "Study_area/")
geodf = gpd.read_file(shp_dir + "Above_180km_clip.shp")

filenames_albedo = glob.glob(in_dir + "*_albedo.tif")
f = filenames_albedo[0]
date = []
for f in filenames_albedo:
    date.append(
        pd.to_datetime(os.path.basename(f)[30:34] + os.path.basename(f)[35:38],
                       format='%Y%j'))
mydate = pd.DatetimeIndex(date)


def myfun(f_name):
    albedo_file = rasterio.open(f_name)
    albedo_out_name = tmp_dir + "/" + Path(f_name).stem + ".tif"
    clip(albedo_file, tile_shp, albedo_out_name)
    albedo_file.close()


for tileID in np.arange(1, len(geodf) + 1):
    print(" Resampling tile ID:" + str(tileID))
    tmp_dir = out_dir + "tmp/tmp_" + str(tileID)
    os.mkdir(tmp_dir)
    tile_shp = geodf[geodf["OBJECTID"] == tileID]

    lazy_results = []
    for f_name in filenames_albedo:
        mycal = dask.delayed(myfun)(f_name)
        lazy_results.append(mycal)
    results = dask.compute(lazy_results)

    date_xr = xr.Variable("time", mydate)
    cliped_names = []
    for fullname in filenames_albedo:
        basename = os.path.basename(fullname)
        cliped_names.append(tmp_dir + "/" + basename)
    da = xr.concat([xr.open_rasterio(f) for f in cliped_names], dim=date_xr)
    albedo_annual = da.groupby('time.year').mean(dim='time').squeeze()
    albedo_annual.rio.to_raster(out_dir + "tiles_netcdf_annual/tile_" +
                                str(tileID) + ".tif")
