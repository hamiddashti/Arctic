import geopandas as gpd
import xarray as xr
import numpy as np
import rasterio
import pandas as pd
import os
from rasterio.mask import mask
import pycrs
import dask
from pathlib import Path
import modis_functions
import glob
from rasterio.enums import Resampling
import rioxarray
import shutil


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


tif_dir = ("/data/ABOVE/MODIS/ALBEDO2/orders/8d2930e665cfdf9b1f358ba0fc39f38d/"
           "Albedo_Boreal_North_America/data/")
out_dir = ("/data/home/hamiddashti/nasa_above/outputs/albedo_processed/")

shp_dir = ("/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/"
           "Study_area/")
# tmp_out = "/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/test/"

# year = 2013
# fnames_albedo = []
# for f in fnames_albedo_all:
#     tmp = int(os.path.basename(f)[15:19])
#     if tmp == year:
#         fnames_albedo.append(f)
# fnames_quality = []
# for f in fnames_albedo:
#     b = f.rsplit("albedo", 1)
#     fnames_quality.append("quality".join(b))

#-----Step one filter original albedo data based on quality flags -----------
geodf = gpd.read_file(shp_dir + "Above_180km_clip.shp")
summer_flag = [0, 1, 2, 4, 5, 6, 16, 17, 18, 20, 21, 22]
winter_flag = [
    0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35,
    36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55
]

fnames_albedo = glob.glob(tif_dir + "*_albedo.tif")
fnames_quality = []
for f in fnames_albedo:
    b = f.rsplit("albedo", 1)
    fnames_quality.append("quality".join(b))

def filtering_fun(i):
    # for i in range(len(fname_albedo_2003)):
    basename = os.path.basename(fnames_albedo[i])
    print(basename)
    albedo_tmp = xr.open_rasterio(fnames_albedo[i])
    quality_tmp = xr.open_rasterio(fnames_quality[i])
    tmp_date = pd.to_datetime(os.path.basename(fnames_albedo[i])[15:19] +
                              os.path.basename(fnames_albedo[i])[20:23],
                              format='%Y%j')
    if 5 <= tmp_date.month <= 9:
        # Summer months
        da_albedo_clean = albedo_tmp.where(quality_tmp.isin(summer_flag))
    else:
        # Winter months
        da_albedo_clean = albedo_tmp.where(quality_tmp.isin(winter_flag))

    clipped = da_albedo_clean.rio.clip(geodf.geometry)
    clipped.rio.to_raster(out_dir + "filtered_data/" + str(year) + "/cliped_" +
                          basename)


lazy_results = []
for i in range(len(fnames_albedo)):
    mycal = dask.delayed(filtering_fun)(i)
    lazy_results.append(mycal)
results = dask.compute(lazy_results)

# -------- Step two: clip the filtered data based 175 Above tiles -------------
# we doing this beacuse for some reasons we failed to run dask

geodf = gpd.read_file(shp_dir + "Above_180km_clip.shp")


def clip_fun(f_name):
    albedo_file = rasterio.open(f_name)
    albedo_out_name = tmp_dir + "/" + Path(f_name).stem + ".tif"
    clip(albedo_file, tile_shp, albedo_out_name)
    albedo_file.close()


fnames = glob.glob(out_dir + "filtered_data/" + str(year) + "/cliped*")
for tileID in np.arange(1, len(geodf) + 1):
    print(tileID)
    tmp_dir = out_dir + "filtered_data_cliped/" + str(year) + "/tmp_" + str(
        tileID)
    os.mkdir(tmp_dir)
    tile_shp = geodf[geodf["OBJECTID"] == tileID]
    lazy_results = []
    for f_name in fnames:
        mycal = dask.delayed(clip_fun)(f_name)
        lazy_results.append(mycal)
    results = dask.compute(lazy_results)


# ---------- Step 3: resample the cliped file into annual and seasonal---------
def resampling_fun(tileID):
    # for tileID in np.arange(1, len(geodf) + 1):
    for year in [2003, 2013]:
        print(str(year) + " --> tileID:" + str(tileID))
        tmp_dir = out_dir + "filtered_data_cliped/" + str(
            year) + "/tmp_" + str(tileID)
        fnames = glob.glob(tmp_dir + "/*.tif")
        mydate = []
        for f in fnames:
            mydate.append(
                pd.to_datetime(os.path.basename(f)[22:26] +
                               os.path.basename(f)[27:30],
                               format='%Y%j'))
        date_xr = xr.Variable("time", mydate)
        da = xr.concat([xr.open_rasterio(f) for f in fnames], dim=date_xr)
        da.to_netcdf(out_dir+"filtered_data_cliped/")
        # da_annual = da.groupby('time.year').mean(dim='time').squeeze()
        # da_annual.rio.to_raster(out_dir + "annual_tiles/" + str(year) +
        #                         "/annual_albedo_tile_" + str(tileID) + ".tif")
        da_seasonal = modis_functions.weighted_season_resmaple(da)
        da_seasonal.rio.to_raster(out_dir + "seasonal_tiles/" + str(year) +
                                  "/seasonal_albedo_tile_" + str(tileID) +
                                  ".tif")


lazy_results = []
for tileID in np.arange(1, len(geodf) + 1):
    mycal = dask.delayed(resampling_fun)(tileID)
    lazy_results.append(mycal)
results = dask.compute(lazy_results)

# ---------- Step 4: Mosaic and reproject the resampled files ---------------


def mosaicing(out_dir, fnames, out_name, nodata, crs):
    from rasterio.enums import Resampling
    from rasterio.merge import merge
    import rasterio
    import numpy as np

    src_files_to_mosaic = []
    for f in fnames:
        src = rasterio.open(f)
        src_files_to_mosaic.append(src)
    mosaic, out_trans = merge(
        datasets=src_files_to_mosaic,
        #   resampling=Resampling.bilinear,
        method="first",
        nodata=nodata)
    out_meta = src.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "crs": crs,
    })
    with rasterio.open(out_dir + out_name, "w", **out_meta) as dest:
        dest.write(mosaic)


fnames = glob.glob(out_dir + "annual_tiles/" + str(year) + "/*.tif")
epsg_code = 102001
crs = pycrs.parse.from_epsg_code(epsg_code).to_proj4()
mosaicing(out_dir=out_dir + "annual_mosaic_resampled/",
          fnames=fnames,
          out_name="albedo_annual_mosaic_" + str(year) + ".tif",
          nodata=np.nan,
          crs=crs)

lst_ref = xr.open_dataset(
    "/data/home/hamiddashti/nasa_above/outputs/"
    "lst_processed/albers_proj_lst_mean_Annual.nc",
    decode_coords="all")
lst_ref = lst_ref["lst_mean_Annual"].isel(year=0)
ds = xr.open_rasterio(out_dir +
                      "annual_mosaic_resampled/albedo_annual_mosaic_" +
                      str(year) + ".tif")
# time = pd.DatetimeIndex(pd.date_range("2002","2014",freq="AS")).year
time = pd.DatetimeIndex(pd.date_range(str(year), str(year), freq="AS")).year
ds = ds.assign_coords({"band": time}).rename({"band": "year"})
domain = gpd.read_file(shp_dir + "CoreDomain.shp")
ds_domain = ds.rio.clip(domain.geometry)
ds_reproj = ds_domain.rio.reproject_match(
    lst_ref, resampling=Resampling.bilinear) * 0.001
ds_reproj.to_netcdf(out_dir + "annual_mosaic_resampled/annual_albedo_" +
                    str(year) + ".nc")

# Step 5: concatenate the 2003 and 2013

da_2003 = xr.open_dataarray(out_dir +
                            "annual_mosaic_resampled/annual_albedo_2003.nc")
da_2013 = xr.open_dataarray(out_dir +
                            "annual_mosaic_resampled/annual_albedo_2013.nc")
ds = xr.concat([da_2003, da_2013], dim="year")
ds.to_netcdf(out_dir + "final_albedo/annual/final_annual_albedo.nc")
