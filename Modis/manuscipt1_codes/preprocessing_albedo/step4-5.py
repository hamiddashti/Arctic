# This code for preprocessing the albedo data

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
"""
'''
Step one filter original albedo data based on quality flags and clip 
to the above extent

''''

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
    # clipped.rio.to_raster(out_dir + "filtered_data/" + str(year) + "/cliped_" +
    #                       basename)
    clipped.rio.to_raster(out_dir + "filtered_data/cliped_" + basename)


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


# ---------- Step 3: Concatenate daily files cliped in the last step---------
def concat_tiles(tileID):
    # for tileID in np.arange(1, 176):
    print("tileID:" + str(tileID))
    tmp_dir = out_dir + "step2_filtered_data_cliped/tmp_" + str(tileID)
    fnames = glob.glob(tmp_dir + "/*.tif")
    mydate = []
    for f in fnames:
        mydate.append(
            pd.to_datetime(os.path.basename(f)[22:26] +
                           os.path.basename(f)[27:30],
                           format='%Y%j'))
    date_xr = xr.Variable("time", mydate)
    da = xr.concat([xr.open_rasterio(f) for f in fnames], dim=date_xr)
    da = da.sortby("time").squeeze()
    da.to_netcdf(out_dir + "step3_resampling_tiles/daily_concated_tile_" +
                 str(tileID) + ".nc")

lazy_results = []
for tileID in np.arange(1, 176):
    mycal = dask.delayed(concat_tiles)(tileID)
    lazy_results.append(mycal)
results = dask.compute(lazy_results)
"""


# ----------------------- Step 4 temporal resampled ----------------------------------
def resample(tileID):
    print("tileID:" + str(tileID))
    fname = out_dir + "step3_daily_tiles/da_concat_" + str(tileID) + ".nc"
    da = xr.open_dataarray(fname)
    da_annual = da.groupby('time.year').mean(dim='time')
    da_annual.rio.to_raster(out_dir +
                            "step4_resampling/annual/annual_albedo_tile_" +
                            str(tileID) + ".tif")
    da_seasonal = modis_functions.weighted_season_resmaple(da)
    da_seasonal.rio.to_raster(
        out_dir + "step4_resampling/seasonal/seasonal_albedo_tile_" +
        str(tileID) + ".tif")
    da_monthly = da.resample(time="1MS").mean()
    da_monthly.rio.to_raster(out_dir +
                             "step4_resampling/monthly/monthly_albedo_tile_" +
                             str(tileID) + ".tif")
    # Save a netcdf file as well to get the date and assing it to mosaic in the
    # next step
    if tileID == 1:
        da_annual.to_netcdf(out_dir +
                            "step4_resampling/annual/annual_albedo_tile_" +
                            str(tileID) + ".nc")
        da_seasonal.to_netcdf(
            out_dir + "step4_resampling/seasonal/seasonal_albedo_tile_" +
            str(tileID) + ".nc")
        da_monthly.to_netcdf(out_dir +
                             "step4_resampling/monthly/monthly_albedo_tile_" +
                             str(tileID) + ".nc")


lazy_results = []
for tileID in np.arange(1, 176):
    mycal = dask.delayed(resample)(tileID)
    lazy_results.append(mycal)
results = dask.compute(lazy_results)

# ---------- Step 5: Mosaic and reproject the resampled files ---------------


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


# ------ mosaic resample annual data
fnames = glob.glob(out_dir + "step4_resampling/annual/*.tif")
epsg_code = 102001  #epsg code of the original data
crs = pycrs.parse.from_epsg_code(epsg_code).to_proj4()
mosaicing(out_dir=out_dir + "step5_mosaic_reproject/annual/mosaic/",
          fnames=fnames,
          out_name="albedo_annual_mosaic_orig_proj.tif",
          nodata=np.nan,
          crs=crs)
# ------ Reproject to LST pixel size
lst_ref = xr.open_dataset(
    "/data/ABOVE/Final_data/LST_Final/LST/Annual_Mean/lst_mean_annual.nc",
    decode_coords="all")
lst_ref = lst_ref["__xarray_dataarray_variable__"].isel(year=0)
lst_ref.rio.write_crs(4326, inplace=True)

# original mosaic file
ds = xr.open_rasterio(
    out_dir +
    "step5_mosaic_reproject/annual/mosaic/albedo_annual_mosaic_orig_proj.tif")
# This is one of the tiles in netcdf created in step 4 to just get the time
# dimension and assign it to new mosaic/reprojected data
da_annual = xr.open_dataarray(
    "/data/home/hamiddashti/nasa_above/outputs/albedo_processed/"
    "step4_resampling/annual/annual_albedo_tile_1.nc")

ds = ds.rename({
    "band": "year"
}).assign_coords({
    "year": da_annual.year
}).rename({"year": "time"})
# clip data to the core ddomain extent
domain = gpd.read_file(shp_dir + "CoreDomain.shp")
ds_domain = ds.rio.clip(domain.geometry)
ds_reproj = ds_domain.rio.reproject_match(
    lst_ref, resampling=Resampling.bilinear) * 0.001
ds_reproj.to_netcdf(
    out_dir + "step5_mosaic_reproject/annual/resampled/annual_albedo.nc")

# ---------- Resample and reproject seasonal data
# ------ mosaic resample annual data
fnames = glob.glob(out_dir + "step4_resampling/seasonal/*.tif")
epsg_code = 102001  #epsg code of the original data
crs = pycrs.parse.from_epsg_code(epsg_code).to_proj4()
mosaicing(out_dir=out_dir + "step5_mosaic_reproject/seasonal/mosaic/",
          fnames=fnames,
          out_name="albedo_seasonal_mosaic_orig_proj.tif",
          nodata=np.nan,
          crs=crs)
# ------ Reproject to LST pixel size
# original mosaic file
ds = xr.open_rasterio(
    out_dir +
    "step5_mosaic_reproject/seasonal/mosaic/albedo_seasonal_mosaic_orig_proj.tif"
)
# This is one of the tiles in netcdf created in step 4 to just get the time
# dimension and assign it to new mosaic/reprojected data
da_seasonal = xr.open_dataarray(
    "/data/home/hamiddashti/nasa_above/outputs/albedo_processed/"
    "step4_resampling/seasonal/seasonal_albedo_tile_1.nc")

ds = ds.rename({"band": "time"}).assign_coords({"time": da_seasonal.time})
# clip data to the core ddomain extent

ds_domain = ds.rio.clip(domain.geometry)
ds_reproj = ds_domain.rio.reproject_match(
    lst_ref, resampling=Resampling.bilinear) * 0.001
ds_reproj.to_netcdf(
    out_dir + "step5_mosaic_reproject/seasonal/resampled/seasonal_albedo.nc")

# ---------- Resample and reproject monthly data
# ------ mosaic resample annual data
fnames = glob.glob(out_dir + "step4_resampling/monthly/*.tif")
epsg_code = 102001  #epsg code of the original data
crs = pycrs.parse.from_epsg_code(epsg_code).to_proj4()
mosaicing(out_dir=out_dir + "step5_mosaic_reproject/monthly/mosaic/",
          fnames=fnames,
          out_name="albedo_monthly_mosaic_orig_proj.tif",
          nodata=np.nan,
          crs=crs)
# ------ Reproject to LST pixel size
# original mosaic file
ds = xr.open_rasterio(
    out_dir +
    "step5_mosaic_reproject/monthly/mosaic/albedo_monthly_mosaic_orig_proj.tif"
)
# This is one of the tiles in netcdf created in step 4 to just get the time
# dimension and assign it to new mosaic/reprojected data
da_monthly = xr.open_dataarray(
    "/data/home/hamiddashti/nasa_above/outputs/albedo_processed/"
    "step4_resampling/monthly/monthly_albedo_tile_1.nc")

ds = ds.rename({"band": "time"}).assign_coords({"time": da_monthly.time})
# clip data to the core ddomain extent

ds_domain = ds.rio.clip(domain.geometry)
ds_reproj = ds_domain.rio.reproject_match(
    lst_ref, resampling=Resampling.bilinear) * 0.001
ds_reproj.to_netcdf(
    out_dir + "step5_mosaic_reproject/monthly/resampled/monthly_albedo.nc")

print("Albedo processing done!")
