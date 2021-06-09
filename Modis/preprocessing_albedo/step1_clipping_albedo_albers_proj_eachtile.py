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
    # int(tif_file.crs.data["init"][5:])
    out_meta.update({
        "driver": "GTiff",
        "height": out_img.shape[1],
        "width": out_img.shape[2],
        "transform": out_transform,
        "crs": pycrs.parse.from_epsg_code(epsg_code).to_proj4(),
        # "crs":tif_file.crs
    })
    # out_file = outname
    with rasterio.open(outname, "w", **out_meta) as dest:
        dest.write(out_img)


# --------------------------------------------------------------------------
#                          Begin preparation
# ---------------------------------------------------------------------------

tif_dir = ("/data/ABOVE/MODIS/ALBEDO2/orders/8d2930e665cfdf9b1f358ba0fc39f38d/"
           "Albedo_Boreal_North_America/data/")
out_dir = ("/data/ABOVE/MODIS/ALBEDO2/orders/8d2930e665cfdf9b1f358ba0fc39f38d/"
           "Albedo_Boreal_North_America/data/Tiles/")
# out_dir = (
#     "/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/test/")
shp_dir = ("/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/"
           "Study_area/")
# tif_dir = 'F:\\MYD21A2\\tmp\\'
# out_dir = 'F:\\MYD21A2\\'
# shp_dir = 'F:\\MYD21A2\\Study_area\\'

geodf = gpd.read_file(shp_dir + "Above_180km_clip.shp")

filenames_albedo = glob.glob(tif_dir + "*_albedo.tif")
# filenames_albedo = filenames_albedo[0:20]
filenames_quality = []
for f in filenames_albedo:
    b = f.rsplit("albedo", 1)
    filenames_quality.append("quality".join(b))

# filenames_quality = glob.glob(in_dir + "*_quality.tif")
date = []
for f in filenames_albedo:
    date.append(
        pd.to_datetime(os.path.basename(f)[15:19] + os.path.basename(f)[20:23],
                       format='%Y%j'))
time = pd.DatetimeIndex(date)
summer_flag = [0, 1, 2, 4, 5, 6, 16, 17, 18, 20, 21, 22]
winter_flag = [
    0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35,
    36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55
]

# 1: clip the albedo and quality data and save them in a temporary folder
# 2:
tileID=75
for tileID in np.arange(1, len(geodf) + 1):
    # for tileID in np.arange(1, 2):
    print(" Cliping tile ID:" + str(tileID))
    tmp_dir = out_dir + "tmp/testtmp_" + str(tileID)
    os.mkdir(tmp_dir)
    tile_shp = geodf[geodf["OBJECTID"] == tileID]

    for albedo_name, quality_name in zip(filenames_albedo, filenames_quality):
        albedo_file = rasterio.open(albedo_name)
        albedo_out_name = tmp_dir + "/" + Path(albedo_name).stem + ".tif"
        clip(albedo_file, tile_shp, albedo_out_name)
        quality_file = rasterio.open(quality_name)
        quality_out_name = tmp_dir + "/" + Path(quality_name).stem + ".tif"
        clip(quality_file, tile_shp, quality_out_name)

    date_xr = xr.Variable("time", time)
    a = xr.open_rasterio(tmp_dir + "/" + os.path.basename(filenames_albedo[0]))
    chunks = {"x": int(a.sizes["x"]), "y": int(a.sizes["x"]), "band": 1}
    da_albedo_init = xr.open_rasterio(tmp_dir + "/" +
                                      os.path.basename(filenames_albedo[0]),
                                      chunks=chunks)
    da_qc_init = xr.open_rasterio(tmp_dir + "/" +
                                  os.path.basename(filenames_quality[0]),
                                  chunks=chunks)

    # Initiate the xarray dataset
    if 5 <= time[0].month <= 9:
        # Summer months
        da_albedo_init = da_albedo_init.where(da_qc_init.isin(summer_flag))
    else:
        # Winter months
        da_albedo_init = da_albedo_init.where(da_qc_init.isin(winter_flag))

    ds_init = da_albedo_init.to_dataset(name="Albedo")
    ds_init = ds_init.assign_coords({"time": date_xr[0]})
    for i in np.arange(1, len(filenames_albedo)):
        a = xr.open_rasterio(tmp_dir + "/" +
                             os.path.basename(filenames_albedo[i]))
        chunks = {"x": int(a.sizes["x"]), "y": int(a.sizes["x"]), "band": 1}
        da_tmp = xr.open_rasterio(tmp_dir + "/" +
                                  os.path.basename(filenames_albedo[i]))
        da_qa_tmp = xr.open_rasterio(tmp_dir + "/" +
                                     os.path.basename(filenames_quality[i]))
        if 5 <= date[i].month <= 9:
            # Summer month
            da_tmp = da_tmp.where(da_qa_tmp.isin(summer_flag))
        else:
            # Winter month
            da_tmp = da_tmp.where(da_qa_tmp.isin(winter_flag))
        # da_tmp = da_tmp.rio.reproject_match(da_albedo_init,
        #                                     resampling=Resampling.nearest)
        ds_tmp = da_tmp.to_dataset(name="Albedo")
        ds_tmp = ds_tmp.assign_coords({"time": date_xr[i]})
        if i == 1:
            ds_final = xr.concat([ds_init, ds_tmp], dim="time")
        else:
            ds_final = xr.concat([ds_final, ds_tmp], dim="time")
    # Replace 32767 with nan
    # ds_final = ds_final.where(ds_final != 32767)
    # Get rid of extra 1D dimesnsions
    ds_final = ds_final.squeeze()
    ds_final = ds_final.sortby("time")
    albedo_annual = ds_final.groupby("time.year").mean(dim="time")
    albedo_growing = modis_functions.growing_season(ds_final)
    albedo_month_resample = ds_final.resample(time="1MS").mean()
    albedo_month_group = ds_final.groupby("time.month").mean()
    albedo_season_resample = modis_functions.weighted_season_resmaple(ds_final)
    albedo_season_group = modis_functions.weighted_season_group(ds_final)

    # print('########## Saving tile:'+str(tileID))
    # albedo_annual.to_netcdf(out_dir + "Netcdf/albedo_annual_tile_" + str(tileID) + ".nc")
    albedo_annual["Albedo"].rio.to_raster(out_dir +
                                          "Netcdf/albedo_annual_tile_" +
                                          str(tileID) + ".tif")
    # print("annual saved")
    # albedo_growing["Albedo"].rio.to_raster(out_dir +
    #                                        "Netcdf/albedo_growing_tile_" +
    #                                        str(tileID) + ".tif")
    # print("growing saved")
    # albedo_month_resample["Albedo"].rio.to_raster(
    #     out_dir + "Netcdf/albedo_month_resample_tile_" + str(tileID) + ".tif")
    # print("month resmaple saved")
    # albedo_month_group["Albedo"].rio.to_raster(
    #     out_dir + "Netcdf/albedo_month_group_tile_" + str(tileID) + ".tif")
    # print("month group saved")
    # albedo_season_resample["Albedo"].rio.to_raster(
    #     out_dir + "Netcdf/albedo_season_resample_tile_" + str(tileID) + ".tif")
    # print("season resamle saved")
    # albedo_season_group["Albedo"].rio.to_raster(
    #     out_dir + "Netcdf/albedo_season_group_tile_" + str(tileID) + ".tif")
    # print("season group saved")
    # ds_final.to_netcdf(out_dir + "Netcdf/ds_all_tile_" + str(tileID) + ".nc")
    # print("all saved")
    # Remove the tmp folder
    # from rasterio.merge import merge
    # ds = glob.glob(out_dir + "Netcdf/*.tif")
    # crs = ds_final.rio.crs
    # mosaicing(out_dir=out_dir + "Netcdf/",fnames=ds,out_name="mosic_test4.tif",nodata=np.nan,crs=crs)
    shutil.rmtree(tmp_dir, ignore_errors=True)

import time

t1 = time.time()
myfun(tileID=20)
t2 = time.time()
print(f"time spend:{t2-t1}")

lazy_results = []
for tileID in np.arange(1, len(geodf) + 1):
    mycal = dask.delayed(myfun)(tileID)
    lazy_results.append(mycal)
results = dask.compute(lazy_results)

print("--------------------------------------------")
print("              End of the script             ")
print("--------------------------------------------")
"""
# Creating the date range feom Dec 2002 to the end of the Feb 2015
# The main reason for inculding the Dec 2002 and JAn-Feb 2015 is we want to
# cover the winters of 2003 and 2014 based on climotoligical
# seasosns (SON, DFJ, MAM and JJA)
date1 = pd.to_datetime("12/1/2002", format="%m/%d/%Y")
date2 = pd.to_datetime("02/28/2014", format="%m/%d/%Y")
date = pd.date_range(date1, date2)

# Create the daily filenames for albedoa and QC in the geographic CRS
filenames_albedo = []
filenames_qc = []
for t in date:
    year = t.year
    doy = t.dayofyear
    fname_albedo = ("bluesky_albedo_" + str(year) + "_" + str(doy).zfill(3) +
                    "_albedo.tif")
    fname_qc = ("bluesky_albedo_" + str(year) + "_" + str(doy).zfill(3) +
                "_quality.tif")
    filenames_albedo.append(fname_albedo)
    filenames_qc.append(fname_qc)


# Using dask to parallelize
def myfun(i):
    albedo_file = rasterio.open(tif_dir + filenames_albedo[i])
    quality_file = rasterio.open(tif_dir + filenames_qc[i])
    outname_albedo = out_dir + "clip_" + filenames_albedo[i]
    outname_quality = out_dir + "clip_" + filenames_qc[i]
    clip(albedo_file, shp_file=shp_file, outname=outname_albedo)
    clip(quality_file, shp_file=shp_file, outname=outname_quality)


delayed_results = []
for i in range(0, len(filenames_albedo)):
    mycals = dask.delayed(myfun)(i)
    delayed_results.append(mycals)
print("\ncliping is on going\n")
results = dask.compute(*delayed_results)
# # Read the shapefile
# # Note: if the crs of the shape file is 4326 (Geographic) then we dont need to assign the CRS (commented lines below)
# # The geopandas pick it up authomatically
# geodf = gpd.read_file(shp_dir + "Above_180km_clip.shp")
# # geodf.crs
# # Assign the crs
# # prj = [l.strip() for l in open(shp_dir + "Above_180km_Clip_Geographic.shp", "r")][0]
# # geodf.crs = prj
# ##################### End of initialization #########################

# # --------------------------------------------------------------------------
# #                           Begin Clipping
# #
# # 1: read a tile form the shape file
# # 2: Create a tmp folder
# # 2: Clip the albedo and QC files using rasterio mask based on the tile and save them in the tmp folder
# # 3: Start filtering based on summer (month 5-9) and winter (rest of the year)[1] flags and stack them in
# # a netcdf file
# # 4:  Replace 32767 with nan and remove extra dimensions in the netcdf file
# # 5: Save the netcdf file
# # 6: Remove the tmp folder
# # Note 1: The main reason for adding a new tmp folder in each iteration and removing it at the end is that
# # there might be some bug in xarray catching. It reads the files from previous tmp iteration.... I might
# # open and issue in xarray github.
# #
# #  Reference for the definition of the summer&winter and flags:
# # [1] Potter, S, Solvik, K, Erb, A, et al. Climate change decreases the cooling effect from postfire
# # albedo in boreal North America. Glob Change Biol. 2020; 26: 1592– 1607. https://doi.org/10.1111/gcb.14888

# # ---------------------------------------------------------------------------
# print("------------> Start Clipping\n")
# tileID = 131
# for tileID in np.arange(1, len(geodf) + 1):
#     print(" Cliping tile ID:" + str(tileID))
#     # Create the tmp folder
#     tmp_dir = out_dir + "tmp_" + str(tileID)
#     os.mkdir(tmp_dir)
#     # Get the tile
#     tile_shp = geodf[geodf["OBJECTID"] == tileID]
#     # Clip based on the rasterio mask and save files in tmp folder
#     for f in filenames_albedo:
#         tif_file = rasterio.open(tif_dir + f)
#         outname = tmp_dir + "/" + Path(f).stem + ".tif"
#         modis_functions.tif_clip(tif_file, tile_shp, outname)
#     for f in filenames_qc:
#         tif_file = rasterio.open(tif_dir + f)
#         out_name = tmp_dir + "/" + Path(f).stem + ".tif"
#         modis_functions.tif_clip(tif_file, tile_shp, out_name)

#     # Definition of the flags and criteria for choosing them is presented in:
#     # https://daac.ornl.gov/ABOVE/guides/Albedo_Boreal_North_America.html
#     # https://doi.org/10.1111/gcb.14888

#     summer_flag = [0, 1, 2, 4, 5, 6, 16, 17, 18, 20, 21, 22]
#     winter_flag = [0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 18, 19, 20, 21, 22, 23]
#     date_xr = xr.Variable("time", date)
#     a = xr.open_rasterio(tmp_dir + "/" + filenames_albedo[0])
#     chunks = {"x": int(a.sizes["x"]), "y": int(a.sizes["x"]), "band": 1}
#     da_albedo_init = xr.open_rasterio(tmp_dir + "/" + filenames_albedo[0],
#                                       chunks=chunks)
#     da_qc_init = xr.open_rasterio(tmp_dir + "/" + filenames_qc[0],
#                                   chunks=chunks)

#     if 5 <= date[0].month <= 9:
#         # Summer months
#         da_albedo_init = da_albedo_init.where(da_qc_init.isin(summer_flag))
#     else:
#         # Winter months
#         da_albedo_init = da_albedo_init.where(da_qc_init.isin(winter_flag))
#     ds_init = da_albedo_init.to_dataset(name="Albedo")
#     ds_init = ds_init.assign_coords({"time": date_xr[0]})
#     for i in np.arange(1, len(filenames_albedo)):
#         a = xr.open_rasterio(tmp_dir + "/" + filenames_albedo[i])
#         chunks = {"x": int(a.sizes["x"]), "y": int(a.sizes["x"]), "band": 1}
#         da_tmp = xr.open_rasterio(tmp_dir + "/" + filenames_albedo[i])
#         da_qa_tmp = xr.open_rasterio(tmp_dir + "/" + filenames_qc[i])
#         if 5 <= date[i].month <= 9:
#             # Summer month
#             da_tmp = da_tmp.where(da_qa_tmp.isin(summer_flag))
#         else:
#             # Winter month
#             da_tmp = da_tmp.where(da_qa_tmp.isin(winter_flag))
#         da_tmp = da_tmp.rio.reproject_match(da_albedo_init,
#                                             resampling=Resampling.nearest)
#         ds_tmp = da_tmp.to_dataset(name="Albedo")
#         ds_tmp = ds_tmp.assign_coords({"time": date_xr[i]})
#         if i == 1:
#             ds_final = xr.concat([ds_init, ds_tmp], dim="time")
#         else:
#             ds_final = xr.concat([ds_final, ds_tmp], dim="time")
#     # Replace 32767 with nan
#     ds_final = ds_final.where(ds_final != 32767)
#     # Get rid of extra 1D dimesnsions
#     ds_final = ds_final.squeeze()
#     # print('########## Saving tile:'+str(tileID))
#     ds_final.to_netcdf(out_dir + "NetCDF2/Albedo_Tile_" + str(tileID) + ".nc")
#     # Remove the tmp folder
#     shutil.rmtree(tmp_dir)
#     t2 = time()

print("--------------------------------------------")
print("              End of the script             ")
print("--------------------------------------------")
"""