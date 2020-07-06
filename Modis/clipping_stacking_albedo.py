import xarray as xr
import geopandas as gpd
import rasterio
import os
import pandas as pd
import modis_functions
from pathlib import Path
import shutil
import numpy as np
import rioxarray
from rasterio.enums import Resampling
from time import time

t1 = time()

# --------------------------------------------------------------------------
#                          Begin preparation
# ---------------------------------------------------------------------------

tif_dir = "/data/ABOVE/MODIS/blue_sky_albedo/orders/df99d6e7e26fb666143dbf2de2de4707/Albedo_Boreal_North_America/data_geographic/"
out_dir = "/data/ABOVE/MODIS/blue_sky_albedo/orders/df99d6e7e26fb666143dbf2de2de4707/Albedo_Boreal_North_America/"
shp_dir = "/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/Study_area/"
# tif_dir = 'F:\\MYD21A2\\tmp\\'
# out_dir = 'F:\\MYD21A2\\'
# shp_dir = 'F:\\MYD21A2\\Study_area\\'

# Creating the date range feom Dec 2002 to the end of the Feb 2015
# The main reason for inculding the Dec 2002 and JAn-Feb 2015 is we want to cover the winters of 2003 and
#  2014 based on climotoligical seasosns (SON, DFJ, MAM and JJA)

date1 = pd.to_datetime("12/1/2002", format="%m/%d/%Y")
date2 = pd.to_datetime("2/28/2015", format="%m/%d/%Y")
date = pd.date_range(date1, date2)

# Create the daily filenames for albedoa and QC in the geographic CRS
filenames_albedo = []
for t in date:
    year = t.year
    doy = t.dayofyear
    fname = (
        "bluesky_albedo_"
        + str(year)
        + "_"
        + str(doy).zfill(3)
        + "_albedo_Geographic.tif"
    )
    filenames_albedo.append(fname)
filenames_albedo

filenames_qc = []
for t in date:
    year = t.year
    doy = t.dayofyear
    fname = (
        "bluesky_albedo_"
        + str(year)
        + "_"
        + str(doy).zfill(3)
        + "_quality_Geographic.tif"
    )
    filenames_qc.append(fname)
filenames_qc

# Read the shapefile
# Note: if the crs of the shape file is 4326 (Geographic) then we dont need to assign the CRS (commented lines below)
# The geopandas pick it up authomatically
geodf = gpd.read_file(shp_dir + "Above_180km_Clip_Geographic_Core.shp")
# geodf.crs
# Assign the crs
# prj = [l.strip() for l in open(shp_dir + "Above_180km_Clip_Geographic.shp", "r")][0]
# geodf.crs = prj
##################### End of initialization #########################

# --------------------------------------------------------------------------
#                           Begin Clipping
#
#  1: read a tile form the shape file
# 2: Create a tmp folder
# 2: Clip the albedo and QC files using rasterio mask based on the tile and save them in the tmp folder
# 3: Start filtering based on summer (month 5-9) and winter (rest of the year)[1] flags and stack them in
# a netcdf file
# 4:  Replace 32767 with nan and remove extra dimensions in the netcdf file
# 5: Save the netcdf file
# 6: Remove the tmp folder
# Note 1: The main reason for adding a new tmp folder in each iteration and removing it at the end is that
# there might be some bug in xarray catching. It reads the files from previous tmp iteration.... I might
# open and issue in xarray github.
#
#  Reference for the definition of the summer&winter and flags:
# [1] Potter, S, Solvik, K, Erb, A, et al. Climate change decreases the cooling effect from postfire
# albedo in boreal North America. Glob Change Biol. 2020; 26: 1592– 1607. https://doi.org/10.1111/gcb.14888

# ---------------------------------------------------------------------------
print("------------> Start Clipping\n")

for tileID in np.arange(1, len(geodf) + 1):
    print(" Cliping tile ID:" + str(tileID))
    # Create the tmp folder
    tmp_dir = out_dir + "tmp_" + str(tileID)
    os.mkdir(tmp_dir)
    # Get the tile
    tile_shp = geodf[geodf["OBJECTID"] == tileID]
    # Clip based on the rasterio mask and save files in tmp folder
    for f in filenames_albedo:
        tif_file = rasterio.open(tif_dir + f)
        out_name = tmp_dir + "/" + Path(f).stem + ".tif"
        modis_functions.tif_clip(tif_file, tile_shp, out_name)
    for f in filenames_qc:
        tif_file = rasterio.open(tif_dir + f)
        out_name = tmp_dir + "/" + Path(f).stem + ".tif"
        modis_functions.tif_clip(tif_file, tile_shp, out_name)

    # Definition of the flags and criteria for choosing them is presented in:
    # https://daac.ornl.gov/ABOVE/guides/Albedo_Boreal_North_America.html
    # https://doi.org/10.1111/gcb.14888

    summer_flag = [0, 1, 2, 4, 5, 6, 16, 17, 18, 20, 21, 22]
    winter_flag = [0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    date_xr = xr.Variable("time", date)
    a = xr.open_rasterio(tmp_dir + "/" + filenames_albedo[0])
    chunks = {"x": int(a.sizes["x"]), "y": int(a.sizes["x"]), "band": 1}
    da_albedo_init = xr.open_rasterio(
        tmp_dir + "/" + filenames_albedo[0], chunks=chunks
    )
    da_qc_init = xr.open_rasterio(tmp_dir + "/" + filenames_qc[0], chunks=chunks)

    if 5 <= date[0].month <= 9:
        # Summer months
        da_albedo_init = da_albedo_init.where(da_qc_init.isin(summer_flag))
    else:
        # Winter months
        da_albedo_init = da_albedo_init.where(da_qc_init.isin(winter_flag))
    ds_init = da_albedo_init.to_dataset(name="Albedo")
    ds_init = ds_init.assign_coords({"time": date_xr[0]})
    for i in np.arange(1, len(filenames_albedo)):
        a = xr.open_rasterio(tmp_dir + "/" + filenames_albedo[i])
        chunks = {"x": int(a.sizes["x"]), "y": int(a.sizes["x"]), "band": 1}
        da_tmp = xr.open_rasterio(tmp_dir + "/" + filenames_albedo[i])
        da_qa_tmp = xr.open_rasterio(tmp_dir + "/" + filenames_qc[i])
        if 5 <= date[i].month <= 9:
            # Summer month
            da_tmp = da_tmp.where(da_qa_tmp.isin(summer_flag))
        else:
            # Winter month
            da_tmp = da_tmp.where(da_qa_tmp.isin(winter_flag))
        da_tmp = da_tmp.rio.reproject_match(
            da_albedo_init, resampling=Resampling.nearest
        )
        ds_tmp = da_tmp.to_dataset(name="Albedo")
        ds_tmp = ds_tmp.assign_coords({"time": date_xr[i]})
        if i == 1:
            ds_final = xr.concat([ds_init, ds_tmp], dim="time")
        else:
            ds_final = xr.concat([ds_final, ds_tmp], dim="time")
    # Replace 32767 with nan
    ds_final = ds_final.where(ds_final != 32767)
    # Get rid of extra 1D dimesnsions
    ds_final = ds_final.squeeze()
    # print('########## Saving tile:'+str(tileID))
    ds_final.to_netcdf(out_dir + "NetCDF2/Albedo_Tile_" + str(tileID) + ".nc")
    # Remove the tmp folder
    shutil.rmtree(tmp_dir)
    t2 = time()

print("--------------------------------------------")
print("              End of the script             ")
print("--------------------------------------------")
print(f"Total time:{(t2-t1)/3600}hours")

