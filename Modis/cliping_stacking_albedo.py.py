import xarray as xr
import geopandas as gpd
import glob
import rasterio
import os
import pandas as pd
import modis_functions
from pathlib import Path
import shutil
import numpy as np
import rioxarray
from rasterio.enums import Resampling
import matplotlib.pylab as plt
from time import time

t1 = time()
tif_dir = "/data/ABOVE/MODIS/blue_sky_albedo/orders/df99d6e7e26fb666143dbf2de2de4707/Albedo_Boreal_North_America/data_geographic/"
out_dir = "/data/ABOVE/MODIS/blue_sky_albedo/orders/df99d6e7e26fb666143dbf2de2de4707/Albedo_Boreal_North_America/"
shp_dir = "/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/Study_area/"
# tif_dir = 'F:\\MYD21A2\\tmp\\'
# out_dir = 'F:\\MYD21A2\\'
# shp_dir = 'F:\\MYD21A2\\Study_area\\'

date1 = pd.to_datetime("12/1/2002",format='%m/%d/%Y')
date2 = pd.to_datetime("2/28/2015",format='%m/%d/%Y')
date = pd.date_range(date1, date2)

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

geodf = gpd.read_file(shp_dir + "Above_180km_Clip_Geographic.shp")
# geodf.crs
# Assign the crs
# prj = [l.strip() for l in open(shp_dir + "Above_180km_Clip_Geographic.shp", "r")][0]
# geodf.crs = prj


# for i in np.arange(1,len(geodf)+1)
#tileID = 1
for tileID in np.arange(3, 4):
	tmp_dir = out_dir + "tmp" + str(tileID)
	os.mkdir(tmp_dir)
	tile_shp = geodf[geodf["OBJECTID"] == tileID]
	for f in filenames_albedo:
		tif_file = rasterio.open(tif_dir + f)
		out_name = tmp_dir + "/" + Path(f).stem + ".tif"
		modis_functions.tif_clip(tif_file, tile_shp, out_name)
	for f in filenames_qc:
		tif_file = rasterio.open(tif_dir + f)
		out_name = tmp_dir + "/" + Path(f).stem + ".tif"
		modis_functions.tif_clip(tif_file, tile_shp, out_name)

	summer_flag = [0,1,2,4,5,6,16,17,18,20,21,22]
	winter_flag = [0,1,2,3,4,5,6,7,15,16,17,18,19,20,21,22,23]
	date_xr = xr.Variable("time", date)
	a = xr.open_rasterio(tmp_dir + "/" + filenames_albedo[0])
	chunks = {"x": int(a.sizes["x"]), "y": int(a.sizes["x"]), "band": 1}
	da_albedo_init = xr.open_rasterio(tmp_dir + "/" + filenames_albedo[0],chunks=chunks)
	da_qc_init = xr.open_rasterio(tmp_dir + "/" + filenames_qc[0],chunks=chunks)
	if 5<=date[0].month <=9:
		da_albedo_init = da_albedo_init.where(da_qc_init.isin(summer_flag))
	else:
		da_albedo_init = da_albedo_init.where(da_qc_init.isin(winter_flag))
		
	ds_init = da_albedo_init.to_dataset(name="Albedo")
	ds_init = ds_init.assign_coords({"time": date_xr[0]})
	for i in np.arange(1, len(filenames_albedo)):
		print(filenames_albedo[i])
		a = xr.open_rasterio(tmp_dir + "/" + filenames_albedo[i])
		chunks = {"x": int(a.sizes["x"]), "y": int(a.sizes["x"]), "band": 1}
		da_tmp = xr.open_rasterio(tmp_dir + "/" + filenames_albedo[i])
		da_qa_tmp = xr.open_rasterio(tmp_dir + "/" + filenames_qc[i])
		if 5<=date[i].month <=9:
			da_tmp = da_tmp.where(da_qa_tmp.isin(summer_flag))
		else:
			da_tmp = da_tmp.where(da_qa_tmp.isin(winter_flag))
		da_tmp = da_tmp.rio.reproject_match(da_albedo_init, resampling=Resampling.nearest)
		ds_tmp = da_tmp.to_dataset(name="Albedo")
		ds_tmp = ds_tmp.assign_coords({"time": date_xr[i]})
		if i == 1:
			ds_final = xr.concat([ds_init, ds_tmp], dim="time")
		else:
			ds_final = xr.concat([ds_final, ds_tmp], dim="time")
	ds_final = ds_final.where(ds_final != 32767)
	# ds_final.to_netcdf(out_dir + "NetCDF/" + "Albedo_Tile_" + str(tileID) + ".nc")
	ds_final = ds_final.squeeze()
	# print('Reprojectng tile:'+str(tileID))
	# t = ds_final.rio.reproject("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs",resampling=Resampling.bilinear)
	print('########## Saving tile:'+str(tileID))
	ds_final.to_netcdf(out_dir+"NetCDF/Albedo_Tile_" + str(tileID) + ".nc")
	# ds_final.to_netcdf(out_dir +'Tile_' + str(tileID) + ".nc")
	shutil.rmtree(tmp_dir)
	t2 = time()
	print(f'total passed time:{(t2-t1)/3600}hours')

