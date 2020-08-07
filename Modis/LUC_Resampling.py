# Script to calculate the percent cover of the classes 
import glob
import os
import modis_functions
import rioxarray
import xarray as xr

in_dir =  '/data/ABOVE/LANDSAT/LANDCOVER/Annual_Landcover_ABoVE_1691/data/years/reclassify/'
out_dir = '/data/ABOVE/LANDSAT/LANDCOVER/Annual_Landcover_ABoVE_1691/data/years/reclassify/'
shp_file = '/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/Study_area/Above_180km_Clip_Geographic_Core.shp'
# Reprojecting to geographic CRS (since all other files are in this projection)


fnames1 = glob.glob(in_dir+'Recalssified*.tif')
for f1 in  fnames1:
	print(f1)
	da=xr.open_rasterio(f1)
	da_reproj=da.rio.reproject(4326)
	da_reproj.rio.to_raster(out_dir+'geo_'+os.path.basename(f1))


# This is just a 1000*1000 m raster (modis LST; could be anything else e.g. albedo, ET... or even a fishnet)
ref_raster = out_dir+'ref_raster.tif'

fnames2 = glob.glob(in_dir+'geo*Recalssified*.tif')
for f2 in fnames2:
	print(f2)
	out_raster = out_dir+'percent_cover_'+os.path.basename(f2)
	modis_functions.percent_cover(fine_raster=f2,coarse_raster=ref_raster,out_raster=out_raster,n=7)
	
	# Clipping the produced percent cover maps based on the ABoVE core domain	
	out_file = out_dir+'percent_cover_clipped_'+os.path.basename(f2)
	modis_functions.tif_clip(out_raster, shp_file, out_file)




