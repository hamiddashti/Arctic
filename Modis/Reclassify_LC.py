import xarray as xr
import rioxarray
# import geopandas
from pyproj import CRS
from rasterio.warp import reproject, Resampling
import glob
import os

in_dir = '/data/ABOVE/LANDSAT/LANDCOVER/Annual_Landcover_ABoVE_1691/data/years/mosaic/'
out_dir = '/data/ABOVE/LANDSAT/LANDCOVER/Annual_Landcover_ABoVE_1691/data/years/reclassify/'
fnames = glob.glob(in_dir+'*Mosaic*.tif')
print(len(fnames))
counter = 0 
for f in fnames:
	counter = counter+1
	print(counter)
	da = xr.open_rasterio(f)
	# Convert the CRS from string 
	cc = CRS.from_string(da.crs)

	# Reclassifying the dataset
	da = xr.where(da==6,5,da)
	da = xr.where(da==7,6,da)
	da = xr.where(da==8,6,da)
	da = xr.where(da==9,6,da)
	da = xr.where(da==10,7,da)	

	# Reprojecting to Geographic CRS
	da.rio.write_crs(cc.to_string(), inplace=True)
	#da4326 = da.rio.reproject(4326,resampling=Resampling.nearest)
	#print(f'saving --> {f}')
	basename = os.path.basename(f)   
	da.rio.to_raster(out_dir+'Recalssified_'+basename,compress='lzw')
