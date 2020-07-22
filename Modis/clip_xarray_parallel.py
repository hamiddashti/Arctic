if __name__ == "__main__":
	# For dask applications we need to guard it with __name__ = "__main__"
	import xarray as xr
	import geopandas as gpd
	from shapely.geometry import box, mapping
	import numpy as np
	import time
	import dask
	#from dask.distributed import Client, LocalCluster
	import dask.multiprocessing
	
	t1 = time.time()
	# Input directories 
	in_dir = '/data/ABOVE/MODIS/MYD21A2/'
	shp_dir = "/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/Study_area/"
	out_dir = '/data/ABOVE/MODIS/MYD21A2/MYD_QC_Night_Tiles/'
	
	# Read shapefile
	geodf = gpd.read_file(shp_dir + "Above_180km_Clip_Geographic_Core.shp")
	chunks = {"lat": 2692, "lon": 8089}
	
	# Read the nc file 
	ds = xr.open_dataset(in_dir+'MYD_QC_Night.nc',chunks=chunks)
	#ds = xr.open_dataset('subqc.nc',chunks=chunks)
	ds = ds.rename({"lat": "y", "lon": "x"})
	da = ds["QC_Night"]

	def myclip(tileID):
		import xarray as xr
		import rioxarray
		print(" Cliping tile ID:" + str(tileID))
		tile_shp = geodf[geodf["OBJECTID"] == tileID]
		da.rio.set_crs(4326)
		da_clip = da.rio.clip(tile_shp.geometry.apply(mapping), tile_shp.crs)
		da_clip.to_netcdf(out_dir+'MYD_QC_Night_Tile_'+str(tileID)+'.nc')
	
	dask.config.set(scheduler='processes')
	#IDs = np.arange(1, len(geodf)+1)
	IDs = np.arange(1, 3)
	lazy_results=[]
	for tileID in IDs:
		lazy_result = dask.delayed(myclip)(tileID)
		lazy_results.append(lazy_result)
	
	from dask.diagnostics import ProgressBar
	with ProgressBar():
		futures = dask.persist(*lazy_results)
		results = dask.compute(*futures)
	t2 = time.time()
	print(t2 - t1)