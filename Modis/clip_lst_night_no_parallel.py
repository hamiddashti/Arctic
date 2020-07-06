import xarray as xr
import geopandas as gpd
from shapely.geometry import box, mapping
import numpy as np
import time
import os

t1 = time.time()

# Input directories
in_dir = "/data/ABOVE/MODIS/MYD21A2/"
shp_dir = "/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/Study_area/"
out_dir = "/data/ABOVE/MODIS/MYD21A2/MYD_LST_Night_Tiles/"
chunks = {"lat": 2692, "lon": 8089}


def myclip(tileID):
    import xarray as xr
    import rioxarray

    print(" Cliping LST Night tile ID:" + str(tileID))
    tile_shp = geodf[geodf["OBJECTID"] == tileID]
    da.rio.set_crs(4326)
    da_clip = da.rio.clip(tile_shp.geometry.apply(mapping), tile_shp.crs)
    da_clip.to_netcdf(out_dir + "MYD_LST_Night_Tile_" + str(tileID) + ".nc")


# Read shapefile
geodf = gpd.read_file(shp_dir + "Above_180km_Clip_Geographic_Core.shp")
IDs = np.arange(1, len(geodf) + 1)
# Read the nc file
ds = xr.open_dataset(in_dir + "MYD_LST_Night.nc", chunks=chunks)
# ds = xr.open_dataset('subqc.nc',chunks=chunks)
ds = ds.rename({"lat": "y", "lon": "x"})
da = ds["LST_Night_1KM"]
for tileID in IDs:
    myclip(tileID)
t2 = time.time()
print(t2 - t1)
