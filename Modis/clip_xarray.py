import xarray as xr
import rioxarray
import geopandas
from shapely.geometry import box, mapping

geodf = geopandas.read_file("clip.shp")
# ndvi = xr.open_dataarray("NDVI_processed.nc")
lst = xr.open_dataarray("LST_processed.nc")


# lst = rioxarray.open_rasterio("LST_processed.nc")["__xarray_dataarray_variable__"]

# print("\ncliping NDVI")
# ndvi_clipped = ndvi.rio.clip(geodf.geometry.apply(mapping), geodf.crs)
# ndvi_clipped.to_netcdf("ndvi_clip.nc")

print("\ncliping LST")
lst_clipped = lst.rio.clip(geodf.geometry.apply(mapping), geodf.crs)
lst_clipped.to_netcdf("lst_clip.nc")


print("\n ############## Done! #####################")
