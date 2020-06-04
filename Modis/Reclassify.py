import xarray as xr
import rioxarray
import geopandas
from pyproj import CRS
from rasterio.warp import reproject, Resampling

in_dir = 'F:\\MYD21A2\\outputs\\LULC\\'
out_dir = 'F:\\MYD21A2\\outputs\\LULC\\'

da = xr.open_rasterio(in_dir+'ABoVE_LandCover_Simplified_Bh08v04.tif')
# Convert the CRS from string 
cc = CRS.from_string(da.crs)

# Reclassifying the dataset
da = xr.where(da==6,5,da)
da = xr.where(da==7,6,da)
da = xr.where(da==8,6,da)
da = xr.where(da==9,6,da)
da = xr.where(da==10,6,da)

# Reprojecting to Geographic CRS
da.rio.write_crs(cc.to_string(), inplace=True)
da4326 = da.rio.reproject(4326,resampling=Resampling.nearest)

da4326.rio.to_raster(out_dir+'ABoVE_LandCover_Simplified_Bh08v04_reclassify.tif')
