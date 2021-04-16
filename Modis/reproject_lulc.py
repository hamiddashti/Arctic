import xarray as xr
import rioxarray
from pyproj import CRS
from rasterio.warp import reproject, Resampling
import numpy as np

in_dir = ("/data/ABOVE/LANDSAT/LANDCOVER/Annual_Landcover_ABoVE_1691/"
          "data/mosaic/")
out_dir = ("/data/ABOVE/LANDSAT/LANDCOVER/Annual_Landcover_ABoVE_1691/"
           "data/mosaic/")

years = list(np.arange(1984, 2015))
for year in years:
    fname = "mosaic_" + str(year) + ".tif"
    print("reprojecting --> " + str(year))
    da = xr.open_rasterio(in_dir + fname)
    da4326 = da.rio.reproject(4326, resampling=Resampling.nearest)
    da4326.rio.to_raster(out_dir + "mosaic_reproject_" + str(year) + ".tif",
                         compress='lzw')
print("Reprojection done!")