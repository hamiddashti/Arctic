import xarray as xr
import rioxarray
from pyproj import CRS
from rasterio.warp import reproject, Resampling
import numpy as np
from pyproj import CRS

in_dir ="/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/" 

ds = xr.open_dataset(
    "/data/ABOVE/MODIS/LST_ORIGINAL_PROJ/MYD11A2.006_1km_aid0001.nc"
)

lc = xr.open_rasterio(
    "/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/ABoVE_LandCover_Simplified_Bh01v02.tif"
)



ds.LST_Day_1km.attrs
a = ds.crs.attrs
cc = CRS.from_cf(a)
cc.to_string()
xds = ds.rename_dims({"xdim": "x", "ydim": "y"})

xds.coords["x"] = xds.x
xds.coords["y"] = xds.y
xds.coords["time"] = xds.time

xds.rio.write_crs(cc.to_string(), inplace=True)

ds.rio.write_crs(ds.crs.attrs)
da = xds["LST_Day_1km"].isel(time=0)
da = da.reset_coords(["xdim","ydim"],drop=True)
xds_reproj2 = da.rio.reproject(4326)
xds_reproj2.to_netcdf(in_dir+"test5.nc")
