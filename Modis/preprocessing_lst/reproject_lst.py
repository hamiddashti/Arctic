# This is a file to reproject the lst files from
import xarray as xr
import rioxarray
from pyproj import CRS
import glob
import os
from rasterio.warp import Resampling
from osgeo import gdal

lst = xr.open_dataset(
    "/data/ABOVE/ABoVE_Final_Data/LST/MYD21A2.006_1km_aid0001.nc")

# This file has the ABoVE standard projection ( Canada_Albers_Equal_Area_Conic)
luc_2003 = xr.open_rasterio(
    "/data/ABOVE/LANDSAT/LANDCOVER/Annual_Landcover_ABoVE_1691/data/"
    "mosaic/mosaic_2003.tif")

in_dir = "/data/ABOVE/ABoVE_Final_Data/LST/processed/"
out_dir = "/data/ABOVE/ABoVE_Final_Data/LST/processed/"
fnames = glob.glob(in_dir + "lst*")
cc = CRS.from_cf(lst.crs.attrs)
f = fnames[0]
for f in fnames:
    print(f)
    try:
        da = xr.open_dataarray(f, chunks={"time": 10}, decode_coords="all")
    except:
        da = xr.open_dataarray(f, decode_coords="all")
    basename = os.path.basename(f)
    var_name = os.path.splitext(basename)[0]
    da = da.rename({"ydim": "y", "xdim": "x"})
    da = da.rename(var_name)
    da.rio.write_crs(cc.to_string(), inplace=True)
    da_reproject = da.rio.reproject(luc_2003.rio.crs,
                                    resampling=Resampling.bilinear)
    # try:
    #     da_reproject.attrs.pop("grid_mapping")
    #     da_reproject.to_netcdf(in_dir + "4albers_proj_" + os.path.basename(f))
    # except:
    da_reproject.to_netcdf(in_dir + "albers_proj_" + basename)
print("All Done!")
