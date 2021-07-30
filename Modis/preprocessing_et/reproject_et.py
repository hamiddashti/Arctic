import xarray as xr
import rioxarray
from pyproj import CRS
import pycrs
from rasterio.warp import Resampling
import glob
import os

# This is a reference file (LST)
lst_file = xr.open_dataset(
    "/data/home/hamiddashti/nasa_above/outputs/"
    "lst_processed/albers_proj_lst_day_Annual.nc",
    decode_coords="all")

# -------------------- Annual processing -------------------------------
in_dir = "/data/ABOVE/MODIS/ET/Annual_ET/"
out_dir = "/data/home/hamiddashti/nasa_above/outputs/et_processed/Annual/"
fnames = glob.glob(in_dir + "*nc")
for f in fnames:
    basename = os.path.basename(f)
    var_name = os.path.splitext(basename)[0]
    print(basename)
    da = xr.open_dataarray(f, decode_coords="all")
    da = da.rename(var_name)
    da.rio.write_crs(4326, inplace=True)  # The original crs is geographic
    f_reproj = da.rio.reproject_match(
        lst_file, resampling=Resampling.bilinear)  # match to LST data
    f_reproj.to_netcdf(out_dir + "albers_proj_" + basename)
# -------------------- End of annual processing ------------------------

# -------------------- Seasonal season processing -----------------------

in_dir = "/data/ABOVE/MODIS/ET/Seasonal_ET/"
out_dir = "/data/home/hamiddashti/nasa_above/outputs/et_processed/Seasonal/"
fnames = glob.glob(in_dir + "*nc")
for f in fnames:
    basename = os.path.basename(f)
    var_name = os.path.splitext(basename)[0]
    print(basename)
    da = xr.open_dataarray(f, decode_coords="all")
    da = da.rename(var_name)
    da.rio.write_crs(4326, inplace=True)  # The original crs is geographic
    f_reproj = da.rio.reproject_match(
        lst_file, resampling=Resampling.bilinear)  # match to LST data
    f_reproj.to_netcdf(out_dir + "albers/albers_proj_" + basename)
# -------------------- End of annual processing ------------------------
