from threading import TIMEOUT_MAX
import xarray as xr
import rioxarray
import rasterio
from rasterio.enums import Resampling
import pycrs
import numpy as np 
import pandas as pd

def custom_merge(old_data, new_data, old_nodata, new_nodata, index=None, roff=None, coff=None):
    old_data[:] = np.nanmax(old_data, new_data) 

def mosaicing(out_dir, fnames, out_name, nodata, crs):
    from rasterio.enums import Resampling
    from rasterio.merge import merge
    import rasterio
    import numpy as np

    src_files_to_mosaic = []
    for f in fnames:
        src = rasterio.open(f)
        src_files_to_mosaic.append(src)
    mosaic, out_trans = merge(datasets=src_files_to_mosaic,
                            #   resampling=Resampling.bilinear,
                            method="first",
                              nodata=nodata)
    out_meta = src.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "crs": crs,
    })
    with rasterio.open(out_dir + out_name, "w", **out_meta) as dest:
        dest.write(mosaic)

lst_ref = xr.open_dataset(
    "/data/home/hamiddashti/nasa_above/outputs/"
    "lst_processed/albers_proj_lst_mean_Annual.nc",
    decode_coords="all")
lst_ref = lst_ref["lst_mean_Annual"].isel(year=0)

in_dir = ("/data/home/hamiddashti/nasa_above/outputs/albedo_processed/"
          "tiles_netcdf_annual/")

out_dir = ("/data/home/hamiddashti/nasa_above/outputs/albedo_processed/"
           "tiles_netcdf_annual/")

epsg_code = 102001
crs = pycrs.parse.from_epsg_code(epsg_code).to_proj4()
fnames = []
for tileID in range(1, 176):
    tmp = in_dir+"tile_" + str(tileID) + ".tif"
    fnames.append(tmp)

mosaicing(out_dir=out_dir,
          fnames=fnames,
          out_name="albedo_annual_mosaic.tif",
          nodata=np.nan,
          crs=crs)


ds = xr.open_rasterio(in_dir+"albedo_annual_mosaic.tif")
time = pd.DatetimeIndex(pd.date_range("2002","2014",freq="AS")).year
ds = ds.assign_coords({"band":time}).rename({"band":"year"})
ds_reproj = ds.rio.reproject_match(lst_ref,resampling=Resampling.nearest)
ds_reproj = ds_reproj.where(lst_ref!=np.nan)

ds_reproj.loc[2003].to_netcdf("/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/test/test_mos3.nc")

ds_reproj.to_netcdf(out_dir+"albedo_annual.nc")

ds_reproj.to_netcdf(out_dir+"albedo_annual2.nc") 
~np.isnan([np.nan,1])