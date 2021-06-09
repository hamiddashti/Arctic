import os
import numpy as np
from osgeo import gdal

gdal.UseExceptions()
import glob
import xarray as xr
import pandas as pd

in_dir = ("/data/ABOVE/LANDSAT/LANDCOVER/Annual_Landcover_ABoVE_1691/"
          "data/mosaic/")
# out_dir = ("/data/home/hamiddashti/nasa_above/outputs/percent_cover/"
#            "percent_cover_albert/")
out_dir = (
    "/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/test/")

# Create the reference raster with coarser resolution (MODIS; 1000m)
# A modis sample
# mean_annual = xr.open_dataarray("/data/ABOVE/ABoVE_Final_Data/LST/processed/"
#                                 "lst_mean_Annual.nc")
# An LUC sample to get the ABoVE CRS
# luc_2003 = xr.open_rasterio("/data/ABOVE/LANDSAT/LANDCOVER/"
# "Annual_Landcover_ABoVE_1691/data/mosaic/mosaic_2003.tif")
# a = mean_annual.isel(year=0)
# a = a.rename({"xdim":"x","ydim":"y"})
# cc = CRS.from_cf(lst.crs.attrs)
# a = a.rio.write_crs(cc)
# b = a.rio.reproject(luc_2003.crs)
# b.rio.to_raster(in_dir+"ref_raster.tif")


years = pd.date_range(start="1984", end="2015", freq="Y").year
year = years[15]
fname_all = []
for year in years:
    print("calculate percent cover year --> " + str(year))
    fname = "mosaic_" + str(year) + ".tif"
    # Get data from raster with classifications
    ds = gdal.Open(in_dir + fname)
    band = ds.GetRasterBand(1)
    class_ar = band.ReadAsArray()
    gt = ds.GetGeoTransform()
    pj = ds.GetProjection()
    ds = band = None  # close

    # Define the raster values for each class, to relate to each band
    class_ids = (np.arange(10) + 1).tolist()

    # Make a new bit rasters
    bit_name = str(year) + "_bit_raster.tif"
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(
        out_dir + bit_name,
        class_ar.shape[1],
        class_ar.shape[0],
        len(class_ids),
        gdal.GDT_Byte,
        ["NBITS=1", "COMPRESS=LZW", "INTERLEAVE=BAND"],
    )
    ds.SetGeoTransform(gt)
    ds.SetProjection(pj)
    for bidx in range(ds.RasterCount):
        band = ds.GetRasterBand(bidx + 1)
        # create boolean
        selection = class_ar == class_ids[bidx]
        band.WriteArray(selection.astype("B"))

    ds = band = None  # save, close

    # Open raster from step 1
    src_ds = gdal.Open(out_dir + bit_name)

    # Open a template or copy array, for dimensions and NODATA mask
    cpy_ds = gdal.Open(in_dir + "ref_raster.tif")
    band = cpy_ds.GetRasterBand(1)

    # WARNING WARNING WARNING: MAKE SURE NODATA IS ACTUALY NAN
    if np.isnan(band.GetNoDataValue()):
        # cpy_mask = (band.ReadAsArray() == band.GetNoDataValue())

        cpy_mask = np.isnan(band.ReadAsArray())
        # basename = os.path.basename(f)
        outname = out_dir + str(year) + "_percent_cover64.tif"
        # Result raster, with same resolution and position as the copy raster
        dst_ds = drv.Create(
            outname,
            cpy_ds.RasterXSize,
            cpy_ds.RasterYSize,
            len(class_ids),
            gdal.GDT_Float64,
            ["INTERLEAVE=BAND"],
        )
        dst_ds.SetGeoTransform(cpy_ds.GetGeoTransform())
        dst_ds.SetProjection(cpy_ds.GetProjection())

        # Do the same as gdalwarp -r average; this might take a while to finish
        gdal.ReprojectImage(src_ds, dst_ds, None, None, gdal.GRA_Average)

        # Convert all fractions to percent, and apply the same
        # NODATA mask from the copy raster
        NODATA = np.nan
        for bidx in range(dst_ds.RasterCount):
            band = dst_ds.GetRasterBand(bidx + 1)
            ar = band.ReadAsArray()
            ar[cpy_mask] = NODATA
            band.WriteArray(ar)
            # band.SetNoDataValue(NODATA)
        # Save and close all rasters
        src_ds = cpy_ds = dst_ds = band = None

        fname_all.append(outname)
    else:
        print("The No data value of the modis is not NAN!!")
        break

fname_all = []
for year in years:
    f = out_dir + str(year) + "_percent_cover.tif"
    fname_all.append(f)
chunks = {"y": 4343, "x": 4172}
da = xr.concat([xr.open_rasterio(f, chunks=chunks) for f in fname_all],
               dim=years)
da = da.rename({"concat_dim": "time", "x": "lon", "y": "lat"})
da_2003_2014 = da.loc[2003:2014]
da.to_netcdf(out_dir + "LULC_10.nc")
da_2003_2014.to_netcdf(out_dir + "LULC_10_2003_2014.nc")
print("all done!")
