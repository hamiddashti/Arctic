import os
import numpy as np
from osgeo import gdal
gdal.UseExceptions()
import glob
import xarray as xr

in_dir = ("/data/ABOVE/LANDSAT/LANDCOVER/Annual_Landcover_ABoVE_1691/"
          "data/years/mosaic/")
out_dir = ("/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/"
           "outputs/percent_cover/")
years = list(np.arange(1984, 2015))

for year in years:

    fname = str(year) + "_Final_Mosaic.tif"
    print(fname)
    # for i in np.arange(1, 2):

    # Get data from raster with classifications

    ds = gdal.Open(fname)
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
    cpy_ds = gdal.Open(out_dir + "lst_ref.tif")
    band = cpy_ds.GetRasterBand(1)

    # WARNING WARNING WARNING: MAKE SURE NODATA IS ACTUALY NAN
    if np.isnan(band.GetNoDataValue()):
        # cpy_mask = (band.ReadAsArray() == band.GetNoDataValue())
        cpy_mask = np.isnan(band.ReadAsArray())
        # basename = os.path.basename(f)
        outname = out_dir + str(year) + "_percent_cover.tif"
        # Result raster, with same resolution and position as the copy raster
        dst_ds = drv.Create(
            outname,
            cpy_ds.RasterXSize,
            cpy_ds.RasterYSize,
            len(class_ids),
            gdal.GDT_Float32,
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
    else:
        print("The No data value of the modis is not NAN!!")
        break