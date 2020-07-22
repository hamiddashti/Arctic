import os
import numpy as np
from osgeo import gdal
gdal.UseExceptions()
import glob

in_dir = '/data/ABOVE/LANDSAT/LANDCOVER/Annual_Landcover_ABoVE_1691/data/Reclassify/'
out_dir = '/data/ABOVE/LANDSAT/LANDCOVER/Annual_Landcover_ABoVE_1691/data/percent_cover/'
fnames = glob.glob(in_dir+'*Simplified*.tif')
years = list(np.arange(1984, 2015))
fnames = fnames[99:]


"""
!!!!!!!!!!! MAKE SURE THE LST NODATA VALUE IS NAN !!!!!!!!!!!!!

"""

for f in fnames:
    
    print(f)
    for i in np.arange(1, 32):

        
        # Get data from raster with classifications

        ds = gdal.Open(f)
        band = ds.GetRasterBand(int(i))
        class_ar = band.ReadAsArray()
        gt = ds.GetGeoTransform()
        pj = ds.GetProjection()
        ds = band = None  # close

        # Define the raster values for each class, to relate to each band
        class_ids = (np.arange(6) + 1).tolist()

        # Make a new bit rasters
        drv = gdal.GetDriverByName("GTiff")
        ds = drv.Create(
            "bit_raster.tif",
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
        src_ds = gdal.Open("bit_raster.tif")

        # Open a template or copy array, for dimensions and NODATA mask
        cpy_ds = gdal.Open(out_dir + "Albedo.tif")
        band = cpy_ds.GetRasterBand(1)

        # WARNING WARNING WARNING: MAKE SURE NODATA IS ACTUALY NAN
        if np.isnan(band.GetNoDataValue()):
            # cpy_mask = (band.ReadAsArray() == band.GetNoDataValue())
            cpy_mask = np.isnan(band.ReadAsArray())
            basename = os.path.basename(f)
            outname = out_dir + str(years[i-1]) + "/" + "percent_cover_" + basename
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

            # Convert all fractions to percent, and apply the same NODATA mask from the copy raster
            NODATA = -9999
            for bidx in range(dst_ds.RasterCount):
                band = dst_ds.GetRasterBand(bidx + 1)
                ar = band.ReadAsArray() * 100.0
                ar[cpy_mask] = NODATA
                band.WriteArray(ar)
                band.SetNoDataValue(NODATA)

            # Save and close all rasters
            src_ds = cpy_ds = dst_ds = band = None
        else:
            print("The No data value of the modis is not NAN!!")
            break
