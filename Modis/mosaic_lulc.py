import xarray as xr
import pandas as pd
import rasterio
from rasterio.merge import merge
import glob

years = pd.date_range(start = '1984',end = '2015',freq='A').year

for year in years:
    
    in_dir = "/data/ABOVE/LANDSAT/LANDCOVER/Annual_Landcover_ABoVE_1691/data/years/"+str(year)+"/"
    out_dir = "/data/ABOVE/LANDSAT/LANDCOVER/Annual_Landcover_ABoVE_1691/data/years/mosaic/"+str(year)+"_Final_Mosaic.tif"

    fnames = glob.glob(in_dir + "**Sim*.tif")
    
    src_files_to_mosaic = []

    for fp in fnames:
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)

    mosaic, out_trans = merge(src_files_to_mosaic)

    out_meta = src.meta.copy()

    # Update the metadata
    out_meta.update(
        {
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "crs": src.crs
        }
    )
    print("saving " +str(year))
    with rasterio.open(out_dir, "w", **out_meta,compress='lzw') as dest:
        dest.write(mosaic)


