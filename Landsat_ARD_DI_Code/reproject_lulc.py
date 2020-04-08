import xarray as xr
import rasterio 
import rioxarray 
import numpy as np
#from rasterio.enums import Resampling
from landsat_functions import reproject
import os
import glob

in_dir = '/data/ABOVE/LANDSAT/LANDCOVER/Annual_Landcover_ABoVE_1691/data/'
#out_dir = '/data/ABOVE/LANDSAT/LANDCOVER/Annual_Landcover_ABoVE_1691/data_reproject/'
out_dir = '/data/home/hamiddashti/mnt/nasa_above/working/landsat/time_series/'
#in_dir = 'F:\\tmp\\'
#out_dir = 'F:\\tmp\\'

absname = glob.glob(in_dir+'*'+'.tif')
n = len(absname)

target_crs = xr.open_rasterio('/data/ABOVE/LANDSAT/ARD/h08v03/VI_LST/ARDCube_2009_119_008003_LT05.dat').rio.crs
print('Total number of files to be reprojected is:'+str(n))

for i in np.arange(2,3):
    print('projecting file number:'+str(i))
    print(os.path.basename(absname[i]))
    reproject(absname[i],target_crs,out_dir)


