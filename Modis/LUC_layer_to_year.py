import xarray as xr
import rioxarray
import glob 
import pandas as pd
import numpy as np
import os

indir = '/data/ABOVE/LANDSAT/LANDCOVER/Annual_Landcover_ABoVE_1691/data/'
fnames = glob.glob(indir+"*Simplified_*.tif")
years = pd.date_range(start = '1984',end = '2015',freq='A').year

counter = 0
for f in fnames:
	counter = counter+1
	print(counter)
	da= xr.open_rasterio(f)
	da.attrs.pop('scales')
	da.attrs.pop('offsets')
	for j in np.arange(0,31):
		outdir = '/data/ABOVE/LANDSAT/LANDCOVER/Annual_Landcover_ABoVE_1691/data/years/'+str(years[j])+'/'
		outname = str(years[j])+'_'+os.path.basename(f)
		da.isel(band=j).rio.to_raster(outdir+outname)

