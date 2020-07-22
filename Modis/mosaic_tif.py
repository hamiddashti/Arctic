import modis_functions
import numpy as np
import glob
years = list(np.arange(1984, 2015))
for year in years:
	print('Working on: '+str(year))
	in_dir = '/data/ABOVE/LANDSAT/LANDCOVER/Annual_Landcover_ABoVE_1691/data/percent_cover/'+str(year)+'/'
	out_dir = '/data/ABOVE/LANDSAT/LANDCOVER/Annual_Landcover_ABoVE_1691/data/percent_cover/mosaic/'
	fnames = glob.glob(in_dir+'*.tif')
	outname = str(year)+ '_percent_cover2.tif'
	modis_functions.mosaicing(out_dir = out_dir, fnames = fnames, out_name = outname,nodata=0)