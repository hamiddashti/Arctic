''' 

Created May 17, 2013

bill.smith

Runs MRTBatch operations to mosaic, resample to WGS84 geotiff and crop MODIS imagery to the AOI.

'''

from osgeo import gdal, gdalconst

import os, datetime

import numpy as np

from pyhdf.SD import SD, SDC



def wget_Tiles(cdir, odir, tiles):

	currTiles = []

	files = [f for f in os.listdir(cdir) if f.endswith('.hdf')]

	for i in range (len(files)):

		for j in range (len(tiles)):

			if tiles[j] in files[i]:

				currTiles.append(files[i])

	return currTiles



def get_Tiles(alltiles, tiles):

	currTiles = []

	for i in range (len(alltiles)):

		for j in range (len(tiles)):

			if tiles[j] in alltiles[i]:

				currTiles.append(alltiles[i])

	return currTiles



def band_Info(infile):

	sd = SD(infile, SDC.READ)

	bands = np.array(sd.datasets().keys())

	orderIdxs = np.argsort([sd.nametoindex(bandName) for bandName in bands])

	sd = None

	return bands[orderIdxs]



def spec_Lookup(prod):

	'''returns a list of bands of interest for each product'''

	BandsOfInterestFile = '/Users/Bill/workspace/MODIS_Reproject/src/BOI_by_Prod.txt'

	d = dict((line.strip().split(' = ') for line in file(BandsOfInterestFile)))

	return [int(x) for x in (d[prod]).split(',')]

	



def mrt_Mosaic(tiles, wdir, tdir, prod): 

	''' Mosaics list of tiles for each band of interest, and stores them in a temp directory '''

	

	#create temp file in temp directory with list of files to mosaic

	tempfname = "".join([tdir, "files_to_mosaic.txt"])

	fout = open(tempfname, 'w')

	fnames = ["".join([wdir, x, "\n"]) for x in tiles]

	fout.writelines(fnames)

	fout.close()

	

	#find band names

	bands = band_Info("".join([wdir, tiles[0]]))

	

	#get desired bands to include

	specStrList = spec_Lookup(prod)

	

	#create spectral subset string for each band; mosaic each band separately and output to individual hdfs

	# Note: each band is mosaic separately to avoid exceeding 2GB file size limit (HDF limitation)

	for i in range(len(specStrList)):

		specStr = []

		for j in range(len(bands)):

			specStr.append("0")

		specStr[specStrList[i]] = '1'

		spectralSubset = ' '.join(specStr)

	

		#mosaic files found in the temp list

		os.chdir(wdir)

		mosaic_name = ''.join([tdir, 'mosaic.', bands[specStrList[i]].replace(" ", "_"), '.hdf'])

		mosaic_cmd = "".join(['/Users/Bill/software/bin/mrtmosaic -i ', tempfname, ' -o ', mosaic_name, ' -s "', spectralSubset, '" -g ', tdir, "mosaicLog.txt"])

		print mosaic_cmd

		os.system(mosaic_cmd)

	

	#delete temp file

	del_cmd = ' '.join(['rm', tempfname])

	os.system(del_cmd)



def get_RT(band_name):

	'''Retrieves resample type from lookup file'''

	resamp_chart = '/Users/Bill/workspace/MODIS_Reproject/src/resample_type_chart.txt'

	d = dict((line.strip().split(' = ') for line in file(resamp_chart)))

	return (d[band_name])	



def make_Param(infile, prod, tempDir):

	''' creates parameter file for mrt resample tool and returns the filename of the soon-to-be resampled geotiff'''

	fout = open("".join([tempDir, 'resample.prm']), 'w')



	resampling_type = get_RT(infile[7:-4])

	output_fname = "".join([prod, '.tif']) #MRT will add band name to the file; resulting name = prod.band.tif

	

	lines= "\n\n".join(["".join(["INPUT_FILENAME = ", tempDir, infile]), "SPECTRAL_SUBSET = ( 1 )", "SPATIAL_SUBSET_TYPE = INPUT_LAT_LONG", "SPATIAL_SUBSET_UL_CORNER = ( 59.999999995 -179.999999955 )\nSPATIAL_SUBSET_LR_CORNER = ( 19.999999997 -69.282032295 )", "".join(["OUTPUT_FILENAME = ", tempDir, output_fname]), "".join(["RESAMPLING_TYPE = ", resampling_type]), "OUTPUT_PROJECTION_TYPE = GEO", "OUTPUT_PROJECTION_PARAMETERS = (\n 0.0 0.0 0.0\n 0.0 0.0 0.0\n 0.0 0.0 0.0\n 0.0 0.0 0.0\n 0.0 0.0 0.0 )", 'DATUM = WGS84'])

	

	#if res == 1km: 

		#"\n".join([lines , 'OUTPUT_PIXEL_SIZE = .008333333\n'])



	fout.writelines(lines)

	fout.close()



def get_NoData(infile, band):

	sd = SD(infile, SDC.READ)

	sds = sd.select(band)

	ndata = sds.getfillvalue()

	sd = None

	sds = None

	return ndata

	

def set_Hdr(infile, ndVal):

	sd = gdal.Open(infile, gdalconst.GA_Update)

	band = sd.GetRasterBand(1)

	band.SetNoDataValue(ndVal)

	sd = None

	band = None



def get_lonlat(geot,c,r):

	'''Affine Transform: Converts pixel row and column to spatially referenced coordinates (in native projection)'''

	lon = (geot[0] + c*geot[1] + r*geot[2]) + geot[1] / 2.0

	lat = (geot[3] + c*geot[4] + r*geot[5]) + geot[5] / 2.0

	return lon,lat

	

def correct_GeoT(infile, doCrop, data_mask = None):

	'''corrects GeoTransform values to fix GDAL pixel is point vs pixel is value problem; also crops no data values of a raster'''

	ds = gdal.Open(infile)

	geot = ds.GetGeoTransform()

	proj = ds.GetProjection()

	ndata = ds.GetRasterBand(1).GetNoDataValue()

	dtype = ds.GetRasterBand(1).DataType

	

	a = ds.ReadAsArray()

	if data_mask is None:

		mask_nd = a != ndata

	else:

		mask_nd = data_mask

	

	if doCrop is True:

		nonzero_rows, nonzero_cols = np.nonzero(mask_nd)

		nonzero_rows = np.unique(nonzero_rows)

		nonzero_cols = np.unique(nonzero_cols)

		#Fix. Do not crop out cols/rows in middle of raster

		#nonzero_rows = np.arange(nonzero_rows[0],nonzero_rows[-1]+1)

		#nonzero_cols = np.arange(nonzero_cols[0],nonzero_cols[-1]+1)

		

		min_row, max_row = nonzero_rows[0], nonzero_rows[-1]

		min_col, max_col = nonzero_cols[0], nonzero_cols[-1]

		

		a_crop = a[nonzero_rows, :]

		a_crop = a_crop[:, nonzero_cols]

		

		lon, lat = get_lonlat(geot, min_col, min_row)

		lon = lon-(geot[1]/2.0)

		lat = lat + np.abs(geot[5]/2.0)

		

		rows = a_crop.shape[0]

		columns = a_crop.shape[1]

		a= a_crop

		a_crop = None

		outfile = "".join([infile[:-3], 'cropped.tif'])

		

	else:	

		lon = geot[0]-(geot[1]/2.0)

		lat = geot[3]+ np.abs(geot[5]/2.0)

		rows = ds.GetRasterBand(1).YSize

		columns = ds.GetRasterBand(1).XSize

		outfile = "".join([infile[:-3], 'corrected.tif'])

	

	geot = list(geot)

	geot[0] = lon

	geot[3] = lat

	

	ds_out = gdal.GetDriverByName('GTiff').Create(outfile, columns, rows, 1, dtype)

	band_out = ds_out.GetRasterBand(1)

	

	if ndata is not None:

		band_out.Fill(ndata)

		band_out.SetNoDataValue(ndata)

	

	ds_out.SetGeoTransform(geot)

	ds_out.SetProjection(proj)

	band_out.WriteArray(a)

	ds_out.FlushCache()

	ds_out = None

	ds = None

	

	return outfile



def INT8_to_INT16(infile):

	'''converts INT8 files to INT16 for GDAL compatibility'''

	ds = gdal.Open(infile)

	geot = ds.GetGeoTransform()

	proj = ds.GetProjection()

	ndata = ds.GetRasterBand(1).GetNoDataValue()

	

	a = ds.ReadAsArray()

	a = a.astype('int8')

	b = np.array(a, dtype=np.int16)

	

	cols = b.shape[1]

	rows = b.shape[0]

		

	ds_out = gdal.GetDriverByName('GTiff').Create(infile, cols, rows, 1, 3)

	band_out = ds_out.GetRasterBand(1)

	

	band_out.Fill(ndata)

	band_out.SetNoDataValue(ndata)

	

	ds_out.SetGeoTransform(geot)

	ds_out.SetProjection(proj)

	band_out.WriteArray(b)

	ds_out.FlushCache()

	ds_out = None

	ds = None

	

	return infile

	

def mrt_Resample(infile, prod, tempDir):

	'''resamples hdfs into geotiffs, adds fill value to headers, and corrects GeoTransform values to fix GDAL pixel is point vs pixel is area problem; returns filename with correct GeoT values'''

	

	cmd = "/Users/Bill/software/bin/resample -p resample.prm"

	

	band = band_Info(infile)

	

	make_Param(infile, prod, tempDir)

	

	outfile = ".".join([prod, band[0].replace(" ", "_"), 'tif'])

	

	print cmd

	os.system(cmd)

	os.system("rm resample.prm")

	

	#add fill value to geotiff header

	noData = get_NoData(infile, 0)

	set_Hdr(outfile, noData)

	

	#if dtype = int8, gdal will read as uint8. first convert the gtiff to dtype = INT16 for gdal compatibility

	if band[0] == 'BRDF_Albedo_Band_Mandatory_Quality_Band1':

		outfile = INT8_to_INT16(outfile)

	if band[0] == 'BRDF_Albedo_Band_Mandatory_Quality_Band2':

		outfile = INT8_to_INT16(outfile)

	

	#correct GeoTransform UL corner values (Fix GDAL pixel is point vs pixel is area problem)

	coutfile = correct_GeoT(outfile, False)

	

	#remove mosaicBand.hdf and (uncorrected) prod.band.tif 

	os.system(' '.join(['rm', infile]))

	os.system(' '.join(['rm', outfile]))

	

	return coutfile



def clip(infile):

	''' clips geotiff to a shapefile, then uses correct_GeoT to crop no data values; returns filename of clipped, cropped geotiff '''

	shape = '/Users/Bill/Data/NASA_ABOVE/Boundaries/shp/ABoVE_Study_Domain.shp'

	shapeLayer = 'ABoVE_Study_Domain'

	outfile = ''.join([infile[:-3], 'clipped.tif'])

	

	#get no data value from infile

	sd = gdal.Open(infile)

	ndata = sd.GetRasterBand(1).GetNoDataValue()

	cmd = "".join(['/Users/Bill/anaconda/bin/gdalwarp -cutline "', shape, '" -crop_to_cutline -dstnodata "', str(ndata), '" ', infile, ' ', outfile])

	os.system(cmd)

	coutfile = correct_GeoT(outfile, False)

	

	os.system(' '.join(['rm', infile]))

	os.system(' '.join(['rm', outfile]))

	

	#return coutfile

	return coutfile



def get_AcqDate(tileName):

	''' returns julian and gregorian acquisition dates of tile'''

	#julian acquisition date

	ajdate = tileName[9:16]

	

	#find gregorian acquisition date

	tt = (datetime.datetime.strptime(ajdate, '%Y%j')).timetuple()

	if tt.tm_mon < 10:

		mon = ''.join(['0', str(tt.tm_mon)])

	else:

		mon = str(tt.tm_mon)

	

	if tt.tm_mday < 10:

		day = ''.join(['0', str(tt.tm_mday)])

	else:

		day = str(tt.tm_mday)

	

	agdate = '.'.join([str(tt.tm_year), mon, day])

		

	return ajdate, agdate	



def get_ProdDate():

	'''get production date of tif from the system '''

	now = datetime.datetime.now()

	datestr = ".".join([str(now.year), str(now.month), str(now.day)])

	

	#find and return Julian date of production

	fmt = '%Y.%m.%d'

	dt = datetime.datetime.strptime(datestr, fmt)

	tt = dt.timetuple()

	return ('%d%03d' % (tt.tm_year, tt.tm_yday))



############################################################################################################################################################

products=['MCD43A4']

ModTiles=['h12v01','h13v01','h14v01','h15v01',

'h10v02','h11v02','h12v02','h13v02','h14v02','h15v02',

'h09v03','h10v03','h11v03','h12v03','h13v03','h14v03',

'h08v04','h09v04','h10v04','h11v04','h12v04','h13v04','h14v04']

year=['2018','2019']

year=['2019']

doy=['001','017','033','049','065','081','097',

	'113','129','145','161','177','193','209',

	'225','241','257','273','289','305' ,'321',

	'337','353'] #16days

doy=['257']

outDir = '/Volumes/DataStorage/Data/MODIS/'

outDir2 = ['ABOVE_MCD43A4c6_500m/']

#########DATA PROCESSING###########################################

ModDir = '/Volumes/DataStorage/Data/MODIS/'

ModDir2 = ['MCD43A4_RAW/']

tempDir = outDir + outDir2[0] + 'temp/'

	

for i in range(len(year)):

	for j in range(len(doy)):

		workingDir = "".join([ModDir, ModDir2[0], year[i], '/'])

		AllTiles = sorted([f for f in os.listdir(workingDir) if ''.join([year[i],doy[j],'.']) in f and '.006.' in f])

		tiles = get_Tiles(AllTiles, ModTiles)

		print AllTiles

		if (len(tiles)>0):

			#Mosaic

			mrt_Mosaic(tiles, workingDir, tempDir, products[0])

			#Reproject

			os.chdir(tempDir)

			mosaics = [m for m in os.listdir(tempDir) if m.endswith('.hdf')]

			for k in range(0, len(mosaics)):

				gtif = mrt_Resample(mosaics[k], products[0], tempDir)

				#Crop to shp

				cropped_gtif = clip(gtif)

				#file management

				ajdate, agdate = get_AcqDate(tiles[0])

				prodDate = get_ProdDate()

				out = ''.join([outDir, outDir2[0], year[i], '/', agdate])

				cmd = ''.join(['mkdir -p ', out])

				os.system(cmd)

				cmd = ''.join(['mv ', cropped_gtif, ' ', ''.join([out, '/', cropped_gtif[:-29], 'A', ajdate, '.wgs84', '.', prodDate, '.tif'])])

				os.system(cmd)