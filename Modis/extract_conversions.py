import numpy as np
# from sklearn.metrics import confusion_matrix
import rasterio
import fiona
import pandas as pd
import xarray as xr 
# import matplotlib.pylab as plt
# import geopandas as gpd
from rasterio import features
from rasterio.mask import mask
import time

t1 = time.time()

in_dir = '/data/ABOVE/Final_data/'
out_dir = '/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/outputs/Sensitivity/'
shape_file = in_dir+"shp_files/ABoVE_1km_Grid_4326.shp"
# shape_file = "/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/test/shp_test.shp"
# shape_file = "one_pix.shp"
NUMBER_OF_CLASSES = 7
PIX_IGNORE = 3 # Minimum number of LUC pixels changed in a modis pixel (3 pixels is slightly above our 1% threshold for considering a LUC change)
years = pd.date_range(start="2003",end="2015",freq="A").year
class_names = ["EF","DF","shrub","herb","sparse","wetland","water"]

conversion_type = []
for i in range(0,NUMBER_OF_CLASSES):
	for j in range(0,NUMBER_OF_CLASSES):
		# if (i==j):
		# 	continue
		# else:
		tmp=class_names[i]+"_"+class_names[j]
		conversion_type.append(tmp)

def mymask(tif,shp):
	# To mask landsat LUC pixels included in each MODIS pixel
	out_image,out_transform = rasterio.mask.mask(tif, shp,all_touched=False, crop=True)
	# out_meta = tif.meta
	# return out_image,out_meta,out_transform
	return out_image, out_transform


def confusionmatrix(actual, predicted,unique,imap):
	"""
	Generate a confusion matrix for multiple classification
	@params:
		actual      - a list of integers or strings for known classes
		predicted   - a list of integers or strings for predicted classes
		# normalize   - optional boolean for matrix normalization
		unique		- is the unique numbers assigned to each class
		imap		- mapping of classes 

	@return:
		matrix      - a 2-dimensional list of pairwise counts
	"""

	matrix = [[0 for _ in unique] for _ in unique]
	# Generate Confusion Matrix
	for p, a in list(zip(actual, predicted)):
		if ((p>len(unique)) or (a>len(unique))):
			continue
		matrix[imap[p]][imap[a]] += 1
	# Matrix Normalization
	# if normalize:
	sigma = sum([sum(matrix[imap[i]]) for i in unique])
	matrix_normalized = [row for row in map(lambda i: list(map(lambda j: j / sigma, i)), matrix)]
	return matrix, matrix_normalized

def skip_diag_strided(A):
	# Removing the diagonal of a matrix
	m = A.shape[0]
	strided = np.lib.stride_tricks.as_strided
	s0,s1 = A.strides
	return strided(A.ravel()[1:], shape=(m-1,m), strides=(s0+s1,s1)).reshape(m,-1)
print('reading the shapefile')
with fiona.open(shape_file, "r") as shapefile:
	shapes = [feature["geometry"] for feature in shapefile]

# inputs to the confusionmatrix function
unique = np.arange(1,NUMBER_OF_CLASSES+1) 
imap = {key: i for i, key in enumerate(unique)}

ds_names=[]
for year in years:
	print(year+1)	
	luc_year1 = in_dir+'LUC/geo_Recalssified_'+str(year)+'_Final_Mosaic.tif'
	luc_year2 = in_dir+'LUC/geo_Recalssified_'+str(year+1)+ '_Final_Mosaic.tif'
	delta_var = in_dir+'LST_Final/LST/Annual_Mean/Natural_Variability_outputs/delta_lst_changed_lulc_component_'+str(year+1)+'.tif'
	# delta_var = '/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/test/delta_lst_luc_'+str(year+1)+'.tif'
	with rasterio.open(delta_var) as var:
		var_image = mymask(var,shapes)[0] # Just extracting the image from rasterio.mask.mask 
		var_image_ravel = var_image.ravel(order = "F")
	
	luc1 = rasterio.open(luc_year1)
	luc2 = rasterio.open(luc_year2)

	confused_total = []
	confused_normalized_Total = []
	for i in range(len(shapes)):
	# for i in range(1000):
		print(i)
		if np.isnan(var_image_ravel[i]):
			continue
		else:
			luc1_masked = mymask(luc1,shp= [shapes[i]])[0] # Just extracting the image from rasterio.mask.mask 
			luc2_masked = mymask(luc2,shp= [shapes[i]])[0] # Just extracting the image from rasterio.mask.mask 
			conf_tmp,conf_normal_tmp =np.asarray(confusionmatrix(luc1_masked.ravel(), luc2_masked.ravel(),unique,imap))
			conf_tmp = np.ravel(skip_diag_strided(conf_tmp),order = "C")
			conf_normal_tmp = np.ravel(skip_diag_strided(conf_normal_tmp),order = "C")
			confused_total.append(conf_tmp) 
			confused_normalized_Total.append(conf_normal_tmp) 

	delta_var_total = var_image_ravel[~np.isnan(var_image_ravel)]
	delta_var_total = delta_var_total.reshape(len(delta_var_total),1)
	confused_total = np.array(confused_total).reshape(len(delta_var_total),len(conversion_type),1)
	confused_normalized_Total = np.array(confused_normalized_Total).reshape(len(delta_var_total),len(conversion_type),1)
	confused_total[confused_total<PIX_IGNORE]=0
	confused_normalized_Total[confused_total<PIX_IGNORE]=0
	
	ds = xr.Dataset(
		data_vars={
		"delta_var_total" : (("ID","time"),delta_var_total),
		"confused_total" : (("ID","Conversions","time"),confused_total),
		"confused_normalized_Total" : (("ID","Conversions","time"),confused_normalized_Total)},
		coords={
		"ID": range(len(delta_var_total)),
		"Conversions":conversion_type,
		"time": np.atleast_1d(year+1)}
		)

	fname =out_dir+"ds_"+str(year+1)+".nc"
	ds.to_netcdf(fname)
	ds_names.append(fname)

ds_final = xr.concat([xr.open_dataset(f) for f in ds_names], dim="time")
print('saving the final dataset')
ds_final.to_netcdf(out_dir+'ds_final.nc')
t2 = time.time()
print(f"Total time: {t2-t1}")









