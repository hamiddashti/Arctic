
# This script drives the confusion table of the end points
import numpy as np
import rasterio
import fiona
import pandas as pd
import xarray as xr 
from rasterio import features
from rasterio.mask import mask
import time
t1 = time.time()
analyses_mode  = "Seasonal"
in_dir = "/data/ABOVE/Final_data/"
out_dir = "/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/outputs/"
NUMBER_OF_CLASSES = 7

unique = np.arange(1,NUMBER_OF_CLASSES+1) 
imap = {key: i for i, key in enumerate(unique)}

luc1 = rasterio.open(in_dir+'LUC/geo_Recalssified_2003_Final_Mosaic.tif')
luc2 = rasterio.open(in_dir+'LUC/geo_Recalssified_2013_Final_Mosaic.tif')

seasons = ["DJF","MAM","JJA","SON"]
if analyses_mode =="Seasonal":
	lst = xr.open_dataarray("/data/ABOVE/Final_data/LST_Final/LST/Seasonal_Mean/LST_Mean_MAM.nc")
	lst2003 = lst.loc['2003'].squeeze()
	lst2013 = lst.loc['2013'].squeeze()
	lst_lulc = xr.open_dataarray(
		out_dir
		+ "Natural_Variability/Natural_Variability_"+analyses_mode+"_outputs/EndPoints/"+seasons[1]+"/delta_lst_changed_lulc_component_2013.nc"
	)

	lst_nv = xr.open_dataarray(
	out_dir
	+ "Natural_Variability/Natural_Variability_"+analyses_mode+"_outputs/EndPoints/"+seasons[1]+"/delta_lst_changed_nv_component_2013.nc"
	)

	lst_diff_total = xr.open_dataarray(
	out_dir
	+ "Natural_Variability/Natural_Variability_"+analyses_mode+"_outputs/EndPoints/"+seasons[1]+"/delta_lst_total_2013.nc"
	)

	# albedo = xr.open_dataarray(in_dir + "ALBEDO_Final/Seasonal_Albedo/Albedo_Mean_"+seasons[1]+".nc")
	albedo = xr.open_dataarray(in_dir + "ALBEDO_Final/Seasonal_Albedo/Albedo_Mean_MAM.nc")
	albedo2003 = albedo.loc['2003'].squeeze()
	albedo2013 = albedo.loc['2013'].squeeze()
	albedo_diff = albedo2013 - albedo2003

	EC = xr.open_dataarray(in_dir + "ET_Final/Seasonal_ET/EC_Mean_MAM.nc") # Vegetation transpiration
	EI = xr.open_dataarray(in_dir + "ET_Final/Seasonal_ET/EI_Mean_MAM.nc") # Vegetation transpiration
	ES = xr.open_dataarray(in_dir + "ET_Final/Seasonal_ET/ES_Mean_MAM.nc") # Vegetation transpiration
	EW = xr.open_dataarray(in_dir + "ET_Final/Seasonal_ET/EW_Mean_MAM.nc") # Vegetation transpiration
	ET = xr.open_dataarray(in_dir + "ET_Final/Seasonal_ET/ET_Mean_MAM.nc") # Vegetation transpiration
	EC = EC.fillna(0)
	EI = EI.fillna(0)
	ES = ES.fillna(0)
	EW = EW.fillna(0)
	ET = ET.fillna(0)
	ECI = EC + EI  # canopy evapotranspiration
	ESW = ES + EW  # soil/water/ice/snow evaporation

	EC2003 = EC.loc['2003'].squeeze()
	EC2013 = EC.loc['2013'].squeeze()
	EI2003 = EI.loc['2003'].squeeze()
	EI2013 = EI.loc['2013'].squeeze()
	ES2003 = ES.loc['2003'].squeeze()
	ES2013 = ES.loc['2013'].squeeze()
	EW2003 = EW.loc['2003'].squeeze()
	EW2013 = EW.loc['2013'].squeeze()
	ET2003 = ET.loc['2003'].squeeze()
	ET2013 = ET.loc['2013'].squeeze()
	ECI2003 = ECI.loc['2003'].squeeze()
	ECI2013 = ECI.loc['2013'].squeeze()
	ESW2003 = ESW.loc['2003'].squeeze()
	ESW2013 = ESW.loc['2013'].squeeze()
	
	EC_diff = EC2013 - EC2003
	EI_diff = EI2013 - EI2003
	ES_diff = ES2013 - ES2003
	EW_diff = EW2013 - EW2003
	ET_diff = ET2013 - ET2003
	ECI_diff = ECI2013 - ECI2003
	ESW_diff = ESW2013 - ESW2003

class_names = ["EF","DF","shrub","herb","sparse","wetland","water"]
conversion_type = []
for i in range(0,NUMBER_OF_CLASSES):
	for j in range(0,NUMBER_OF_CLASSES):
		# if (i==j):
		# 	continue
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


shape_file = in_dir+"shp_files/ABoVE_1km_Grid_4326.shp"
print('reading the shapefile')
with fiona.open(shape_file, "r") as shapefile:
	shapes = [feature["geometry"] for feature in shapefile]

lst_2003_vals = np.ravel(lst2003.values,order="F")
lst_2013_vals = np.ravel(lst2013.values,order="F")
lst_lulc_vals = np.ravel(lst_lulc.values,order="F")
lst_nv_vals = np.ravel(lst_nv.values,order="F")
lst_total_vals = np.ravel(lst_diff_total.values,order="F")
albedo_2003_vals = np.ravel(albedo2003.values,order="F")
albedo_2013_vals = np.ravel(albedo2013.values,order="F") 
albedo_diff_vals = np.ravel(albedo_diff.values,order="F")
EC_2003_vals = np.ravel(EC2003.values,order="F")
EC_2013_vals = np.ravel(EC2013.values,order="F")
EI_2003_vals = np.ravel(EI2003.values,order="F")
EI_2013_vals = np.ravel(EI2013.values,order="F")
ES_2003_vals = np.ravel(ES2003.values,order="F")
ES_2013_vals = np.ravel(ES2013.values,order="F")
EW_2003_vals = np.ravel(EW2003.values,order="F")
EW_2013_vals = np.ravel(EW2013.values,order="F")
ET_2003_vals = np.ravel(ET2003.values,order="F")
ET_2013_vals = np.ravel(ET2013.values,order="F")
ESW_2003_vals = np.ravel(ESW2003.values,order="F")
ESW_2013_vals = np.ravel(ESW2013.values,order="F")
ECI_2003_vals = np.ravel(ECI2003.values,order="F")
ECI_2013_vals = np.ravel(ECI2013.values,order="F")
EC_diff_vals = np.ravel(EC_diff.values,order="F")
EI_diff_vals = np.ravel(EI_diff.values,order="F")
ES_diff_vals = np.ravel(ES_diff.values,order="F")
EW_diff_vals = np.ravel(EW_diff.values,order="F")
ET_diff_vals = np.ravel(ET_diff.values,order="F")
ECI_diff_vals = np.ravel(ECI_diff.values,order="F")
ESW_diff_vals = np.ravel(ESW_diff.values,order="F")

final_lst_2003=[]
final_lst_2013=[]
final_normal=[]
final_lst_lulc = []
final_lst_nv = []
final_lst_total = []
final_albedo_2003=[]
final_albedo_2013=[]
final_albedo=[]
final_EC_2003=[]
final_EC_2013=[]
final_EI_2003=[]
final_EI_2013=[]
final_ES_2003=[]
final_ES_2013=[]
final_EW_2003=[]
final_EW_2013=[]
final_ET_2003=[]
final_ET_2013=[]
final_ECI_2003=[]
final_ECI_2013=[]
final_ESW_2003=[]
final_ESW_2013=[]
final_EC = []
final_EI = []
final_ES = []
final_EW = []
final_ET = []
final_ECI = []
final_ESW = []

for i in range(len(shapes)):
	print(i)
	if np.isnan(lst_lulc_vals[i]):
		continue
	luc1_masked = mymask(tif = luc1,shp=[shapes[i]])[0]
	luc2_masked = mymask(tif = luc2,shp=[shapes[i]])[0]
	conf_tmp,conf_normal_tmp =np.asarray(confusionmatrix(luc1_masked.ravel(), luc2_masked.ravel(),unique,imap))
	conf_normal_tmp2 = np.ravel(conf_normal_tmp,order = "C")
	final_normal.append(conf_normal_tmp2)
	final_lst_2003.append(lst_2003_vals[i])
	final_lst_2013.append(lst_2013_vals[i])
	final_lst_lulc.append(lst_lulc_vals[i])
	final_lst_nv.append(lst_nv_vals[i])
	final_lst_total.append(lst_total_vals[i])
	final_albedo_2003.append(albedo_2003_vals[i])
	final_albedo_2013.append(albedo_2013_vals[i])
	final_albedo.append(albedo_diff_vals[i])
	final_EC_2003.append(EC_2003_vals[i])
	final_EC_2013.append(EC_2013_vals[i])
	final_EI_2003.append(EI_2003_vals[i])
	final_EI_2013.append(EI_2013_vals[i])
	final_ES_2003.append(ES_2003_vals[i])
	final_ES_2013.append(ES_2013_vals[i])
	final_EW_2003.append(EW_2003_vals[i])
	final_EW_2013.append(EW_2013_vals[i])
	final_ET_2003.append(ET_2003_vals[i])
	final_ET_2013.append(ET_2013_vals[i])
	final_ECI_2003.append(ECI_2003_vals[i])
	final_ECI_2013.append(ECI_2013_vals[i])
	final_ESW_2003.append(ESW_2003_vals[i])
	final_ESW_2013.append(ESW_2013_vals[i])
	final_EC.append(EC_diff_vals[i])
	final_EI.append(EI_diff_vals[i])
	final_ES.append(ES_diff_vals[i])
	final_EW.append(EW_diff_vals[i])
	final_ET.append(ET_diff_vals[i])
	final_ECI.append(ECI_diff_vals[i])
	final_ESW.append(ESW_diff_vals[i])

final_normal = np.array(final_normal)
final_lst_2003 = np.array(final_lst_2003)
final_lst_2013 = np.array(final_lst_2013)
final_lst_lulc = np.array(final_lst_lulc)
final_lst_nv = np.array(final_lst_nv)
final_lst_total = np.array(final_lst_total)
final_albedo_2003 = np.array(final_albedo_2003)
final_albedo_2013 = np.array(final_albedo_2013)
final_albedo = np.array(final_albedo)
final_EC_2003 = np.array(final_EC_2003)
final_EC_2013 = np.array(final_EC_2013)
final_EI_2003 = np.array(final_EI_2003)
final_EI_2013 = np.array(final_EI_2013)
final_ES_2003 = np.array(final_ES_2003)
final_ES_2013 = np.array(final_ES_2013)
final_EW_2003 = np.array(final_EW_2003)
final_EW_2013 = np.array(final_EW_2013)
final_ET_2003 = np.array(final_ET_2003)
final_ET_2013 = np.array(final_ET_2013)
final_ECI_2003 = np.array(final_ECI_2003)
final_ECI_2013 = np.array(final_ECI_2013)
final_ESW_2003 = np.array(final_ESW_2003)
final_ESW_2013 = np.array(final_ESW_2013)
final_EC = np.array(final_EC)
final_EI = np.array(final_EI)
final_ES = np.array(final_ES)
final_EW = np.array(final_EW)
final_ET = np.array(final_ET)
final_ECI = np.array(final_ECI)
final_ESW = np.array(final_ESW)

ds = xr.Dataset(
data_vars={
"NORMALIZED_CONFUSION":(("ID","Conversion"),final_normal),
"LST_2003":(("ID"),final_lst_2003),
"LST_2013":(("ID"),final_lst_2013),
"DELTA_LST_LULC":(("ID"),final_lst_lulc),
"DELTA_LST_NV":(("ID"),final_lst_nv),
"DELTA_LST_TOTAL":(("ID"),final_lst_total),	
"ALBEDO_2003":(("ID"),final_albedo_2003),
"ALBEDO_2013":(("ID"),final_albedo_2013),
"DELTA_ALBEDO":(("ID"),final_albedo),
"EC_2003":(("ID"),final_EC_2003),
"EC_2013":(("ID"),final_EC_2013),
"EI_2003":(("ID"),final_EI_2003),
"EI_2013":(("ID"),final_EI_2013),
"ES_2003":(("ID"),final_ES_2003),
"ES_2013":(("ID"),final_ES_2013),
"EW_2003":(("ID"),final_EW_2003),
"EW_2013":(("ID"),final_EW_2013),
"ET_2003":(("ID"),final_ET_2003),
"ET_2013":(("ID"),final_ET_2013),
"ECI_2003":(("ID"),final_ECI_2003),
"ECI_2013":(("ID"),final_ECI_2013),
"ESW_2003":(("ID"),final_ESW_2003),
"ESW_2013":(("ID"),final_ESW_2013),
"DELTA_EC":(("ID"),final_EC),
"DELTA_EI":(("ID"),final_EI),
"DELTA_ES":(("ID"),final_ES),
"DELTA_EW":(("ID"),final_EW),
"DELTA_ET":(("ID"),final_ET),
"DELTA_ECI":(("ID"),final_ECI),
"DELTA_ESW":(("ID"),final_ESW)},
coords={
	"ID" : range(len(final_lst_lulc)),
	"Conversion":conversion_type}
)
ds.to_netcdf(out_dir+"Sensitivity/EndPoints/Seasonal/Confusion_Table_MAM.nc")
t2 = time.time()
print(f"Total elapssed time:{t2-t1}")