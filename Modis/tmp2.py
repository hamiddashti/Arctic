import dask
import xarray as xr 
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np

def reject_outliers(data, m):
	# m is number of std
	import numpy as np
	data = data.astype(float)
	data[abs(data - np.nanmean(data)) > m * np.nanstd(data)] = np.nan
	
	return data

def lsg(predictors,target):
	# least square solution
	a = np.linalg.inv(np.matmul(predictors.T, predictors))
	b = np.matmul(predictors.T,target)
	coefficients = np.matmul(a,b)
	return coefficients

ds = xr.open_dataset('/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/outputs/Sensitivity/ds_2004.nc')
labels = ds['Conversions'].values

ignore_list = np.array([6,12,13,18,19,20,24,25,26,27,30,31,32,33,34,36,37,38,39,40,41])
include_list = np.array([0,1,2,3,4,5,7,8,9,10,11,14,15,16,17,21,22,23,28,29,35])
labels[include_list]

X = ds['confused_normalized_Total'].squeeze().values
Y = ds['delta_var_total'].values

# Remove rows with all zeros
I = np.where(X.any(axis=1))[0]
X = X[I]
Y=Y[I]

for i in ignore_list:
	# Remove all rows (i.e. pixels) where there is reverse transition
	print(i)
	Y= Y[(X[:,i]==0)]
	X= X[(X[:,i]==0),:]

# Remove all the columns associated with reverse transitions
XX = X[:,include_list]

# Find and remove outliers
Y_clean = reject_outliers(Y,2)
yy = Y_clean[~np.isnan(Y_clean)]
XX = XX[np.array(~np.isnan(Y_clean)).squeeze(),:]


# Create a probability vector 
# count_nonzero = np.count_nonzero(XX,axis=0)
# prob=(1/(count_nonzero/sum(count_nonzero)))/sum(1/(count_nonzero/sum(count_nonzero)))
# import time 
# t1=time.time()
# var_slope = []
# for n in range(0,10):
# 	print(n)
# 	subset_x = []
# 	subset_y = []
# 	count=0
# 	while (count < 1000):
# 		# print(count)
# 		# Now choose a random number with the assgined probabilities 
# 		pp1 = np.random.choice(np.arange(0,XX.shape[1]),size=1)
# 		# Returns the index of all rows where column pp2 is not zero
# 		idx = np.nonzero(XX[:,pp1])[0]
# 		# if idx.size==0:
# 		# 	count = count+1
# 		# 	continue
# 		# Now draw another random sample from indices
# 		idx2 = idx[np.random.choice(np.arange(0,len(idx)),size=1)]
# 		subset_x.append(XX[idx2,:])
# 		subset_y.append(yy[idx2])

# 		# if len(subset)==1000:
# 		# 	break
# 		# Remove that sample form the dataset
# 		# XX = np.delete(XX,idx2,axis=0)
# 		count = count+1

# 	xx = np.asarray(subset_x).squeeze()
# 	x_final = np.c_[xx,np.ones(xx.shape[0])]  # add bias term
# 	y_final = np.asarray(subset_y).squeeze()
# 	estimates = lsg(predictors=x_final,target=y_final)
# 	var_slope.append(estimates)
# var_slope_mc= np.asarray(var_slope).squeeze()
# var_slope_mc_mean = np.mean(var_slope_mc,axis=0)
# var_slope_mc_std = np.std(var_slope_mc,axis=0)

# t2=time.time()
# print(t2-t1)

def my_cal(n):
	var_slope = []
	# for n in range(0,10):
	print(n)
	subset_x = []
	subset_y = []
	count=0
	while (count < 10000):
		# print(count)
		# Now choose a random number with the assgined probabilities 
		pp1 = np.random.choice(np.arange(0,XX.shape[1]),size=1)
		# Returns the index of all rows where column pp2 is not zero
		idx = np.nonzero(XX[:,pp1])[0]
		# if idx.size==0:
		# 	count = count+1
		# 	continue
		# Now draw another random sample from indices
		idx2 = idx[np.random.choice(np.arange(0,len(idx)),size=1)]
		subset_x.append(XX[idx2,:])
		subset_y.append(yy[idx2])

		# if len(subset)==1000:
		# 	break
		# Remove that sample form the dataset
		# XX = np.delete(XX,idx2,axis=0)
		count = count+1

	xx = np.asarray(subset_x).squeeze()
	x_final = np.c_[xx,np.ones(xx.shape[0])]  # add bias term
	y_final = np.asarray(subset_y).squeeze()
	estimates = lsg(predictors=x_final,target=y_final)
	var_slope.append(estimates)
	return var_slope

import time 
t1 = time.time()
dask.config.set(scheduler='processes')
	#IDs = np.arange(1, len(geodf)+1)
IDs = np.arange(0,10000)
lazy_results=[]
for n in IDs:
	lazy_result = dask.delayed(my_cal)(n)
	lazy_results.append(lazy_result)

from dask.diagnostics import ProgressBar
with ProgressBar():
	futures = dask.persist(*lazy_results)
	results = dask.compute(*futures)
t2 = time.time()

print(t2-t1) 

var_slope_mc= np.asarray(results).squeeze()
var_slope_mc_mean = np.mean(var_slope_mc,axis=0)
var_slope_mc_std = np.std(var_slope_mc,axis=0)

print("Mean is: ")
print(var_slope_mc_mean)
