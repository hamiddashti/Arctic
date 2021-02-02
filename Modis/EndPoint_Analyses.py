'''
Analyzing the results of the EndPoint_Confusion.py. 
Main goal is to plot LST, Albedo etc after a big change between 2003 and 2013
'''

import xarray as xr 
import numpy as np 
import matplotlib.pylab as plt 
import pandas as pd 

lower_list = np.array([7,14,15,21,22,23,28,29,30,31,35,36,37,38,39,40,43,44,45,46,47])
diagnoal_list = np.array([0,8,16,24,32,40,48])
upper_list = np.array([1,2,3,4,5,6,9,10,11,12,13,17,18,19,20,25,26,27,33,34,41])

in_dir = '/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/outputs/Sensitivity/EndPoints/'

ds = xr.open_dataset(in_dir+'Confusion_Table.nc')
Conversion = ds['Conversion']
Conversion_all = np.delete(Conversion,diagnoal_list)
normalized_confusion = ds['NORMALIZED_CONFUSION'].values
lst_lulc = ds['LST_LULC']


normalized_confusion_tmp = np.delete(normalized_confusion,diagnoal_list,axis=1) 
all_zeros = np.all((normalized_confusion_tmp==0),axis=1)
normalized_confusion_all = normalized_confusion_tmp[~all_zeros]

