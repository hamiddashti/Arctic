# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 15:21:30 2020
Main goal of this script is to read the .dat landsat files produced by the Dong
step1 to step3 codes and save them into netcdf. The netcdf files are seperated by the 
bands which includes: 
    
    'NDVI, NIRv, EVI, EVI2, SAVI, MSAVI, NDMI, NBR, NBR2, VI_QA, LST, LST_QA'
    
and are stacked over time 1982-2014. 

Note: these files are original .dat file, in anther word I did not process them for quality etc.
    
@author: hamiddashti
"""

import numpy as np
import xarray as xr
import glob
import pandas as pd
import os
import landsat_functions

# This is the path to the input file    
in_path = '/data/ABOVE/LANDSAT/ARD/h08v03/VI_LST/'

out_path = '/data/ABOVE/LANDSAT/ARD/h08v03/VI_LST_NetCDF_Original/' # this is where netcdf files will be stored

tile = '008003'  # ARD tile number
nrow = 5000      # Numbr of rows of ARD files (5000) 
ncol = 5000      #Numbr of columns of ARD files (5000) 

#################################################################################
# get the filesnames in dat folder and sort them based on date
absname = glob.glob(in_path+'*'+tile+'*'+'.dat')
n=len(absname)
date = []
fnames = []

for i in np.arange(0,n):
    #print(a[i])
    year = int(absname[i][46:50])
    doi = int(absname[i][51:54])
    date.append(pd.to_datetime(year * 1000 + doi, format='%Y%j'))
    fnames.append(os.path.basename(absname[i]))
    
df = pd.DataFrame({'Date':date ,'fnames':fnames})
df = df.sort_values(by='Date')
#################################################################################

# Start reading the dat files from sorted filename and convert them to netcdf

da0 = landsat_functions.open_dat(in_path,df.fnames[0],nrow,ncol)
da0 = da0.assign_coords({'time':df.Date[0]})
NDVI = da0.isel(band=0)
NIRv = da0.isel(band=1)
EVI = da0.isel(band=2)
EVI2 = da0.isel(band=3)
SAVI = da0.isel(band=4)
MSAVI = da0.isel(band=5)
NDMI = da0.isel(band=6)
NBR = da0.isel(band=7)
NBR2 = da0.isel(band=8)
VI_QA = da0.isel(band=9)
LST = da0.isel(band=10)
LST_QA = da0.isel(band=11)

for i in np.arange(1,n):
    print(i)
    da = landsat_functions.open_dat(in_path,df.fnames[i],nrow,ncol)
    da = da.assign_coords({'time':df.Date[i]})
    
    tmp_ndvi = da.isel(band = 0)
    NDVI = xr.concat([NDVI,tmp_ndvi],'time')

    tmp_nirv = da.isel(band = 1)
    NIRv = xr.concat([NIRv,tmp_nirv],'time')
    
    tmp_evi = da.isel(band = 2)
    EVI = xr.concat([EVI,tmp_evi],'time')
    
    tmp_evi2 = da.isel(band = 3)
    EVI2 = xr.concat([EVI2,tmp_evi],'time')
    
    tmp_savi = da.isel(band = 4)
    SAVI = xr.concat([SAVI,tmp_savi],'time')
    
    tmp_msavi = da.isel(band = 5)
    MSAVI = xr.concat([MSAVI,tmp_msavi],'time')
    
    tmp_ndmi = da.isel(band = 6)
    NDMI = xr.concat([NDMI,tmp_ndmi],'time')
    
    tmp_nbr = da.isel(band = 7)
    NBR = xr.concat([NBR,tmp_nbr],'time')
    
    tmp_nbr2 = da.isel(band = 8)
    NBR2 = xr.concat([NBR2,tmp_nbr2],'time')
    
    tmp_vi_qa = da.isel(band = 9)
    VI_QA = xr.concat([VI_QA,tmp_vi_qa],'time')
    
    tmp_lst = da.isel(band = 10)
    LST = xr.concat([LST,tmp_lst],'time')
    
    tmp_lst_qa = da.isel(band = 11)
    LST_QA = xr.concat([LST_QA,tmp_lst_qa],'time')
##############################################################################
# saving the files
print('saving NDVI')
NDVI.to_netcdf(out_path+'NDVI_original.nc')

print('saving NIRv')
NIRv.to_netcdf(out_path+'NIRv_original.nc')

print('saving EVI')
EVI.to_netcdf(out_path+'EVI_original.nc')


print('saving EVI2')
EVI2.to_netcdf(out_path+'EVI2_original.nc')


print('saving SAVI')
SAVI.to_netcdf(out_path+'SAVI_original.nc')


print('saving MSAVI')
MSAVI.to_netcdf(out_path+'MSAVI_original.nc')

print('saving NDMI')
NDMI.to_netcdf(out_path+'NDMI_original.nc')


print('saving NBR')
NBR.to_netcdf(out_path+'NBR_original.nc')


print('saving NBR2')
NBR2.to_netcdf(out_path+'NBR_original.nc')


print('saving VI_QA')
VI_QA.to_netcdf(out_path+'VI_QA_original.nc')


print('saving LST')
LST.to_netcdf(out_path+'LST_original.nc')


print('saving LST_QA')
LST_QA.to_netcdf(out_path+'LST_QA_original.nc')


























  

