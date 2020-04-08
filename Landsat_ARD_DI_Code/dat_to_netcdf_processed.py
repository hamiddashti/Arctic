# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 15:21:30 2020
Main goal of this script is to read the .dat landsat files produced by the Dong
step1 to step3 codes and save them into netcdf. The netcdf files are seperated by the 
bands which includes: 
    
    'NDVI, NIRv, EVI, EVI2, SAVI, MSAVI, NDMI, NBR, NBR2, VI_QA, LST, LST_QA'
    
and are stacked over time 1982-2014. 

Note: these files are filtered based on quality flag provided by Dong code.
    
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

out_path = '/data/ABOVE/LANDSAT/ARD/h08v03/VI_LST_NetCDF_Processed/Lower_Quality' # this is where netcdf files will be stored

tile = '008003'  # ARD tile number
nrow = 5000      # Numbr of rows of ARD files (5000) 
ncol = 5000      #Numbr of columns of ARD files (5000) 

#################################################################################
# get the filesnames in dat folder and sort them based on date
absname = glob.glob(in_path+'*'+tile+'*'+'.dat')
n=len(absname)
print('total files are:')
print(n)
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
df.index = np.arange(0,len(df))
#################################################################################

# Start reading the dat files from sorted filename and convert them to netcdf

da0 = landsat_functions.open_dat(in_path,df.fnames[0],nrow,ncol)
da0 = da0.assign_coords({'time':df.Date[0]})

VI_QA = da0.isel(band=9)
VI_QA = (VI_QA.where(VI_QA.values != -32768 ))

NDVI = da0.isel(band=0)
NDVI = (NDVI.where(NDVI.values != -32768 ))/10000 # Filter for no data
NDVI = NDVI.where(VI_QA.values==20)
NDVI = NDVI.where((NDVI.values > 0.0001) & (NDVI.values <1.0001))
QA = ~xr.ufuncs.isnan(NDVI)

NIRv = da0.isel(band=1)
NIRv = (NIRv.where(NIRv.values != -32768 ))/10000 # Filter for no data
NIRv = NIRv.where(QA)

EVI = da0.isel(band=2)
EVI = (EVI.where(EVI.values != -32768 ))/10000 # Filter for no data
EVI= EVI.where(QA)

EVI2 = da0.isel(band=3)
EVI2 = (EVI2.where(EVI2.values != -32768 ))/10000 # Filter for no data
EVI2= EVI2.where(QA)

SAVI = da0.isel(band=4)
SAVI = (SAVI.where(SAVI.values != -32768 ))/10000 # Filter for no data
SAVI = SAVI.where(QA)

MSAVI = da0.isel(band=5)
MSAVI = (MSAVI.where(MSAVI.values != -32768 ))/10000 # Filter for no data
MSAVI = MSAVI.where(QA)

NDMI = da0.isel(band=6)
NDMI = (NDMI.where(NDMI.values != -32768 ))/10000 # Filter for no data
NDMI = NDMI.where(QA)

NBR = da0.isel(band=7)
NBR = (NBR.where(NBR.values != -32768 ))/10000 # Filter for no data
NBR = NBR.where(QA)

NBR2 = da0.isel(band=8)
NBR2 = (NBR2.where(NBR2.values != -32768 ))/10000 # Filter for no data
NBR2 = NBR2.where(QA)

LST = da0.isel(band=10)
LST = (LST.where(QA))*0.1

LST_QA = da0.isel(band=11)
LST_QA= (LST_QA.where(QA))*0.01



for i in np.arange(1,n):
    print(i)
    
    da = landsat_functions.open_dat(in_path,df.fnames[i],nrow,ncol)
    da = da.assign_coords({'time':df.Date[i]})
    
    tmp_vi_qa = da.isel(band = 9)
    tmp_vi_qa = tmp_vi_qa.where(tmp_vi_qa.values != -32768 )
    VI_QA = xr.concat([VI_QA,tmp_vi_qa],'time')
    
    tmp_ndvi = da.isel(band = 0)
    tmp_ndvi = (tmp_ndvi.where(tmp_ndvi.values != -32768 ))/10000 # Filter for no data
    tmp_ndvi = tmp_ndvi.where(tmp_vi_qa.values==20)
    tmp_ndvi = tmp_ndvi.where((tmp_ndvi.values > 0.0001) & (tmp_ndvi.values <1.0001))
    QA = ~xr.ufuncs.isnan(tmp_ndvi)
    NDVI = xr.concat([NDVI,tmp_ndvi],'time')

    tmp_nirv = da.isel(band = 1)
    tmp_nirv = (tmp_nirv.where(tmp_nirv.values != -32768 ))/10000 # Filter for no data
    tmp_nirv = tmp_nirv.where(QA)
    NIRv = xr.concat([NIRv,tmp_nirv],'time')
    
    tmp_evi = da.isel(band = 2)
    tmp_evi = (tmp_evi.where(tmp_evi.values != -32768 ))/10000 # Filter for no data
    tmp_evi= tmp_evi.where(QA)
    EVI = xr.concat([EVI,tmp_evi],'time')
    
    tmp_evi2 = da.isel(band = 3)
    tmp_evi2 = (tmp_evi2.where(tmp_evi2.values != -32768 ))/10000 # Filter for no data
    tmp_evi2= tmp_evi2.where(QA)
    EVI2 = xr.concat([EVI2,tmp_evi],'time')
    
    tmp_savi = da.isel(band = 4)
    tmp_savi = (tmp_savi.where(tmp_savi.values != -32768 ))/10000 # Filter for no data
    tmp_savi = tmp_savi.where(QA)
    SAVI = xr.concat([SAVI,tmp_savi],'time')
    
    tmp_msavi = da.isel(band = 5)
    tmp_msavi = (tmp_msavi.where(tmp_msavi.values != -32768 ))/10000 # Filter for no data
    tmp_msavi = tmp_msavi.where(QA)
    MSAVI = xr.concat([MSAVI,tmp_msavi],'time')
    
    tmp_ndmi = da.isel(band = 6)
    tmp_ndmi = (tmp_ndmi.where(tmp_ndmi.values != -32768 ))/10000 # Filter for no data
    tmp_ndmi = tmp_ndmi.where(QA)
    NDMI = xr.concat([NDMI,tmp_ndmi],'time')
    
    tmp_nbr = da.isel(band = 7)
    tmp_nbr = (tmp_nbr.where(tmp_nbr.values != -32768 ))/10000 # Filter for no data
    tmp_nbr = tmp_nbr.where(QA)
    NBR = xr.concat([NBR,tmp_nbr],'time')
    
    tmp_nbr2 = da.isel(band = 8)
    tmp_nbr2 = (tmp_nbr2.where(tmp_nbr2.values != -32768 ))/10000 # Filter for no data
    tmp_nbr2 = tmp_nbr2.where(QA)
    NBR2 = xr.concat([NBR2,tmp_nbr2],'time')
    
    tmp_lst = da.isel(band = 10)
    tmp_lst = (tmp_lst.where(QA))*0.1
    LST = xr.concat([LST,tmp_lst],'time')
    
    tmp_lst_qa = da.isel(band = 11)
    tmp_lst_qa= (tmp_lst_qa.where(QA))*0.01
    LST_QA = xr.concat([LST_QA,tmp_lst_qa],'time')
##############################################################################
# saving the files
print('saving NDVI')
NDVI.to_netcdf(out_path+'NDVI_processed.nc')

print('saving NIRv')
NIRv.to_netcdf(out_path+'NIRv_processed.nc')

print('saving EVI')
EVI.to_netcdf(out_path+'EVI_processed.nc')


print('saving EVI2')
EVI2.to_netcdf(out_path+'EVI2_processed.nc')


print('saving SAVI')
SAVI.to_netcdf(out_path+'SAVI_processed.nc')


print('saving MSAVI')
MSAVI.to_netcdf(out_path+'MSAVI_processed.nc')

print('saving NDMI')
NDMI.to_netcdf(out_path+'NDMI_processed.nc')


print('saving NBR')
NBR.to_netcdf(out_path+'NBR_processed.nc')


print('saving NBR2')
NBR2.to_netcdf(out_path+'NBR_processed.nc')


print('saving VI_QA')
VI_QA.to_netcdf(out_path+'VI_QA_processed.nc')


print('saving LST')
LST.to_netcdf(out_path+'LST_processed.nc')


print('saving LST_QA')
LST_QA.to_netcdf(out_path+'LST_QA_processed.nc')
