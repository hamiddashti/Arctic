import xarray as xr
import rioxarray
import glob
import pandas as pd
import numpy as np
# This script just convert the tif files (PML_V2:
# https://developers.google.com/earth-engine/datasets/catalog/CAS_IGSNRR_PML_V2)
# products into netcdf. The tif files aggregated to annual by summing up the daily data
# on GEE. 


in_dir = '/data/ABOVE/MODIS/ET/'
out_dir = '/data/ABOVE/MODIS/ET/' 

def growing_sum(da):
	# Resampling monthly data to growing season
	date = pd.to_datetime(da.time.values)
	da = da.assign_coords({"time":date})
	da_grouped = da.where(da.time.dt.month.isin([4, 5, 6, 7, 8, 9, 10])) 
	da_growing = da_grouped.groupby("time.year").sum()
	da_growing = da_growing.where(da_growing!=0)
	return da_growing

date = pd.date_range('12/01/2002','02/28/2015',freq='MS').strftime('%Y-%m')
fname = []
fname.append(in_dir+'Monthly_ET/Dec2002.tif')
for i in np.arange(0,144):
	tmp = in_dir+'Monthly_ET/'+str(i)+'.tif'
	fname.append(tmp)
fname.append(in_dir+'Monthly_ET/Jan2015.tif')
fname.append(in_dir+'Monthly_ET/Feb2015.tif')

chunks = {'x': 8089, 'y': 2692, 'band': 4}
da = xr.concat([xr.open_rasterio(f, chunks=chunks) for f in fname], dim=date)
da = da.rename({'concat_dim':'time'})



"""------------------------------------------------------------------------------

EC ---> Vegetation transpiration [mm/day]
ES ---> Soil evaporation [mm/day]
EI ---> [vaporization of] rain Interception from vegetation canopy
EW ---> Water body, snow and ice evaporation. Penman evapotranspiration is regarded as actual evaporation for them.
ET ---> Total evapotranspiration (EC+ES+EI+EW)

------------------------------------------------------------------------------"""
EC = da.isel(band=0)
ES = da.isel(band=1)
EI = da.isel(band=2)
EW = da.isel(band=3)
ET = da.sum("band")

EC.to_netcdf(out_dir+'Monthly_ET/Final_Monthly_ET/EC_Monthly.nc')
ES.to_netcdf(out_dir+'Monthly_ET/Final_Monthly_ET/ES_Monthly.nc')
EI.to_netcdf(out_dir+'Monthly_ET/Final_Monthly_ET/EI_Monthly.nc')
EW.to_netcdf(out_dir+'Monthly_ET/Final_Monthly_ET/EW_Monthly.nc')
ET.to_netcdf(out_dir+'Monthly_ET/Final_Monthly_ET/ET_Monthly.nc')

# ------------ Growing season ------------------------------

ec= xr.open_dataarray(in_dir+'Monthly_ET/Final_Monthly_ET/EC_Monthly.nc')
ec_growing = growing_sum(ec)
ec_growing.to_netcdf(out_dir+'Growing_ET/EC_Growing.nc')

es= xr.open_dataarray(in_dir+'Monthly_ET/Final_Monthly_ET/ES_Monthly.nc')
es_growing = growing_sum(es)
es_growing.to_netcdf(out_dir+'Growing_ET/ES_Growing.nc')

ei= xr.open_dataarray(in_dir+'Monthly_ET/Final_Monthly_ET/EI_Monthly.nc')
ei_growing = growing_sum(ei)
ei_growing.to_netcdf(out_dir+'Growing_ET/EI_Growing.nc')

ew= xr.open_dataarray(in_dir+'Monthly_ET/Final_Monthly_ET/EW_Monthly.nc')
ew_growing = growing_sum(ew)
ew_growing.to_netcdf(out_dir+'Growing_ET/EW_Growing.nc')

et= xr.open_dataarray(in_dir+'Monthly_ET/Final_Monthly_ET/ET_Monthly.nc')
et_growing = growing_sum(et)
et_growing.to_netcdf(out_dir+'Growing_ET/ET_Growing.nc')

# ----------- Seasonal mean --------------------------------
date = pd.to_datetime(ec.time.values)
ec= xr.open_dataarray(in_dir+'Monthly_ET/Final_Monthly_ET/EC_Monthly.nc')
ec = ec.assign_coords({"time":date})
ec_season_resample = ec.resample(time="QS-DEC").sum()
ec_season_resample.to_netcdf(out_dir +'Seasonal_ET/EC_Season.nc')


es= xr.open_dataarray(in_dir+'Monthly_ET/Final_Monthly_ET/ES_Monthly.nc')
es = es.assign_coords({"time":date})
es_season_resample = es.resample(time="QS-DEC").sum()
es_season_resample.to_netcdf(out_dir +'Seasonal_ET/ES_Season.nc')

ei= xr.open_dataarray(in_dir+'Monthly_ET/Final_Monthly_ET/EI_Monthly.nc')
ei = ei.assign_coords({"time":date})
ei_season_resample = ei.resample(time="QS-DEC").sum()
ei_season_resample.to_netcdf(out_dir +'Seasonal_ET/EI_Season.nc')

ew= xr.open_dataarray(in_dir+'Monthly_ET/Final_Monthly_ET/EW_Monthly.nc')
ew = ew.assign_coords({"time":date})
ew_season_resample = ew.resample(time="QS-DEC").sum()
ew_season_resample.to_netcdf(out_dir +'Seasonal_ET/EW_Season.nc')

et= xr.open_dataarray(in_dir+'Monthly_ET/Final_Monthly_ET/ET_Monthly.nc')
et = et.assign_coords({"time":date})
et_season_resample = et.resample(time="QS-DEC").sum()
et_season_resample.to_netcdf(out_dir +'Seasonal_ET/ET_Season.nc')


# ----------- Group by year --------------------------------
ec_annual = ec.groupby("time.year").sum(dim="time")
ec_annual.to_netcdf(out_dir + "Annual_ET/EC_Annual.nc")

es_annual = es.groupby("time.year").sum(dim="time")
es_annual.to_netcdf(out_dir + "Annual_ET/ES_Annual.nc")

ei_annual = ei.groupby("time.year").sum(dim="time")
ei_annual.to_netcdf(out_dir + "Annual_ET/EI_Annual.nc")

ew_annual = ew.groupby("time.year").sum(dim="time")
ew_annual.to_netcdf(out_dir + "Annual_ET/EW_Annual.nc")

et_annual = et.groupby("time.year").sum(dim="time")
et_annual.to_netcdf(out_dir + "Annual_ET/ET_Annual.nc")

print("------- All Dnone! --------")