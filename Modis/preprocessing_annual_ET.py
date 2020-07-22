import xarray as xr
import rioxarray
import glob
import pandas as pd

# This script just convert the tif files (PML_V2:
# https://developers.google.com/earth-engine/datasets/catalog/CAS_IGSNRR_PML_V2)
# products into netcdf. The tif files aggregated to annual by summing up the daily data
# on GEE. 


in_dir = '/data/ABOVE/MODIS/ET/Annual_ET/'
out_dir = '/data/ABOVE/MODIS/ET/Annual_ET/' 

date = pd.date_range('2003','2014',freq='AS')
fname = glob.glob(in_dir+'*tif')
chunks = {'x': 8089, 'y': 2692, 'band': 4}
da = xr.concat([xr.open_rasterio(f, chunks=chunks) for f in fname], dim=date.year)
da = da.rename({'concat_dim':'year'})

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

EC.to_netcdf(out_dir+'EC.nc')
ES.to_netcdf(out_dir+'ES.nc')
EI.to_netcdf(out_dir+'EI.nc')
EW.to_netcdf(out_dir+'EW.nc')
ET.to_netcdf(out_dir+'ET.nc')

print("------- All Dnone! --------")

