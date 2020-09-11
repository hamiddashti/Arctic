import xarray as xr
import rioxarray
WL_EL = xr.open_rasterio(in_dir + "Data/Water_Energy_Limited/Tif/WL_EL_Reclassified.tif")
WL_EL = WL_EL.drop('band')

EC_tmp = EC.isel(year=1) 

test = WL_EL.rio.reproject_match(EC_tmp)
test.to_netcdf(in_dir + 'Data/Water_Energy_Limited/Tif/test.nc')