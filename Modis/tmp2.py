import xarray as xr
import rioxarray
import pandas as pd

in_dir = '/data/ABOVE/Final_data/ET_Final/Monthly_ET/'
out_dir = '/data/ABOVE/Final_data/ET_Final/Growing_ET/'

da= xr.open_dataarray(in_dir+'EC.nc')
date = pd.to_datetime(da.time.values)
da = da.assign_coords({"time":date})
da_grouped = da.where(
    da.time.dt.month.isin([4, 5, 6, 7, 8, 9, 10])
) 
da_growing = da_grouped.groupby("time.year").sum()

test = da.sel(time=slice("2003-01","2003-12"))

da_growing.isel(year=1,y=1500,x=4500)
test.isel(y=1500,x=4500)





da_growing.isel(year=1).to_netcdf(out_dir+'da_growing_test.nc')



da.time.dt.month.isin([4, 5, 6, 7, 8, 9, 10])
type(da.time.dt)

date = pd.date_range('12/01/2002','02/28/2015',freq='MS').strftime('%Y-%m')




albedo_growing.to_netcdf(out_dir + "Albedo_growing_"+str(i)+".nc")

