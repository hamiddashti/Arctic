import xarray as xr 

in_dir = ("/data/home/hamiddashti/nasa_above/outputs/Sensitivity/EndPoints/"
           "Annual/Albers/")

ds = xr.open_dataset(in_dir+"Confusion_Table_Albers.nc")

ds