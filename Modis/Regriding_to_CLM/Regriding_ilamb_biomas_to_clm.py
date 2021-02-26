import xarray as xr
import matplotlib.pylab as plt
import numpy as np
import xesmf as xe

in_dir = "/data/ABOVE/ILAMB_BIOMASS/"
out_dir = ("/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/"
           "regriding_ilamb_biomas_to_clm/")

clm_image = xr.open_dataset(out_dir + 'clm_output_sample.nc')


geocarbon = xr.open_dataset(in_dir+"geocarbon.nc")
global_carbon = xr.open_dataset(in_dir+"global_carbon.nc")
nbcd2000 = xr.open_dataset(in_dir+"nbcd2000.nc")
thurner = xr.open_dataset(in_dir+"thurner.nc")
tropical = xr.open_dataset(in_dir+"tropical.nc")
usfs = xr.open_dataset(in_dir+"usfs.nc")

geocarbon
global_carbon
nbcd2000
thurner
tropical 
usfs 
