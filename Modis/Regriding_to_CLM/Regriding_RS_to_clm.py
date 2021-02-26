# ----------------------------------------------------------
#  The main goal is to regrid RS products to CLM resolution
# ----------------------------------------------------------

# Import libraries
# -----------------
import xarray as xr
import matplotlib.pylab as plt
import numpy as np
import xesmf as xe

# Define functions
#-----------------
# LAI products were downloaded from APPEEARS site
# Filter LAI based on QC
#-----------------------
# This should be done before regriding

# Open the files using xarray

# in_dir = '/xdisk/davidjpmoore/hamiddashti/data/tmp_data'
# in_dir = '/data/ABOVE/MODIS/APPEEARS_LAI/lai/'
in_dir = '/groups/davidjpmoore/hamiddashti/nasa_above/regriding_modis_clm/'
# in_dir = '/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/regriding_modis_clm/'

lai_ds = xr.open_dataset(in_dir + 'MCD15A2H.006_500m_aid0001.nc')
chunks = ({'time': 10})
lai = lai_ds['Lai_500m']
lai_std = lai_ds['LaiStdDev_500m']
lai_qc = lai_ds['FparLai_QC']
# lai_tmp = lai.isel(lon=np.arange(1500,1503),lat=np.arange(3000,3003))
# lai_qc_tmp = lai_qc.isel(lon=np.arange(1500,1503),lat=np.arange(3000,3003))
# lai_tmp_filtered = lai_tmp.where(lai_qc_tmp<=50)
print('Filtering LAI based on APPEARS QC <50')
lai_filtered = lai.where(lai_qc <= 50)
lai_std_filtered = lai_std.where(lai_qc <= 50)
lai_filtered.to_netcdf(in_dir + 'LAI_Filtered.nc')
lai_std_filtered.to_netcdf(in_dir + 'LAI_STD_Filtered.nc')

lai_std_filtered = xr.open_dataarray(in_dir + 'LAI_STD_Filtered.nc')
lai_filtered = xr.open_dataarray(in_dir + 'LAI_Filtered.nc')
# ---------------------- Regriding ------------------------

clm_image = xr.open_dataset(in_dir + 'clm_output_sample.nc')

ds_out = xr.Dataset({
    'lat': (['lat'], clm_image['lat'].values),
    'lon': (['lon'], clm_image['lon'].values - 360)
})

regridder = xe.Regridder(lai_filtered, ds_out, 'bilinear')
print(regridder)
ds_lai = regridder(lai_filtered)
ds_lai.to_netcdf(in_dir + 'lai_filtered_regrid.nc')

# regridder = xe.Regridder(lai_std_filtered, ds_out, 'bilinear')
# print(regridder)
ds_lai_std = regridder(lai_std_filtered)
ds_lai_std.to_netcdf(in_dir + 'LAI_STD_Filtered_regrid.nc')

# Take the mean over the region and compare regridded one wiht the original
lai_filtered_mean = lai_filtered.mean(dim=['lat', 'lon'])
lai_filtered_regrid_mean = ds_lai.mean(dim=['lat', 'lon'])
lai_filtered_mean.plot(color='red', label="Original LAI")
lai_filtered_regrid_mean.plot(color='green', label="Regridded LAI")
plt.legend()
plt.tight_layout()
plt.savefig(in_dir + 'lai_original_regridded.png')
plt.close()
#-----------------------End of regriding LAI ---------------

#-----------------------------------------------------------
#       Regriding biomass products downloaded from Ilamb
#-----------------------------------------------------------
