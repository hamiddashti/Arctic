# ----------------------------------------------------------
#  The main goal is to regrid RS products to CLM resolution
# ----------------------------------------------------------

# Import libraries
# -----------------
import xarray as xr
import matplotlib.pylab as plt
import numpy as np
import xesmf as xe
from dask.diagnostics import ProgressBar
import matplotlib.pylab as plt
import glob
# Define functions
#-----------------

# Filter LAI based on QC
#-----------------------
# This should be done before regriding

# Open the files using xarray

# in_dir = '/data/ABOVE/MODIS/APPEEARS_LAI/lai/'
# in_dir = '/groups/davidjpmoore/hamiddashti/nasa_above/regriding_modis_clm/'
# in_dir = "/data/ABOVE/MODIS/LAI_CLM_DOMAIN/"

in_dir = "/data/ABOVE/MODIS/APPEEARS_LAI/lai/"
out_dir = "/data/ABOVE/MODIS/APPEEARS_LAI/lai/processed/CLM_Domain/"

chunks = ({"time": 1, "lat": 5383, "lon": 4045})
# lai_ds = xr.open_dataset(in_dir + 'LAI_2002_2020_500m.nc',chunks=chunks)
lai_ds = xr.open_dataset(in_dir + 'LAI_500m.nc')
lai_std_ds = xr.open_dataset(in_dir + 'LAI_500m_std.nc')

lai = lai_ds['Lai_500m']
lai_std = lai_std_ds['LaiStdDev_500m']
lai_qc = lai_ds['FparLai_QC']

for year in range(2011, 2021):
    year = str(year)
    print(year)
    lai_tmp = lai.loc[year]
    lai_std_tmp = lai_std.loc[year]
    lai_qc_tmp = lai_qc.loc[year]
    lai_tmp.to_netcdf(out_dir + "years/lai_" + year + ".nc")
    lai_std_tmp.to_netcdf(out_dir + "years/lai_std_" + year + ".nc")
    lai_qc_tmp.to_netcdf(out_dir + "years/lai_qc_" + year + ".nc")

print('Filtering LAI based on APPEARS QC <50')

for year in range(2011, 2021):
    year = str(year)
    lai_tmp = xr.open_dataarray(out_dir + "years/lai_" + year + ".nc")
    lai_std_tmp = xr.open_dataarray(out_dir + "years/lai_std_" + year + ".nc")
    lai_qc_tmp = xr.open_dataarray(out_dir + "years/lai_qc_" + year + ".nc")
    print("Filtering LAI year: " + year)
    lai_filtered_tmp = lai_tmp.where(lai_qc_tmp <= 50)
    lai_filtered_tmp.to_netcdf(out_dir + "years/LAI_Filtered_" + year + ".nc")
    print("Filtering LAI_StD year: " + year)
    lai_std_filtered_tmp = lai_std_tmp.where(lai_qc_tmp <= 50)
    lai_std_filtered_tmp.to_netcdf(out_dir + "years/LAI_std_Filtered_" + year +
                                   ".nc")

# ---------------------- Regriding ------------------------
clm_image = xr.open_dataset(in_dir + 'PPE.spin81.h0.2017-02-01-00000.nc')
lai_filtered = xr.open_dataarray(out_dir + "years/LAI_Filtered_2011.nc")
ds_out = xr.Dataset({
    'lat': (['lat'], clm_image['lat'].values),
    'lon': (['lon'], clm_image['lon'].values - 360)
})
clm_image

ds_in = xr.Dataset({
    'lat': (['lat'], lai_filtered['lat'].values),
    'lon': (['lon'], lai_filtered['lon'].values)
})

regridder = xe.Regridder(ds_in, ds_out, 'bilinear')
for year in range(2011, 2021):
    year = str(year)
    lai_filtered = xr.open_dataarray(out_dir + "years/LAI_Filtered_" + year +
                                     ".nc")
    lai_std_filtered = xr.open_dataarray(out_dir + "years/LAI_std_Filtered_" +
                                         year + ".nc")
    # ---------------------- Regriding ------------------------
    # ds_out = xr.Dataset({
    #     'lat': (['lat'], clm_image['lat'].values),
    #     'lon': (['lon'], clm_image['lon'].values - 360)
    # })
    print("Regriding LAI:" + year)
    # regridder = xe.Regridder(lai_filtered, ds_out, 'bilinear')
    ds_lai = regridder(lai_filtered)
    ds_lai.to_netcdf(out_dir + "lai_filtered_regrid_" + year + ".nc")
    print("Regriding LAI_StD:" + year)
    # regridder = xe.Regridder(lai_std_filtered, ds_out, 'bilinear')
    # print(regridder)
    ds_lai_std = regridder(lai_std_filtered)
    ds_lai_std.to_netcdf(out_dir + "LAI_STD_Filtered_regrid_" + year + ".nc")

# Take the mean over the region and compare regridded one wiht the original
lai_filtered_mean = lai_filtered.mean(dim=['lat', 'lon'])
lai_filtered_regrid_mean = ds_lai.mean(dim=['lat', 'lon'])
lai_filtered_mean.plot(color='red', label="Original LAI")
lai_filtered_regrid_mean.plot(color='green', label="Regridded LAI")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir + 'lai_original_regridded.png')
plt.close()

#Concat the yearly data
lai_fnames = glob.glob(out_dir + "lai_filtered_regrid*")
da_lai_clm = xr.open_mfdataset(lai_fnames)

std_fnames = glob.glob(out_dir + "std_filtered_regrid*")
std_lai_clm = xr.open_mfdataset(lai_fnames)

ds_lai_final = da_lai_clm.where((da_lai_clm != 0) & ((da_lai_clm.notnull())),
                                -9999.0)
ds_std_final = std_lai_clm.where(
    (std_lai_clm != 0) & ((std_lai_clm.notnull())), -9999.0)

ds_lai_final.to_netcdf(out_dir + "final_resampled/lai_025deg_2011_2020.nc")
ds_std_final.to_netcdf(out_dir + "final_resampled/lai_std_025deg_2011_2020.nc")
