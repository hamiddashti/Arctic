import xarray as xr

in_dir_2011 = (
    "/data/ABOVE/MODIS/APPEEARS_LAI/lai/processed/CLM_Domain/final_resampled/")
in_dir_2011_2020 = (
    "/data/ABOVE/MODIS/LAI_CLM_DOMAIN/processed/final_resampled/")

lai_2011 = xr.open_dataarray(in_dir_2011 + "lai_025deg_2011_2020.nc")
lai_std_2011 = xr.open_dataarray(in_dir_2011 + "lai_std_025deg_2011_2020.nc")
lai_2011_2020 = xr.open_dataarray(in_dir_2011_2020 + "lai_025deg_2011_2020.nc")
lai_std_2011_2020 = xr.open_dataarray(in_dir_2011_2020 +
                                      "lai_std_025deg_2011_2020.nc")

lai_2011_2020.isel(time=32).to_netcdf(
    "/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/test/lai_test2.nc"
)

lai_std_2011_2020.isel(time=32).to_netcdf(
    "/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/test/std_test.nc"
)

lai_2011.isel(time=32).to_netcdf(
    "/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/test/lai_test_above.nc"
)

lai_std_2011.isel(time=32).to_netcdf(
    "/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/test/std_test_above.nc"
)

