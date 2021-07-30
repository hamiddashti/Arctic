# This is figure 1 of the MS. After exporting the netcdf files the rest of the
# work has been done in Arcmap
import xarray as xr
import matplotlib.pylab as plt
import rioxarray

out_dir = "/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/test/"

lcc = xr.open_dataarray(
    ("/data/home/hamiddashti/nasa_above/outputs/percent_cover/"
     "percent_cover_albert/LULC_10_2003_2014.nc"),
    decode_cf="all")
lcc = lcc.rename({"lat": "y", "lon": "x"})

dlcc = abs(lcc.loc[2013] - lcc.loc[2003])
dlcc_max_changed = dlcc.max(dim="band")
dlcc_max_changed.rio.to_raster(out_dir + "test.tif")
dlcc_max_changed.to_netcdf(out_dir + "percent_change.nc")

lc_init = lcc.loc[2013]
lc_init_fill = lc_init.fillna(-9999)
dominant_lc = lc_init_fill.argmax(dim="band", skipna=True)
dominant_lc = dominant_lc.where(dlcc_max_changed.notnull())
dominant_lc.to_netcdf(out_dir + "dominant_lc_2013.nc")

lc_init = lcc.loc[2003]
lc_init_fill = lc_init.fillna(-9999)
dominant_lc = lc_init_fill.argmax(dim="band", skipna=True)
dominant_lc = dominant_lc.where(dlcc_max_changed.notnull())
dominant_lc.to_netcdf(out_dir + "dominant_lc_2003.nc")
