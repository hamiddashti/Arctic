import xarray as xr
import rioxarray
import geopandas
from shapely.geometry import box, mapping


geodf = geopandas.read_file("/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/Above_0804_geographic.shp")
#geodf = geopandas.read_file("F:\\MYD21A2\\Study_area\\Above_0804_Albers.shp")
crs = geodf.crs
file = xr.open_dataset('/data/ABOVE/MODIS/MYD21A2/MYD21A2.006_1km_Geographic.nc',chunks={"lat": 2692, "lon":8089})
#file = xr.open_dataset('F:\\MYD21A2\\Original_Images_CoreDomain\\MYD21A2.006_1km_Geographic.nc')
file = file.rename({"lon":"x","lat":"y"})
#file2=file.drop(['crs'])
#file2 = file2.rio.write_crs(crs)
LST_Day = file['LST_Day_1KM']
LST_Day = LST_Day.rio.write_crs(crs)
print("\n############ Cliping LST Day ####################")
file_clipped = LST_Day.rio.clip(geodf.geometry.apply(mapping), geodf.crs)
file_clipped = file_clipped.rio.set_crs(crs)
file_clipped.to_netcdf("/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/LST_Day_0804.nc")

LST_Night = file['LST_Night_1KM']
LST_Night = LST_Night.rio.write_crs(crs)
print("\n############ Cliping LST Night####################")
file_clipped = LST_Night.rio.clip(geodf.geometry.apply(mapping), geodf.crs)
file_clipped = file_clipped.rio.set_crs(crs)
file_clipped.to_netcdf("/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/LST_Night_0804.nc")

QC_Day = file['QC_Day']
QC_Day = QC_Day.rio.write_crs(crs)
print("\n############ Cliping QC_Day ####################")
file_clipped = QC_Day.rio.clip(geodf.geometry.apply(mapping), geodf.crs)
file_clipped = file_clipped.rio.set_crs(crs)
file_clipped.to_netcdf("/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/QC_Day_0804.nc")

QC_Night = file['QC_Night']
QC_Night = QC_Night.rio.write_crs(crs)
print("\n############ Cliping QC_Night ####################")
file_clipped = QC_Night.rio.clip(geodf.geometry.apply(mapping), geodf.crs)
file_clipped = file_clipped.rio.set_crs(crs)
file_clipped.to_netcdf("/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/QC_Night_0804.nc")

print("\n ############## Done! #####################")
