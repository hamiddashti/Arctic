# This is a file to reproject the lst files from
import xarray as xr
import rioxarray
from pyproj import CRS
import glob
import os
from rasterio.warp import Resampling


lst = xr.open_dataset("lst.nc") # File wich carries the original CRS


luc = xr.open_rasterio("luc.tif") # File with the desired projection system

cc_lst = CRS.from_cf(lst.crs.attrs)
cc_luc = luc.rio.crs
>>> print(cc_lst)
PROJCS["unnamed",GEOGCS["WGS 84",DATUM["WGS_1984",
SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]]
,AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]]
,UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]
,AUTHORITY["EPSG","4326"]],PROJECTION["Albers_Conic_Equal_Area"]
,PARAMETER["standard_parallel_1",55],PARAMETER["standard_parallel_2",65]
,PARAMETER["latitude_of_center",50],PARAMETER["longitude_of_center",-154]
,PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["Meter",1]]
>>> print(cc_luc)
CRS.from_wkt('PROJCS["unknown",GEOGCS["unknown",DATUM
["Unknown based on GRS80 ellipsoid",SPHEROID
["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],TOWGS84[0,0,0,0,0,0,0]]
,PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]]
,UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],
PROJECTION["Albers_Conic_Equal_Area"],PARAMETER["latitude_of_center",40]
,PARAMETER["longitude_of_center",-96],PARAMETER["standard_parallel_1",50]
,PARAMETER["standard_parallel_2",70],PARAMETER["false_easting",0],
PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]]
,AXIS["Easting",EAST],AXIS["Northing",NORTH]]')

file_to_reproject = xr.open_dataarray("myfile.nc")

file_to_reproject = file_to_reproject.rio.write_crs(cc_lst)
file_reprojected= file_to_reproject.rio.reproject(cc_luc,
                                    resampling=Resampling.bilinear)
print(file_reprojected)
<xarray.DataArray (season: 4, y: 4343, x: 4172)>
array([[[nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan]]])
Coordinates:
  * x            (x) float64 -3.382e+06 -3.381e+06 ... 4.817e+05 4.826e+05
  * y            (y) float64 5.361e+06 5.36e+06 ... 1.338e+06 1.337e+06
  * season       (season) object 'DJF' 'JJA' 'MAM' 'SON'
    spatial_ref  int64 0

# however after saving the file and reloading it I get a dataset where 
# spatial_ref turned into variable 
file_reprojected.to_netcdf("file_reprojected.nc")
ds= xr.open_dataset("file_reprojected.nc")

print(ds)
<xarray.Dataset>
Dimensions:                        (season: 4, x: 4172, y: 4343)
Coordinates:
  * x                              (x) float64 -3.382e+06 ... 4.826e+05
  * y                              (y) float64 5.361e+06 5.36e+06 ... 1.337e+06
  * season                         (season) object 'DJF' 'JJA' 'MAM' 'SON'
Data variables:
    spatial_ref                    int64 ...
    __xarray_dataarray_variable__  (season, y, x) float64 ...

# Here is the content of spatial_ref
>>> ds["spatial_ref"]
<xarray.DataArray 'spatial_ref' ()>
array(0)
Attributes: (12/19)
    crs_wkt:                        PROJCS["unknown",GEOGCS["unknown",DATUM["...
    semi_major_axis:                6378137.0
    semi_minor_axis:                6356752.314140356
    inverse_flattening:             298.257222101
    reference_ellipsoid_name:       GRS 1980
    longitude_of_prime_meridian:    0.0
    ...                             ...
    longitude_of_central_meridian:  -96.0
    false_easting:                  0.0
    false_northing:                 0.0
    towgs84:                        [0. 0. 0. 0. 0. 0. 0.]
    spatial_ref:                    PROJCS["unknown",GEOGCS["unknown",DATUM["...
    GeoTransform:                   -3382530.3119257297 926.5510319351852 0.0...