# Calculate the area of each pixel in geographic crs
import geopandas as gpd

f = gpd.read_file("/data/ABOVE/Final_data/shp_files/ABoVE_1km_Grid_4326.shp")
f_copy = f.copy()
f_copy = f_copy.to_crs({'proj': 'cea'})
f_copy["area"] = f_copy['geometry'].area / 10**6
f_copy.head(2)
a = f_copy.to_crs(4326)
a.to_file("/data/ABOVE/Final_data/shp_files/ABoVE_1km_Grid_4326_area.shp")
