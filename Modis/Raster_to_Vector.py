import itertools
import rasterio
from shapely.geometry import box
import geopandas as gpd


with rasterio.open(
    ("/data/home/hamiddashti/nasa_above/outputs/grid/grid_ref.tif")) as dataset:
    data = dataset.read(1)

    t = dataset.transform

    move_x = t[0]
    # t[4] is negative, as raster start upper left 0,0 and goes down
    # later for steps calculation (ymin=...) we use plus instead of minus
    move_y = t[4]

    height = dataset.height
    width = dataset.width

    polygons = []
    indices = list(itertools.product(range(width), range(height)))
    for x, y in indices:
        x_min, y_max = t * (x, y)
        x_max = x_min + move_x
        y_min = y_max + move_y
        polygons.append(box(x_min, y_min, x_max, y_max))

ds = rasterio.open("/data/home/hamiddashti/nasa_above/outputs/grid/grid_ref.tif")

data_list = []
for x, y in indices:
    data_list.append(data[y, x])
gdf = gpd.GeoDataFrame(data=data_list,
                       crs=ds.crs,
                       geometry=polygons,
                       columns=['value'])
gdf.to_file("/data/home/hamiddashti/nasa_above/outputs/grid/python_grid.shp")
