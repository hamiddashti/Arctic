import xarray as xr
import rioxarray
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import imageio 
import modis_functions

in_dir = "F:\\MYD21A2\\outputs\\"
out_dir = "F:\\MYD21A2\\outputs\\DeltaLST\\Animation\\"
class_nemes = [
    "Evergreen Forest",
    "Deciduous Forest",
    "Shrubland",
    "Herbaceous",
    "Sparsely Vegetated",
    "Barren",
    "Fen",
    "Bog",
    "Shallows_Littoral",
    "Water",
]

for i in np.arange(2003,2014):
    print(i)
    name = in_dir+'LULC\\PercentCover\\'+str(i)+'_PercentCover_ABoVE_LandCover_Simplified_Bh08v04.tif'
    luc = xr.open_rasterio(name)
    luc = luc.where(luc!= -9999)
    tmp = luc.isel(band=[0,2,6])
    plt.style.use('dark_background')
    plt.rc("font", family="serif")
    xr.plot.imshow(tmp,rgb='band',robust=True,figsize=(4, 3))
    plt.xlabel('',fontsize=12)
    plt.ylabel('',fontsize=12)
    plt.title(str(i),fontsize=14)
    plt.savefig(out_dir+'LUC_'+str(i)+'.tif')

image_list = []
for i in np.arange(2003,2014):
    f = out_dir+'LUC_'+str(i)+'.tif'
    image_list.append(imageio.imread(f))

imageio.mimwrite(out_dir+'animated_LUC.gif', image_list,fps=2)

# -------------------------------------------------------
df = xr.open_dataarray(in_dir+'LST\\lst_mean_annual.nc')

#year=2003
for year in np.arange(2003,2014):
    df.sel(year=year).plot()
    outname = out_dir+'LST_'+str(year)+'.tif'
    modis_functions.meshplot(df.sel(year=year),outname = outname,mode='presentation',label="LST ($^{\circ}$C)",title = str(year))


image_list = []
for i in np.arange(2003,2014):
    f = out_dir+'LST_'+str(i)+'.png'
    image_list.append(imageio.imread(f))

imageio.mimwrite(out_dir+'animated_LST.gif', image_list,fps=2)

from importlib import reload
reload(modis_functions)