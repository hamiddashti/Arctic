from matplotlib.pyplot import colorbar, ylim
import xarray as xr
import matplotlib.pylab as plt
import cartopy.crs as ccrs
import rioxarray
import cartopy
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


def outliers_index(data, m=3.5):
    """
    Returns true if a value is outlier
    https://www.itl.nist.gov/div898/handbook/eda/section3/eda356.htm#MAD
    :param int data: numpy array
    :param int m: # of std to include data 
    """
    import numpy as np
    d = np.abs(data - np.nanmedian(data))
    mdev = np.nanmedian(d)
    s = d / mdev if mdev else 0.
    return ~(s < m)


in_dir = "/data/ABOVE/Final_data/"
out_dir = ("/data/home/hamiddashti/nasa_above/outputs/data_analyses/Annual/"
           "Albers/Figures_MS1/")
out_dir = "/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/test/"
lst_mean = xr.open_dataset(
    in_dir +
    ("LST_Final/LST/Annual_Mean/albers/albers_proj_lst_mean_Annual.nc"),
    decode_cf="all")
lst_mean = lst_mean["lst_mean_Annual"]
lst_mean = lst_mean.rename({"x": "lon", "y": "lat"})
albedo = xr.open_dataarray(
    in_dir + "ALBEDO_Final/Annual_Albedo/albers/final_annual_albedo.nc",
    decode_cf="all")
albedo = albedo.rename({"x": "lon", "y": "lat"})
et = xr.open_dataset(in_dir +
                     "ET_Final/Annual_ET/albers/albers_proj_ET_Annual.nc",
                     decode_cf="all")
et = et["ET_Annual"]
et = et.rename({"x": "lon", "y": "lat"})
et = et.where(lst_mean.notnull())

# luc_dir = (
#     "/data/ABOVE/LANDSAT/LANDCOVER/Annual_Landcover_ABoVE_1691/data/mosaic/")
# luc2003 = xr.open_rasterio(luc_dir + 'mosaic_2003.tif')
# crs = luc2003.rio.crs

dlst_total = lst_mean.loc[2013] - lst_mean.loc[2003]
dalbedo_total = albedo.loc[2013] - albedo.loc[2003]
det_total = et.loc[2013] - et.loc[2003]

I_dlst_total = outliers_index(dlst_total)
I_dalbedo_total = outliers_index(dalbedo_total)
I_det_total = outliers_index(det_total)

dlst_total_clean = dlst_total.where((I_dlst_total == False)
                                    & (I_dalbedo_total == False)
                                    & (I_det_total == False))
dalbedo_total_clean = dalbedo_total.where((I_dlst_total == False)
                                          & (I_dalbedo_total == False)
                                          & (I_det_total == False))
det_total_clean = det_total.where((I_dlst_total == False)
                                  & (I_dalbedo_total == False)
                                  & (I_det_total == False))

titles = ["$\Delta LST$", "$\Delta ALBEDO$", "$\Delta ET$"]
labels = ["[K]", "", "[mm]"]
data = [dlst_total_clean, dalbedo_total_clean, det_total_clean]
# ylim = [1685818.55149696, 4000000.0]

ylim = [1400000.0, 4000000.0]
xlim = [-3300000.0, 0]

plt.close()
ar = 1.0  # initial aspect ratio for first trial
wi = 9  # width of the whole figure in inches, ...
# set it wide enough to cover all columns of sub-plots
hi = wi * ar  # height in inches
# set number of rows/columns
rows, cols = 3, 1
gs = gridspec.GridSpec(rows, cols)
fig = plt.figure(figsize=(wi, hi))
for k in range(0, rows * cols):
    ax = plt.subplot(gs[k], projection=ccrs.AlbersEqualArea())
    im = data[k].plot.imshow(ax=ax, ylim=ylim, xlim=xlim, add_colorbar=False)

    cb = plt.colorbar(im, fraction=0.046, pad=0.04, shrink=0.82)
    cb.set_label(label=labels[k], fontsize=18)
    ax.set_title(titles[k], {'fontname': 'Times New Roman'}, fontsize=18),
    ax.set_ylabel("")
    ax.set_xlabel("")
plt.draw()
xmin, xmax = ax.get_xbound()
ymin, ymax = ax.get_ybound()
y2x_ratio = (ymax - ymin) / (xmax - xmin) * rows / cols
fig.set_figheight(wi * y2x_ratio)
gs.tight_layout(fig)
plt.savefig(out_dir + "FigS1_map_total_change.png")
print("all done!")