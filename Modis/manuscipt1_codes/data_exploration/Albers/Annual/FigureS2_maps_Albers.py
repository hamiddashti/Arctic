import xarray as xr
import matplotlib.pylab as plt
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from xarray.core.duck_array_ops import notnull


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


in_dir = ("/data/home/hamiddashti/nasa_above/outputs/Natural_Variability/"
          "Natural_Variability_Annual_outputs/Albers/")
out_dir = ("/data/home/hamiddashti/nasa_above/outputs/data_analyses/Annual/"
           "Albers/Figures_MS1/")

# out_dir = "/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/test/"

dlst_mean_total = xr.open_dataarray(in_dir + "dlst_total.nc", decode_cf="all")
dalbedo_total = xr.open_dataarray(in_dir + "dalbedo_total.nc")
det_total = xr.open_dataarray(in_dir + "det_total.nc")
det_total = det_total.assign_coords({
    "lat": dlst_mean_total.lat,
    "lon": dlst_mean_total.lon
})
dalbedo_total = dalbedo_total.assign_coords({
    "lat": dlst_mean_total.lat,
    "lon": dlst_mean_total.lon
})

dlst_mean_lcc = xr.open_dataarray(in_dir + "dlst_lcc.nc")
dalbedo_lcc = xr.open_dataarray(in_dir + "dalbedo_lcc.nc")
det_lcc = xr.open_dataarray(in_dir + "det_lcc.nc")

dlst_mean_nv = xr.open_dataarray(in_dir + "dlst_nv.nc")
dalbedo_nv = xr.open_dataarray(in_dir + "dalbedo_nv.nc")
det_nv = xr.open_dataarray(in_dir + "det_nv.nc")

# I_dlst_total = outliers_index(dlst_mean_total)
# I_dalbedo_total = outliers_index(dalbedo_total)
# I_det_total = outliers_index(det_total)

I_dlst_lcc = outliers_index(dlst_mean_lcc)
I_dalbedo_lcc = outliers_index(dalbedo_lcc)
I_det_lcc = outliers_index(det_lcc)

I_dlst_nv = outliers_index(dlst_mean_nv)
I_dalbedo_nv = outliers_index(dalbedo_nv)
I_det_nv = outliers_index(det_nv)

# dlst_total_clean = dlst_mean_total.where((I_dlst_total == False)
#                                          & (I_dalbedo_total == False)
#                                          & (I_det_total == False))
# dalbedo_total_clean = dalbedo_total.where((I_dlst_total == False)
#                                           & (I_dalbedo_total == False)
#                                           & (I_det_total == False))
# det_total_clean = det_total.where((I_dlst_total == False)
#                                   & (I_dalbedo_total == False)
#                                   & (I_det_total == False))
dlst_lcc_clean = dlst_mean_lcc.where((I_dlst_lcc == False)
                                     & (I_dalbedo_lcc == False)
                                     & (I_det_lcc == False))
dalbedo_lcc_clean = dalbedo_lcc.where((I_dlst_lcc == False)
                                      & (I_dalbedo_lcc == False)
                                      & (I_det_lcc == False))
det_lcc_clean = det_lcc.where((I_dlst_lcc == False) & (I_dalbedo_lcc == False)
                              & (I_det_lcc == False))

dlst_nv_clean = dlst_mean_nv.where((I_dlst_nv == False)
                                   & (I_dalbedo_nv == False)
                                   & (I_det_nv == False))

dalbedo_nv_clean = dalbedo_nv.where((I_dlst_nv == False)
                                    & (I_dalbedo_nv == False)
                                    & (I_det_nv == False))

det_nv_clean = det_nv.where((I_dlst_nv == False) & (I_dalbedo_nv == False)
                            & (I_det_nv == False))

dlst_lcc_clean = dlst_lcc_clean.where((dlst_nv_clean.notnull())
                                      & (dlst_lcc_clean.notnull()))
dlst_nv_clean = dlst_nv_clean.where((dlst_nv_clean.notnull())
                                    & (dlst_lcc_clean.notnull()))

dalbedo_lcc_clean = dalbedo_lcc_clean.where((dalbedo_nv_clean.notnull())
                                            & (dalbedo_lcc_clean.notnull()))
dalbedo_nv_clean = dalbedo_nv_clean.where((dalbedo_nv_clean.notnull())
                                          & (dalbedo_lcc_clean.notnull()))
det_lcc_clean = det_lcc_clean.where((det_nv_clean.notnull())
                                    & (det_lcc_clean.notnull()))
det_nv_clean = det_nv_clean.where((det_nv_clean.notnull())
                                  & (det_lcc_clean.notnull()))

dlst_total_clean = dlst_nv_clean + dlst_lcc_clean
dalbedo_total_clean = dalbedo_nv_clean + dalbedo_lcc_clean
det_total_clean = det_nv_clean + det_lcc_clean

data = [
    dlst_total_clean, dlst_nv_clean, dlst_lcc_clean, dalbedo_total_clean,
    dalbedo_nv_clean, dalbedo_lcc_clean, det_total_clean, det_nv_clean,
    det_lcc_clean
]
titles = [
    "$\Delta LST_{Total}$", "$\Delta LST_{NV}$", "$\Delta LST_{LCC}$",
    "$\Delta Albedo_{Total}$", "$\Delta Albedo_{NV}$", "$\Delta Albedo_{LCC}$",
    "$\Delta ET_{Total}$", "$\Delta ET_{NV}$", "$\Delta ET_{LCC}$"
]
labels = ["[K]", "[K]", "[K]", "", "", "", "[mm]", "[mm]", "[mm]"]

ylim = [1685818.55149696, 4800000.0]
xlim = [-3300000.0, 0]

plt.close()
ar = 1.0  # initial aspect ratio for first trial
wi = 18  # width of the whole figure in inches, ...
# set it wide enough to cover all columns of sub-plots
hi = wi * ar  # height in inches
# set number of rows/columns
rows, cols = 3, 3
gs = gridspec.GridSpec(rows, cols)
fig = plt.figure(figsize=(wi, hi))
for k in range(0, rows * cols):
    ax = plt.subplot(gs[k], projection=ccrs.AlbersEqualArea())
    im = data[k].plot.imshow(ax=ax,
                             ylim=ylim,
                             xlim=xlim,
                             add_colorbar=False,
                             cmap="seismic")
    cb = plt.colorbar(im, fraction=0.046, pad=0.02, shrink=0.80)
    cb.set_label(label=labels[k], fontsize=18)
    cb.ax.tick_params(labelsize=16)
    ax.set_title(titles[k], {'fontname': 'Times New Roman'}, fontsize=22),
    ax.set_ylabel("")
    ax.set_xlabel("")
plt.draw()
xmin, xmax = ax.get_xbound()
ymin, ymax = ax.get_ybound()
y2x_ratio = (ymax - ymin) / (xmax - xmin) * rows / cols
fig.set_figheight(wi * y2x_ratio)
gs.tight_layout(fig)
plt.savefig(out_dir + "FigS2_map_total_change.png")
print("all done!")
