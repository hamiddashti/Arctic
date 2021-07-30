import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

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
# out_dir = ("/data/home/hamiddashti/nasa_above/outputs/data_analyses/Annual/"
#            "Albers/Figures_MS1/")
out_dir = "/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/test/"

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
dlcc = xr.open_dataarray(in_dir + "dlcc.nc") * 100
dlcc = dlcc.where(~dlcc.isel(band=7).notnull())
dlcc = dlcc.where(~dlcc.isel(band=8).notnull())
dlcc = dlcc.where(~dlcc.isel(band=9).notnull())

plt.close()
ar = 1.0  # initial aspect ratio for first trial
wi = 18  # width of the whole figure in inches, ...
# set it wide enough to cover all columns of sub-plots
hi = wi * ar  # height in inches
# set number of rows/columns
rows, cols = 7, 1
gs = gridspec.GridSpec(rows, cols)
fig = plt.figure(figsize=(wi, hi))
for k in range(0, rows * cols):
    ax = plt.subplot(gs[k, 0])
    im = dlcc.isel(band=k).plot.hist()

plt.draw()
xmin, xmax = ax.get_xbound()
ymin, ymax = ax.get_ybound()
y2x_ratio = (ymax - ymin) / (xmax - xmin) * rows / cols
fig.set_figheight(wi * y2x_ratio)
gs.tight_layout(fig)
plt.savefig(out_dir + "test.png")
print("all done!")

a = dlcc_hot.max(dim="band")
plt.close()
dlcc_hot.isel(band=2).plot.hist()
plt.savefig(out_dir + "test2.png")

dlst_mean_lcc = xr.open_dataarray(
    in_dir + "Natural_Variability/"
    "Natural_Variability_Annual_outputs/Albers/dlst_lcc.nc")
ct = xr.open_dataarray(
    in_dir +
    "Sensitivity/EndPoints/Annual/Albers/confusion_tables_all_pixels2003_2013.nc"
)
dlcc = xr.open_dataarray(
    in_dir +
    "Natural_Variability/Natural_Variability_Annual_outputs/Albers/dlcc.nc"
) * 100

I_dlst_lcc = outliers_index(dlst_mean_lcc)
dlst_lcc_clean = dlst_mean_lcc.where(I_dlst_lcc == False)
dlst_hot = dlst_lcc_clean.where(dlst_lcc_clean > 2.5)
# ct_hot = ct.where(dlst_hot.notnull())
dlcc_hot = abs(dlcc.where(dlst_lcc_clean > 2.5))

a = dlcc_hot.where(dlcc_hot < 20)
a = dlcc_hot.argmax(dim="band")

a = dlcc.sum(dim="band")

plt.close()
a.plot.hist()
plt.savefig(out_dir + "test.png")

dlcc.isel(lat=2160, lon=1455)
dlst_hot.isel(lat=2160, lon=1455)

ct_hot_stack = ct_hot.stack(z=("lon", "lat"))
ct_hot_stack = ct_hot_stack.isel(
    z=np.arange(0, dlst_hot.shape[0] * dlst_hot.shape[1]))

ct_hot_stack_final = ct_hot_stack.dropna(dim="z")
ct_hot_stack_final = ct_hot_stack_final.where(ct_hot_stack_final > 0.02)

lst_sens = np.array([[0, 1.025, -1.016, -0.97, 0.397, 1.179, -0.919],
                     [-1.025, 0, -0.932, -0.519, 3.703, 5.566, 1.91],
                     [1.016, 0.932, 0, -0.171, 2.249, 4.378, 1.009],
                     [0.97, 0.519, 0.171, 0, 1.075, 3.186, 1.169],
                     [-0.397, -3.703, -2.249, -1.075, 0, 1.644, -2.902],
                     [-1.179, -5.566, -4.378, -3.186, -1.644, 0, -5.977],
                     [0.919, -1.91, -1.009, -1.169, 2.902, 5.977, 0]])

plt.close()
ar = 1.0  # initial aspect ratio for first trial
wi = 13  # width of the whole figure in inches, ...
hi = wi * ar
rows, cols = 7, 7
gs = gridspec.GridSpec(rows, cols)
fig = plt.figure(figsize=(wi, hi))
for i in range(7):
    for j in range(7):
        if i == j:
            continue
        data = ct_hot_stack_final[i, j, :]
        filtered_data = data[~np.isnan(data)]
        ax = plt.subplot(gs[i, j])
        ax.set_ylim(0, 1)
        n = np.round((len(filtered_data) / len(data)) * 100, 3)
        ax.text(0.75, 0.8, n)
        im = ax.boxplot(filtered_data)
plt.savefig(out_dir + "test.png")

plt.close()
ar = 1.0  # initial aspect ratio for first trial
wi = 13  # width of the whole figure in inches, ...
hi = wi * ar
rows, cols = 7, 7
gs = gridspec.GridSpec(rows, cols)
fig = plt.figure(figsize=(wi, hi))
for k in range(0, rows * cols):
    ax = plt.subplot(gs[k])
    im = ax.imshow(data_slope[k], cmap="bwr")
    ax.set_title(titles[k])
    ax.set_xticks(np.arange(len(final_state)))
    ax.set_yticks(np.arange(len(initial_state)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(final_state)
    ax.set_yticklabels(initial_state)
    plt.setp(ax.get_xticklabels(),
             rotation=45,
             ha="right",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    for i in range(len(initial_state)):
        for j in range(len(final_state)):
            if i > j:
                continue
            text = ax.text(j,
                           i,
                           str(data_slope[k][i, j]) + "\n(\u00B1" +
                           str(data_std[k][i, j]) + ")",
                           ha="center",
                           va="center",
                           color="black",
                           weight="heavy",
                           fontsize=7.5)

plt.draw()
xmin, xmax = ax.get_xbound()
ymin, ymax = ax.get_ybound()
y2x_ratio = (ymax - ymin) / (xmax - xmin) * rows / cols
fig.set_figheight(wi * y2x_ratio)
gs.tight_layout(fig)
plt.savefig(out_dir + "test.png")
print("all done!")
