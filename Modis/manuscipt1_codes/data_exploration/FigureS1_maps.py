import xarray as xr
import matplotlib.pylab as plt


def outliers_index(data, m=2):
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


out_dir = ("/data/home/hamiddashti/nasa_above/outputs/data_analyses/Annual/"
           "Geographics/Figures_MS1/")

lst_mean = xr.open_dataarray("/data/ABOVE/Final_data/LST_Final/LST/Annual_Mean"
                             "/lst_mean_annual.nc")
albedo = xr.open_dataarray("/data/ABOVE/Final_data/ALBEDO_Final/"
                           "Annual_Albedo/Albedo_annual.nc")
albedo = albedo.rename({"x": "lon", "y": "lat"})
et = xr.open_dataarray(
    "/data/ABOVE/Final_data/ET_Final/Annual_ET/ET_Annual.nc")
et = et.rename({"x": "lon", "y": "lat"})
et = et.where(lst_mean.notnull())

dlst_total = lst_mean.loc[2013] - lst_mean.loc[2003]
dalbedo_total = albedo.loc[2013] - albedo.loc[2003]
det_total = et.loc[2013] - et.loc[2003]

I_dlst_total = outliers_index(dlst_total, 2)
I_dalbedo_total = outliers_index(dalbedo_total, 2)
I_det_total = outliers_index(det_total, 2)

dlst_total_clean = dlst_total.where((I_dlst_total == False)
                                    & (I_dalbedo_total == False)
                                    & (I_det_total == False))
dalbedo_total_clean = dalbedo_total.where((I_dlst_total == False)
                                          & (I_dalbedo_total == False)
                                          & (I_det_total == False))
det_total_clean = det_total.where((I_dlst_total == False)
                                  & (I_dalbedo_total == False)
                                  & (I_det_total == False))

titles = ["$\Delta LST$ [k]", "$\Delta Albedo$", "$\Delta ET$ [mm]"]
data = [dlst_total_clean, dalbedo_total_clean, det_total_clean]
plt.close()
fig, axes = plt.subplots(3, 1, figsize=(10, 8), facecolor='w', edgecolor='k')
axs = axes.ravel()
for i in range(3):
    data[i].plot(ax=axs[i], cmap="seismic", center=0)
    axs[i].set_title(titles[i], {'fontname': 'Times New Roman'})
    axs[i].set_ylabel("")
    axs[i].set_xlabel("")
plt.tight_layout()
plt.savefig(out_dir + "FigS1_map_total_change.png")
