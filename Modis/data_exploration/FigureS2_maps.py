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


in_dir = ("/data/home/hamiddashti/nasa_above/outputs/Natural_Variability/"
          "Natural_Variability_Annual_outputs/geographic/02_percent/")
out_dir = ("/data/home/hamiddashti/nasa_above/outputs/data_analyses/Annual/"
           "Geographics/Figures_MS1/")

dlst_mean_lcc = xr.open_dataarray(in_dir + "dlst_mean_lcc.nc")
dalbedo_lcc = xr.open_dataarray(in_dir + "albedo_lcc.nc")
dalbedo_lcc = dalbedo_lcc.rename({"x": "lon", "y": "lat"})
det_lcc = xr.open_dataarray(in_dir + "et_lcc.nc")
det_lcc = det_lcc.rename({"x": "lon", "y": "lat"})
dlst_mean_nv = xr.open_dataarray(in_dir + "dlst_mean_nv.nc")
dalbedo_nv = xr.open_dataarray(in_dir + "albedo_nv.nc")
dalbedo_nv = dalbedo_nv.rename({"x": "lon", "y": "lat"})
det_nv = xr.open_dataarray(in_dir + "et_nv.nc")
det_nv = det_nv.rename({"x": "lon", "y": "lat"})

I_dlst_lcc = outliers_index(dlst_mean_lcc, 2)
I_dalbedo_lcc = outliers_index(dalbedo_lcc, 2)
I_det_lcc = outliers_index(det_lcc, 2)
dlst_lcc_clean = dlst_mean_lcc.where((I_dlst_lcc == False)
                                     & (I_dalbedo_lcc == False)
                                     & (I_det_lcc == False))
dlst_lcc_clean.to_netcdf(out_dir + "dlst_lcc_clean.nc")
dalbedo_lcc_clean = dalbedo_lcc.where((I_dlst_lcc == False)
                                      & (I_dalbedo_lcc == False)
                                      & (I_det_lcc == False))
dalbedo_lcc_clean.to_netcdf(out_dir + "dalbedo_lcc_clean.nc")
det_lcc_clean = det_lcc.where((I_dlst_lcc == False) & (I_dalbedo_lcc == False)
                              & (I_det_lcc == False))

I_dlst_nv = outliers_index(dlst_mean_nv, 2)
I_dalbedo_nv = outliers_index(dalbedo_nv, 2)
I_det_nv = outliers_index(det_nv, 2)
dlst_nv_clean = dlst_mean_nv.where((I_dlst_nv == False)
                                   & (I_dalbedo_nv == False)
                                   & (I_det_nv == False))
dlst_nv_clean.to_netcdf("/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/test/dlst_nv_clean.nc")
dalbedo_nv_clean = dalbedo_nv.where((I_dlst_nv == False)
                                    & (I_dalbedo_nv == False)
                                    & (I_det_nv == False))
dalbedo_nv_clean.to_netcdf(out_dir + "dalbedo_nv_clean.nc")
det_nv_clean = det_nv.where((I_dlst_nv == False) & (I_dalbedo_nv == False)
                            & (I_det_nv == False))

data = [
    dlst_nv_clean, dlst_lcc_clean, dalbedo_nv_clean, dalbedo_lcc_clean,
    det_nv_clean, det_lcc_clean
]
titles = [
    "$\Delta LST_{NV}$ [K]", "$\Delta LST_{LCC}$ [K]", "$\Delta Albedo_{NV}$",
    "$\Delta Albedo_{LCC}$", "$\Delta ET_{NV}$ [mm]", "$\Delta ET_{LCC}$ [mm]"
]

plt.close()
fig, axes = plt.subplots(3, 2, figsize=(10, 8), facecolor='w', edgecolor='k')
axs = axes.ravel()
for i in range(6):
    data[i].plot(ax=axs[i], cmap="seismic", center=0)
    axs[i].set_title(titles[i])
    axs[i].set_ylabel("")
    axs[i].set_xlabel("")
plt.tight_layout()
plt.savefig(out_dir + "FigS2_map.png")
