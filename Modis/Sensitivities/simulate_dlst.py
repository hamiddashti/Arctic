import xarray as xr
import numpy as np
import matplotlib.pylab as plt
import pandas as pd

in_dir = "/data/ABOVE/Final_data/"
out_dir = ("/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/"
           "outputs/Sensitivity/EndPoints/Seasonal/")

luc = xr.open_dataarray(in_dir + "LUC/LULC_2003_2014.nc") / 100
luc["year"] = pd.to_datetime(luc.year.values, format="%Y")
luc = luc.rename({"x": "lon", "y": "lat"})
lst = xr.open_dataarray(in_dir + "LST_Final/LST/Seasonal_Mean/LST_Mean_JJA.nc")
lst = lst.loc["2003":"2014"] - 273.15

lst_luc = xr.open_dataarray("/data/home/hamiddashti/mnt/nasa_above/working/"
"modis_analyses/outputs/Natural_Variability/Natural_Variability_Seasonal_Outputs"
"/EndPoints/JJA/delta_lst_changed_lulc_component_2013.nc")


a_luc = luc.isel(lat=range(1400, 1600), lon=range(4400, 4600))
a_lst = lst.isel(lat=range(1400, 1600), lon=range(4400, 4600))
a_lst_luc = lst_luc.isel(lat=range(1400, 1600), lon=range(4400, 4600))

luc_2003 = a_luc.loc[str(2003)].squeeze()
luc_2004 = a_luc.loc[str(2004)].squeeze()
lst_2003 = a_lst.loc[str(2003)].squeeze()
lst_2004 = a_lst.loc[str(2004)].squeeze()

luc_diff = luc_2004-luc_2003
lst_diff = lst_2004-lst_2003



ds_2003_win10 = xr.open_dataset(out_dir + 'dlst_dts_2003_win10.nc')
ds_2003_win50 = xr.open_dataset(out_dir + 'dlst_dts_2003_win50.nc')

ds_2003_win100 = xr.open_dataset(out_dir + 'dlst_dts_2003_win100.nc')
ds_2004_win100 = xr.open_dataset(out_dir + 'dlst_dts_2004_win100.nc')

dlst2003 = ds_2003_win100['dlst_dluc']
pval2003 = ds_2003_win100['pvalue']

dlst2004 = ds_2004_win100['dlst_dluc']
pval2004 = ds_2004_win100['pvalue']

i=101;j=101
a_lst_luc[i,j]
pval2003[:,i,j]
pval2004[:,i,j]

a = luc_diff[:,i,j]
b = lst_diff[i,j]

c1 = dlst2003[:,i,j]
c2 = dlst2004[:,i,j]

np.sum(a*c1)
np.sum(a*c2)
np.sum(a*(c1+c2)/2)










plt.savefig(out_dir+"tmp_fig/test.png")
plt.savefig(out_dir+"tmp_fig/win100.png")
plt.close()
