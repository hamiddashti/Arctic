from sklearn.preprocessing import PowerTransformer
import pandas as pd
import statsmodels.api as sm
import xarray as xr
import matplotlib.pylab as plt
import numpy as np
np.seterr(divide="ignore")
import time
t1 = time.time()
from statsmodels.tools.tools import add_constant
from sklearn.decomposition import TruncatedSVD
import dask


def lsg(predictors, target):
    # least square solution
    a = np.linalg.inv(np.matmul(predictors.T, predictors))
    b = np.matmul(predictors.T, target)
    coefficients = np.matmul(a, b)
    return coefficients


def reject_outliers(data, m):
    # m is number of std
    import numpy as np

    data = data.astype(float)
    data[abs(data - np.nanmean(data)) > m * np.nanstd(data)] = np.nan
    return data


in_dir = "/data/ABOVE/Final_data/"
out_dir = ("/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/"
           "outputs/Sensitivity/EndPoints/Seasonal/")

luc = xr.open_dataarray(in_dir + "LUC/LULC_2003_2014.nc") / 100
luc["year"] = pd.to_datetime(luc.year.values, format="%Y")
luc = luc.rename({"x": "lon", "y": "lat"})

lst = xr.open_dataarray(in_dir + "LST_Final/LST/Seasonal_Mean/LST_Mean_JJA.nc")
lst = lst.loc["2003":"2014"] - 273.15

years = np.arange(2003, 2015)
year=years[0]
lst_tmp = lst.loc[str(year)].squeeze()
luc_tmp = luc.loc[str(year)].squeeze()

a_lst = lst_tmp.isel(lat=range(1400, 1600), lon=range(4400, 4600))
a_luc = luc_tmp.isel(lat=range(1400, 1600), lon=range(4400, 4600))

# win_size = 10
win_sizes = np.array([50])
win_size = win_sizes[0]
# for year in years:
# def yearly_calc(year, lst, luc, win_sizes):
# for win_size in win_sizes:
    # print(win_size)
    # lst_tmp = lst.loc[str(year)].squeeze()
    # luc_tmp = luc.loc[str(year)].squeeze()

    # a_lst = lst_tmp.isel(lat=range(1400, 1600), lon=range(4400, 4600))
    # a_luc = luc_tmp.isel(lat=range(1400, 1600), lon=range(4400, 4600))
    # a_lst = lst_tmp.isel(lat=range(1500, 1505), lon=range(4500, 4505))
    # a_luc = luc_tmp.isel(lat=range(1500, 1505), lon=range(4500, 4505))

b_lst = a_lst.rolling({
    "lat": win_size,
    "lon": win_size
}, center=True).construct({
    "lat": "lat_dim",
    "lon": "lon_dim"
})

b_luc = a_luc.rolling({
    "lat": win_size,
    "lon": win_size
}, center=True).construct({
    "lat": "lat_dim",
    "lon": "lon_dim"
})

shape = a_lst.shape
slopes = []
pvalues = []
bse = []
counter = 0

i=100;j=100
# for i in range(0, shape[0]):
    # for j in range(0, shape[1]):
def my_fun(i,j):
    # counter = counter + 1
    # p = np.round(100 * counter / (shape[0] * shape[0]), 2)
    # print(f"year: {year}; win size:{win_size}; precent complete: {p}")
    # print(f"{i},{j}")
    t1 = b_lst.isel(lat=i, lon=j).values.ravel()
    t2 = b_luc.isel(lat=i, lon=j).values

    if np.isnan(t2).any():
        c = np.zeros(8)
        c.fill(np.nan)
        # slopes_tmp = c
        # p_values_tmp = c
        slopes.append(c)
        pvalues.append(c)
        bse.append(c)
        continue

    t3 = t2.transpose()
    t4 = np.reshape(t3, (win_size * win_size, 7, 1)).squeeze()

    # Get rid of outliers
    lst_outliers = ~np.isnan(reject_outliers(t1, 2))
    t1 = t1[lst_outliers]
    t4 = t4[lst_outliers, :]
    # get rid of all zeros columns
    # t4 = t4[:, ~np.all((t4 == 0), axis=0)]
    # Coonvert it to float64 to avoid "overflow encountered in multiply"
    # warning
    t4 = np.float64(t4)
    pt = PowerTransformer()
    pt.fit(t4)
    t5 = pt.transform(t4)
    t5 = add_constant(t5)

    results = sm.OLS(t1, t5).fit()
    slopes_tmp = np.round(results.params, 5)
    pvalues_tmp = np.round(results.pvalues, 5)
    bse_tmp = np.round(results.bse, 5)
    return slopes_tmp, pvalues_tmp, bse_tmp, 


slopes.append(slopes_tmp)
pvalues.append(pvalues_tmp)
bse.append(bse_tmp)

a_slope = np.array(slopes)
b_slope = np.moveaxis(a_slope.reshape((shape[0], shape[1], 8)), -1, 0)
slopes_copy = a_luc.copy(data=b_slope[1:8, :, :])
# slopes_copy.to_netcdf(out_dir + "slopes_" + str(year) + "_win" +
#                       str(win_size) + ".nc")

a_pvalue = np.array(pvalues)
b_pvalue = np.moveaxis(a_pvalue.reshape((shape[0], shape[1], 8)), -1, 0)
pvalue_copy = a_luc.copy(data=b_pvalue[1:8, :, :])
# pvalue_copy.to_netcdf(out_dir + "pvalue_" + str(year) + "_win" +
#                       str(win_size) + ".nc")

    a_bse = np.array(bse)
    b_bse = np.moveaxis(a_bse.reshape((shape[0], shape[1], 8)), -1, 0)
    bse_copy = a_luc.copy(data=b_bse[1:8, :, :])
    # bse_copy.to_netcdf(out_dir + "bse_" + str(year) + "_win" +
    #                    str(win_size) + ".nc")

    ds = xr.merge([
        slopes_copy.to_dataset(name='dlst_dluc'),
        pvalue_copy.to_dataset(name='pvalue'),
        bse_copy.to_dataset(name='bse')
    ])

    outname = out_dir + 'dlst_dts_' + str(year) + "_win" + str(
        win_size) + ".nc"
    ds.to_netcdf(outname)

# yearly_calc(year=2003, lst=lst, luc=luc, win_sizes=win_sizes)
# dask.config.set(scheduler="processes")
# lazy_results = []
# for year in years:
#     lazy_result = dask.delayed(yearly_calc)(year, lst, luc, win_sizes)
#     lazy_results.append(lazy_result)

# from dask.diagnostics import ProgressBar
# with ProgressBar():
#     futures = dask.persist(*lazy_results)
#     results = dask.compute(*futures)
# dask.compute(lazy_results)
# ------------------------------------------------
