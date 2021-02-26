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
year = years[0]
lst_tmp = lst.loc[str(year)].squeeze()
luc_tmp = luc.loc[str(year)].squeeze()

a_lst = lst_tmp.isel(lat=range(1000, 1500), lon=range(4000, 4500))
a_luc = luc_tmp.isel(lat=range(1000, 1500), lon=range(4000, 4500))

# a_lst = lst_tmp.isel(lat=range(1500, 1505), lon=range(4500, 4506))
# a_luc = luc_tmp.isel(lat=range(1500, 1505), lon=range(4500, 4506))

# win_size = 10
win_sizes = np.array([100])
win_size = win_sizes[0]
# for year in years:
# def yearly_calc(year, lst, luc, win_sizes):
# for win_size in win_sizes:
# print(win_size)
# lst_tmp = lst.loc[str(year)].squeeze()
# luc_tmp = luc.loc[str(year)].squeeze()

a_lst = lst_tmp.isel(lat=range(1400, 1600), lon=range(4400, 4600))
a_luc = luc_tmp.isel(lat=range(1400, 1600), lon=range(4400, 4600))
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

# b_lst = lst_tmp.rolling({
#     "lat": win_size,
#     "lon": win_size
# }, center=True).construct({
#     "lat": "lat_dim",
#     "lon": "lon_dim"
# })

# b_luc = luc_tmp.rolling({
#     "lat": win_size,
#     "lon": win_size
# }, center=True).construct({
#     "lat": "lat_dim",
#     "lon": "lon_dim"
# })

shape = a_lst.shape

# counter = 0


# for i in range(0, shape[0]):
# for j in range(0, shape[1]):
def my_fun(i, j):
    # counter = counter + 1
    # p = np.round(100 * counter / (shape[0] * shape[0]), 2)
    # print(f"year: {year}; win size:{win_size}; precent complete: {p}")
    print(f"{i},{j}")

    t1 = b_lst.isel(lat=i, lon=j).values.ravel()
    t2 = b_luc.isel(lat=i, lon=j).values

    if np.isnan(t2).any():
        c = np.zeros(8)
        c.fill(np.nan)
        slopes_tmp = c
        pvalues_tmp = c
        bse_tmp = c
        #  slopes.append(c)
        # pvalues.append(c)
        # bse.append(c)
        # continue
        return slopes_tmp, pvalues_tmp, bse_tmp

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
    return slopes_tmp, pvalues_tmp, bse_tmp


delayed_results = []
for i in range(0, shape[0]):
    for j in range(0, shape[1]):
        slope_cals = dask.delayed(my_fun)(i, j)
        delayed_results.append(slope_cals)

results = dask.compute(*delayed_results)

results_array = np.array(results)
slopes = results_array[:, 0, :]
slopes = np.moveaxis(slopes.reshape((shape[0], shape[1], 8)), -1, 0)

pvalues = results_array[:, 1, :]
pvalues = np.moveaxis(pvalues.reshape((shape[0], shape[1], 8)), -1, 0)

bse = results_array[:, 2, :]
bse = np.moveaxis(bse.reshape((shape[0], shape[1], 8)), -1, 0)

# my_fun(100,100)[0]
# slopes[:,100,100]
ds = xr.Dataset(
    {
        "slope": (["coefs", "lat", "lon"], slopes),
        "pvalue": (["coefs", "lat", "lon"], pvalues),
        "bse": (["coefs", "lat", "lon"], bse)
    },
    coords={
        "lon": a_luc['lon'],
        "lat": a_luc['lat'],
        "time": a_luc['year']
    })
ds.to_netcdf(out_dir + "test_results100.nc")
# ---------------------------------------------------------------


