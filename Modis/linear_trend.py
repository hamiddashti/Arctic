import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

in_dir = "F:\\working\\LUC\\"
out_dir = "F:\\working\\LUC\\outputs\\"
fig_dir = out_dir + "Figures\\"

lst = xr.open_dataarray(in_dir + "Annual_LST/lst_mean_annual.nc") - 273.15
lst = lst.sel(year=slice("2003", "2014"))
lst = lst.rename({"lat": "y", "lon": "x","year":"time"})

albedo = xr.open_dataarray(in_dir + "Annual_Albedo/Albedo_annual.nc")
albedo = albedo.sel(year=slice("2003", "2014"))
albedo = albedo.rename({'year':'time'})
albedo = albedo.drop_vars('time')

ET = xr.open_dataarray(in_dir + "Annual_ET/ET.nc")
ET = ET.rename({'year':'time'})
ET = ET.drop_vars('time')

t= pd.date_range(start="2003",end="2015",freq="A")
lst['time']=t
albedo['time']=t
ET['time']=t

def dataset_encoding(xds):
    import pandas as pd
    cols = ['source', 'original_shape', 'dtype', 'zlib', 'complevel', 'chunksizes']
    info = pd.DataFrame(columns=cols, index=xds.data_vars)
    for row in info.index:
        var_encoding = xds[row].encoding
        for col in info.keys():
            info.ix[row, col] = var_encoding.pop(col, '')
    
    return info


def xarray_trend(xarr,var_unit):    
    from scipy import stats
    import numpy as np
    # getting shapes
    
    m = np.prod(xarr.shape[1:]).squeeze()
    n = xarr.shape[0]
    
    # creating x and y variables for linear regression
    x = xarr.time.to_pandas().index.to_julian_date().values[:, None]
    y = xarr.to_masked_array().reshape(n, -1)
    
    # ############################ #
    # LINEAR REGRESSION DONE BELOW #
    xm = x.mean(0)  # mean
    ym = y.mean(0)  # mean
    ya = y - ym  # anomaly
    xa = x - xm  # anomaly
    
    # variance and covariances
    xss = (xa ** 2).sum(0) / (n - 1)  # variance of x (with df as n-1)
    yss = (ya ** 2).sum(0) / (n - 1)  # variance of y (with df as n-1)
    xys = (xa * ya).sum(0) / (n - 1)  # covariance (with df as n-1)
    # slope and intercept
    slope = xys / xss
    intercept = ym - (slope * xm)
    # statistics about fit
    df = n - 2
    r = xys / (xss * yss)**0.5
    t = r * (df / ((1 - r) * (1 + r)))**0.5
    p = stats.distributions.t.sf(abs(t), df)
    
    # misclaneous additional functions
    # yhat = dot(x, slope[None]) + intercept
    # sse = ((yhat - y)**2).sum(0) / (n - 2)  # n-2 is df
    # se = ((1 - r**2) * yss / xss / df)**0.5
    
    # preparing outputs
    out = xarr[:2].mean('time')
    # first create variable for slope and adjust meta
    xarr_slope = out.copy()
    xarr_slope.name = '_slope'
    xarr_slope.attrs['units'] = var_unit
    xarr_slope.values = slope.reshape(xarr.shape[1:])
    # do the same for the p value
    xarr_p = out.copy()
    xarr_p.name = '_Pvalue'
    xarr_p.attrs['info'] = "If p < 0.05 then the results from 'slope' are significant."
    xarr_p.values = p.reshape(xarr.shape[1:])
    # join these variables
    xarr_out = xarr_slope.to_dataset(name='slope')
    xarr_out['pval'] = xarr_p

    return xarr_out



lst_trend= xarray_trend(lst,var_unit="temp [c] / year")
lst_slope = lst_trend['slope']
lst_pval = lst_trend['pval']
lst_sig_slope = lst_slope.where(lst_pval<0.05)
lst_sig_slope.plot()
plt.savefig(fig_dir+'LST_Significant_trend.png')
plt.close()

albedo_trend= xarray_trend(albedo,var_unit="Fraction / year")
albedo_slope = albedo_trend['slope']
albedo_pval = albedo_trend['pval']
albedo_sig_slope = albedo_slope.where(albedo_pval<0.05)
albedo_sig_slope.plot()
plt.title("")
plt.savefig(fig_dir+'Albedo_Significant_trend.png')
plt.close()

et_trend= xarray_trend(ET,var_unit="mm / year")
et_slope = et_trend['slope']
et_pval = et_trend['pval']
et_sig_slope = et_slope.where(et_pval<0.05)
et_sig_slope.plot()
plt.title("")
plt.savefig(fig_dir+'ET_Significant_trend.png')
plt.close()
