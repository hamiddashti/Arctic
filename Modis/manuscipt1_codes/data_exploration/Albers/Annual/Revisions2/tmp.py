""" ---------------------------------------------------------------------------
                             Annual data analyses

- This code produce the figures and data for the annual analyses. 

"""
"""----------------------------------------------------------------------------
Importing libraries used in this script
----------------------------------------------------------------------------"""
""" ---------------------------------------------------------------------------
defining functions used in this script
----------------------------------------------------------------------------"""




from logging import PercentStyle
from matplotlib.pyplot import savefig
from numpy.random import sample
import xarray as xr
import matplotlib.pylab as plt
import numpy as np
from matplotlib.pylab import savefig as save
import pandas as pd
from sklearn.linear_model import LinearRegression
import sklearn
import statsmodels.api as sm
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from statsmodels.stats.outliers_influence import summary_table
from xarray.core.duck_array_ops import count
def outliers_index(data, m=3.5):
    """
    Returns true if a value is outlier

    :param int data: numpy array
    :param int m: # of std to include data 
    """
    import numpy as np
    d = np.abs(data - np.nanmedian(data))
    mdev = np.nanmedian(d)
    s = d / mdev if mdev else 0.
    return ~(s < m)


def bootstrap(x, y, sample_weights, n, seed):
    """Bootstraping the weighted linear regression
        """
    np.random.seed(seed)
    # add a column of ones
    X = np.vstack([x, np.ones(len(x))]).T
    predicts = []
    params = []
    for j in range(0, n):
        sample_index = np.random.choice(range(0, len(y)), len(y))
        X_samples = X[sample_index]
        y_samples = y[sample_index]
        w_samples = sample_weights[sample_index]
        mod_wls = sm.WLS(y_samples, X_samples, weights=w_samples)
        res_wls = mod_wls.fit()
        predict = res_wls.predict(X)
        par = np.round(res_wls.params, 3)
        predicts.append(predict)
        params.append(par)
    predicts = np.array(predicts).T
    params = np.array(params).T
    return params, predicts


def binning(df, bins, var):
    """
    Fit OLS between variable of interest and DLCC

    :param dataframe df: pandas dataframe
    :param int bins: similar to bins argument in pandas.cut (bins intervals)
    :param str var: variable of interest   
    """
    # Binning data based on bins intervals
    df["bins"] = pd.cut(df["dlcc"], bins=bins, include_lowest=True)
    # group data based on bins and get the weighted mean of each bin
    bins_mean = df.groupby('bins').mean()

    # uncomment following linesto set the threshold on minimum number
    # of data in each bin since our bins are very small (0.001) we did not set
    counts = df.groupby('bins').count()["dlcc"]
    # bins_mean = bins_mean.where(counts >= 5)

    # Get rid of bins when there is zero data
    x = bins[1:][bins_mean[var].notnull()]
    y = bins_mean[var][bins_mean[var].notnull()].values
    counts = counts[bins_mean[var].notnull()]

    # X = sm.add_constant(x)
    # # Note it is weighted regression
    # mod_ols = sm.WLS(y, X, weights=counts)
    # res_ols = mod_ols.fit()
    # intercept, slope = np.round(res_ols.params, 3)

    # uncomment following linesto set the threshold on minimum number
    # of data in each bin since our bins are very small (0.001) we did not set
    # counts = df.groupby('bins').count()["dlcc"]
    # bins_mean = bins_mean.where(counts > 10)

    # intercept, slope = np.round(res_ols.params, 3)
    # intercept_bse, slope_bse = np.round(res_ols.bse, 3)
    # predicts = res_ols.predict(X)
    # pvalues = res_ols.pvalues
    # st, data, ss2 = summary_table(res_ols, alpha=0.05)
    # predict_mean_ci_low, predict_mean_ci_upp = data[:, 4:6].T
    out_list = [x, y, counts]
    return out_list


class ScalarFormatterForceFormat(ScalarFormatter):

    def _set_format(self):  # Override function that finds format to use.
        self.format = "%1.1f"  # Give format here


yfmt = ScalarFormatterForceFormat()
yfmt.set_powerlimits((0, 0))
N_M = 10000  # Number of bootstrap
in_dir = ("/data/home/hamiddashti/nasa_above/outputs/")
out_dir = ("/data/home/hamiddashti/nasa_above/outputs/data_analyses/Annual/"
           "Albers/Figures_MS1/")

# out_dir = (
#     "/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/test/")

# The map of dLST due to LCC
# dlst_lcc = xr.open_dataarray(
#     in_dir + ("Natural_Variability/Natural_Variability_Annual_outputs/Albers/"
#               "dlst_lcc.nc"))
# dlst_total = xr.open_dataarray(
#     in_dir + ("Natural_Variability/Natural_Variability_Annual_outputs/Albers/"
#               "dlst_total.nc"))

# This is the results of confusion table script
ct = xr.open_dataset(
    in_dir + "Sensitivity/EndPoints/Annual/Albers/Confusion_Table_Albers.nc")

dlst = ct["DLST_MEAN_LCC"]  # Changed LST due to LCC
# weights = ct["WEIGHTS"]  # Weights based on area
# weights = xr.Variable("ID", np.ones(len(dlst)))  # All weights are one
det = ct["DET_LCC"]  # Changed ET due to LCC
dalbedo = ct["DALBEDO_LCC"]  # Changed albedo due to LCC
dlcc = ct["DLCC"]  # Fractional change in land cover
normalized_confusion = ct["NORMALIZED_CONFUSION"]  # Normalized confusion
I_dlst = outliers_index(dlst, 3.5)  # Outlier indices for dLST
I_dalbedo = outliers_index(dalbedo, 3.5)  # Outlier indices for dalbedo
I_det = outliers_index(det, 3.5)  # Outlier indices for det

# Remove outliers based on indices
dlst_clean = dlst.where((I_dlst == False) & (I_dalbedo == False)
                        & (I_det == False))
dalbedo_clean = dalbedo.where((I_dlst == False) & (I_dalbedo == False)
                              & (I_det == False))
det_clean = det.where((I_dlst == False) & (I_dalbedo == False)
                      & (I_det == False))

# weights_clean = weights.where(I_dlst == False)
# dlcc_clean = dlcc.where((I_dlst == False) & (I_dalbedo == False)
#                         & (I_det == False))
# normalized_confusion_clean = normalized_confusion.where((I_dlst == False)
#                                                         & (I_dalbedo == False)
#                                                         & (I_det == False))

dlcc_clean = dlcc
normalized_confusion_clean = normalized_confusion

palette = sns.color_palette("tab10")  # Color palette for plots
lc_names = [
    "EF", "DF", "Shrub", "Herbaceous", "Sparse", "Barren", "Fen", "Bog",
    "Shallow_Littoral", "Water"
]


hot_lst = dlst_clean.where(dlst_clean>=1.5)
hot_lcc = normalized_confusion_clean.where(dlst_clean>=1.5)

n = len(hot_lcc)
result_count = np.zeros((10, 10), float)
for m in range(0, n):
    if xr.ufuncs.isnan(hot_lst.isel(ID=m)):
        continue
    a = hot_lcc.isel(ID=m).values
    np.fill_diagonal(a, 0)
    
    # new_list = []
    # for i, row in enumerate(a):
    #     for j, col in enumerate(row):
    #         new_list.append((col, (i, j)))

    # sorted_list = sorted(new_list, key=lambda x: x[0], reverse=True)
    # top = sorted_list[:3]
    # for k in top:
    #     result_count[tuple(k[1])] = result_count[tuple(k[1])]+1
    if np.any(a>0.02):
        a_max = a.max()
        if a_max<0.5:
            continue
        ind = np.unravel_index(np.argmax(a, axis=None), a.shape)
        # result[tuple(np.append(ind, i))] = a_max
        result_count[tuple(ind)] = result_count[ind]+1
        print(m)
result_prob = np.round(result_count*100/result_count.sum(), 2)
