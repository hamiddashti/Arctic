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
import matplotlib.gridspec as gridspec
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


def nv_vs_lcc(var_total, var_nv, var_lcc):
    # Select pixels we have variable_LCC, variable_total and variable_NV for them
    df = pd.DataFrame(data=None,
                      index=[
                          "\u0394VAR_NV_UP_LCC_UP", "\u0394VAR_NV_UP_LCC_Down",
                          "\u0394VAR_NV_Down_LCC_UP",
                          "\u0394VAR_NV_Down_LCC_Down"
                      ],
                      columns=[
                          "Mean_Total", "StD_Total", "Mean_NV", "StD_NV",
                          "Mean_LCC", "StD_LCC", "Area", "Percent"
                      ])
    dvar_total_tmp = var_total.where((var_total.notnull())
                                     & (var_nv.notnull())
                                     & (var_lcc.notnull()))
    dvar_nv_tmp = var_nv.where((var_total.notnull())
                               & (var_nv.notnull())
                               & (var_lcc.notnull()))
    dvar_lcc_tmp = var_lcc.where((var_total.notnull())
                                 & (var_nv.notnull())
                                 & (var_lcc.notnull()))

    # pixels where both nv and lcc goes up
    dvar_total_nvUp_lccUp = dvar_total_tmp.where((dvar_nv_tmp > 0)
                                                 & (dvar_lcc_tmp > 0))
    dvar_nv_nvUp_lccUp = dvar_nv_tmp.where((dvar_nv_tmp > 0)
                                           & (dvar_lcc_tmp > 0))
    dvar_lcc_nvUp_lccUp = dvar_lcc_tmp.where((dvar_nv_tmp > 0)
                                             & (dvar_lcc_tmp > 0))
    percent_dvar_nvUp_lccUp = (dvar_total_nvUp_lccUp.notnull().sum() /
                               dvar_total_tmp.notnull().sum()) * 100

    df.loc["\u0394VAR_NV_UP_LCC_UP",
           "Mean_Total"] = np.round(dvar_total_nvUp_lccUp.mean().values, 3)
    df.loc["\u0394VAR_NV_UP_LCC_UP",
           "StD_Total"] = np.round(dvar_total_nvUp_lccUp.std().values, 3)
    df.loc["\u0394VAR_NV_UP_LCC_UP",
           "Mean_NV"] = np.round(dvar_nv_nvUp_lccUp.mean().values, 3)
    df.loc["\u0394VAR_NV_UP_LCC_UP",
           "StD_NV"] = np.round(dvar_nv_nvUp_lccUp.std().values, 3)

    df.loc["\u0394VAR_NV_UP_LCC_UP",
           "Mean_LCC"] = np.round(dvar_lcc_nvUp_lccUp.mean().values, 3)
    df.loc["\u0394VAR_NV_UP_LCC_UP",
           "StD_LCC"] = np.round(dvar_lcc_nvUp_lccUp.std().values, 3)
    df.loc["\u0394VAR_NV_UP_LCC_UP",
           "Area"] = np.round(dvar_total_nvUp_lccUp.notnull().sum().values, 3)
    df.loc["\u0394VAR_NV_UP_LCC_UP",
           "Percent"] = np.round(percent_dvar_nvUp_lccUp.values, 3)

    # pixels where nv goes up but lcc goes down
    dvar_total_nvUp_lccDown = dvar_total_tmp.where((dvar_nv_tmp > 0)
                                                   & (dvar_lcc_tmp < 0))
    percent_dvar_nvUp_lccDown = (dvar_total_nvUp_lccDown.notnull().sum() /
                                 dvar_total_tmp.notnull().sum()) * 100
    dvar_nv_nvUp_lccDown = dvar_nv_tmp.where((dvar_nv_tmp > 0)
                                             & (dvar_lcc_tmp < 0))
    dvar_lcc_nvUp_lccDown = dvar_lcc_tmp.where((dvar_nv_tmp > 0)
                                               & (dvar_lcc_tmp < 0))

    df.loc["\u0394VAR_NV_UP_LCC_Down",
           "Mean_Total"] = np.round(dvar_total_nvUp_lccDown.mean().values, 3)
    df.loc["\u0394VAR_NV_UP_LCC_Down",
           "StD_Total"] = np.round(dvar_total_nvUp_lccDown.std().values, 3)

    df.loc["\u0394VAR_NV_UP_LCC_Down",
           "Mean_NV"] = np.round(dvar_nv_nvUp_lccDown.mean().values, 3)
    df.loc["\u0394VAR_NV_UP_LCC_Down",
           "StD_NV"] = np.round(dvar_nv_nvUp_lccDown.std().values, 3)

    df.loc["\u0394VAR_NV_UP_LCC_Down",
           "Mean_LCC"] = np.round(dvar_lcc_nvUp_lccDown.mean().values, 3)
    df.loc["\u0394VAR_NV_UP_LCC_Down",
           "StD_LCC"] = np.round(dvar_lcc_nvUp_lccDown.std().values, 3)
    df.loc["\u0394VAR_NV_UP_LCC_Down",
           "Area"] = np.round(dvar_total_nvUp_lccDown.notnull().sum().values,
                              3)
    df.loc["\u0394VAR_NV_UP_LCC_Down",
           "Area"] = np.round(percent_dvar_nvUp_lccDown.values, 3)

    df.loc["\u0394VAR_NV_UP_LCC_Down",
           "Percent"] = np.round(percent_dvar_nvUp_lccDown.values, 3)

    # Pixels where both nv and lcc goes down
    dvar_total_nvDown_lccDown = dvar_total_tmp.where((dvar_nv_tmp < 0)
                                                     & (dvar_lcc_tmp < 0))
    dvar_nv_nvDown_lccDown = dvar_nv_tmp.where((dvar_nv_tmp < 0)
                                               & (dvar_lcc_tmp < 0))
    dvar_lcc_nvDown_lccDown = dvar_lcc_tmp.where((dvar_nv_tmp < 0)
                                                 & (dvar_lcc_tmp < 0))
    percent_dvar_nvDown_lccDown = (dvar_total_nvDown_lccDown.notnull().sum() /
                                   dvar_total_tmp.notnull().sum()) * 100
    df.loc["\u0394VAR_NV_Down_LCC_Down",
           "Mean_Total"] = np.round(dvar_total_nvDown_lccDown.mean().values, 3)
    df.loc["\u0394VAR_NV_Down_LCC_Down",
           "StD_Total"] = np.round(dvar_total_nvDown_lccDown.std().values, 3)

    df.loc["\u0394VAR_NV_Down_LCC_Down",
           "Mean_NV"] = np.round(dvar_nv_nvDown_lccDown.mean().values, 3)
    df.loc["\u0394VAR_NV_Down_LCC_Down",
           "StD_NV"] = np.round(dvar_nv_nvDown_lccDown.std().values, 3)

    df.loc["\u0394VAR_NV_Down_LCC_Down",
           "Mean_LCC"] = np.round(dvar_lcc_nvDown_lccDown.mean().values, 3)
    df.loc["\u0394VAR_NV_Down_LCC_Down",
           "StD_LCC"] = np.round(dvar_lcc_nvDown_lccDown.std().values, 3)
    df.loc["\u0394VAR_NV_Down_LCC_Down",
           "Area"] = np.round(dvar_total_nvDown_lccDown.notnull().sum().values,
                              3)
    df.loc["\u0394VAR_NV_Down_LCC_Down",
           "Percent"] = np.round(percent_dvar_nvDown_lccDown.values, 3)

    # Pixels where nv goes down but lcc goes up
    dvar_total_nvDown_lccUp = dvar_total_tmp.where((dvar_nv_tmp < 0)
                                                   & (dvar_lcc_tmp > 0))
    dvar_lcc_nvDown_lccUp = dvar_lcc_tmp.where((dvar_nv_tmp < 0)
                                               & (dvar_lcc_tmp > 0))
    dvar_nv_nvDown_lccUp = dvar_nv_tmp.where((dvar_nv_tmp < 0)
                                             & (dvar_lcc_tmp > 0))
    percent_dvar_nvDown_lccUp = (dvar_total_nvDown_lccUp.notnull().sum() /
                                 dvar_total_tmp.notnull().sum()) * 100
    df.loc["\u0394VAR_NV_Down_LCC_UP",
           "Mean_Total"] = np.round(dvar_total_nvDown_lccUp.mean().values, 3)
    df.loc["\u0394VAR_NV_Down_LCC_UP",
           "StD_Total"] = np.round(dvar_total_nvDown_lccUp.std().values, 3)

    df.loc["\u0394VAR_NV_Down_LCC_UP",
           "Mean_NV"] = np.round(dvar_nv_nvDown_lccUp.mean().values, 3)
    df.loc["\u0394VAR_NV_Down_LCC_UP",
           "StD_NV"] = np.round(dvar_nv_nvDown_lccUp.std().values, 3)
    df.loc["\u0394VAR_NV_Down_LCC_UP",
           "Mean_LCC"] = np.round(dvar_lcc_nvDown_lccUp.mean().values, 3)
    df.loc["\u0394VAR_NV_Down_LCC_UP",
           "StD_LCC"] = np.round(dvar_lcc_nvDown_lccUp.std().values, 3)
    df.loc["\u0394VAR_NV_Down_LCC_UP",
           "Area"] = np.round(dvar_total_nvDown_lccUp.notnull().sum().values,
                              3)
    df.loc["\u0394VAR_NV_Down_LCC_UP",
           "Percent"] = np.round(percent_dvar_nvDown_lccUp.values, 3)
    return df


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
dlst_lcc = xr.open_dataarray(
    in_dir + ("Natural_Variability/Natural_Variability_Annual_outputs/Albers/"
              "dlst_lcc.nc"))
dlst_total_map = xr.open_dataarray(
    in_dir + ("Natural_Variability/Natural_Variability_Annual_outputs/Albers/"
              "dlst_total.nc"))
changed = xr.open_dataarray(
    "/data/home/hamiddashti/nasa_above/outputs/Natural_Variability/"
    "Natural_Variability_Annual_outputs/Albers/changed.nc")
# This is the results of confusion table script
ct = xr.open_dataset(
    in_dir + "Sensitivity/EndPoints/Annual/Albers/Confusion_Table_Albers.nc")
lst_mean = xr.open_dataset(
    "/data/ABOVE/Final_data/LST_Final/LST/Annual_Mean/albers/albers_proj_lst_mean_Annual.nc"
)
lst_mean = lst_mean["lst_mean_Annual"]

dlst_lcc = ct["DLST_MEAN_LCC"]  # Changed LST due to LCC
det_lcc = ct["DET_LCC"]  # Changed ET due to LCC
dalbedo_lcc = ct["DALBEDO_LCC"]  # Changed albedo due to LCC
dlst_nv = ct["DLST_MEAN_NV"]  # Changed LST due to NV
det_nv = ct["DET_NV"]  # Changed ET due to NV
dalbedo_nv = ct["DALBEDO_NV"]  # Changed albedo due to NV
dlst_total = ct["DLST_MEAN_TOTAL"]  # Changed LST due to LCC
det_total = ct["DET_TOTAL"]  # Changed ET due to LCC
dalbedo_total = ct["DALBEDO_TOTAL"]  # Changed albedo due to LCC
dlcc = ct["DLCC"]  # Fractional change in land cover
normalized_confusion = ct["NORMALIZED_CONFUSION"]  # Normalized confusion

I_dlst_lcc = outliers_index(dlst_lcc)  # Outlier indices for dLST
I_dalbedo_lcc = outliers_index(dalbedo_lcc)  # Outlier indices for dalbedo
I_det_lcc = outliers_index(det_lcc)  # Outlier indices for det
I_dlst_nv = outliers_index(dlst_nv)  # Outlier indices for dLST
I_dalbedo_nv = outliers_index(dalbedo_nv)  # Outlier indices for dalbedo
I_det_nv = outliers_index(det_nv)  # Outlier indices for det
# I_dlst_total = outliers_index(dlst_total)  # Outlier indices for dLST
# I_dalbedo_total = outliers_index(dalbedo_total,
#                                  )  # Outlier indices for dalbedo
# I_det_total = outliers_index(det_total)  # Outlier indices for det

dlst_lcc_clean = dlst_lcc.where((I_dlst_lcc == False)
                                & (I_dalbedo_lcc == False)
                                & (I_det_lcc == False))
dalbedo_lcc_clean = dalbedo_lcc.where((I_dlst_lcc == False)
                                      & (I_dalbedo_lcc == False)
                                      & (I_det_lcc == False))
det_lcc_clean = det_lcc.where((I_dlst_lcc == False) & (I_dalbedo_lcc == False)
                              & (I_det_lcc == False))
dlst_nv_clean = dlst_nv.where((I_dlst_nv == False) & (I_dalbedo_nv == False)
                              & (I_det_nv == False))
dalbedo_nv_clean = dalbedo_nv.where((I_dlst_nv == False)
                                    & (I_dalbedo_nv == False)
                                    & (I_det_nv == False))
det_nv_clean = det_nv.where((I_dlst_nv == False) & (I_dalbedo_nv == False)
                            & (I_det_nv == False))
# dlst_clean_total = dlst_total.where((I_dlst_total == False)
#                                     & (I_dalbedo_total == False)
#                                     & (I_det_total == False))
# dalbedo_clean_total = dalbedo_total.where((I_dlst_total == False)
#                                           & (I_dalbedo_total == False)
#                                           & (I_det_total == False))
# det_clean_total = det_total.where((I_dlst_total == False)
#                                   & (I_dalbedo_total == False)
#                                   & (I_det_total == False))
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

dlst_clean_total = dlst_nv_clean + dlst_lcc_clean
dalbedo_clean_total = dalbedo_nv_clean + dalbedo_lcc_clean
det_clean_total = det_nv_clean + det_lcc_clean

dlcc_clean = dlcc
normalized_confusion_clean = normalized_confusion

# ------------------------------------------------------------------------
#                              Some statistics
# ------------------------------------------------------------------------

# Get some stats on difference in LST, ET and albedo
df_stats = pd.DataFrame(data=None,
                        index=[
                            "\u0394LST_Total[K]", "\u0394LST_NV[K]",
                            "\u0394LST_LCC[K]", "\u0394Albedo_Total",
                            "\u0394Albedo_NV", "\u0394Albedo_LCC",
                            "\u0394ET_Total[mm]", "\u0394ET_NV[mm]",
                            "\u0394ET_LCC[mm]"
                        ],
                        columns=["Max", "Min", "Mean", "StD"])

df_stats.loc["\u0394LST_LCC[K]", "Max"] = np.round(dlst_lcc_clean.max().values,
                                                   1)
df_stats.loc["\u0394LST_LCC[K]", "Min"] = np.round(dlst_lcc_clean.min().values,
                                                   1)
df_stats.loc["\u0394LST_LCC[K]",
             "Mean"] = np.round(dlst_lcc_clean.mean().values, 1)
df_stats.loc["\u0394LST_LCC[K]", "StD"] = np.round(dlst_lcc_clean.std().values,
                                                   1)
df_stats.loc["\u0394LST_NV[K]", "Max"] = np.round(dlst_nv_clean.max().values,
                                                  1)
df_stats.loc["\u0394LST_NV[K]", "Min"] = np.round(dlst_nv_clean.min().values,
                                                  1)
df_stats.loc["\u0394LST_NV[K]", "Mean"] = np.round(dlst_nv_clean.mean().values,
                                                   1)
df_stats.loc["\u0394LST_NV[K]", "StD"] = np.round(dlst_nv_clean.std().values,
                                                  1)
df_stats.loc["\u0394LST_Total[K]",
             "Max"] = np.round(dlst_clean_total.max().values, 1)
df_stats.loc["\u0394LST_Total[K]",
             "Min"] = np.round(dlst_clean_total.min().values, 1)
df_stats.loc["\u0394LST_Total[K]",
             "Mean"] = np.round(dlst_clean_total.mean().values, 1)
df_stats.loc["\u0394LST_Total[K]",
             "StD"] = np.round(dlst_clean_total.std().values, 1)
df_stats.loc["\u0394Albedo_LCC",
             "Max"] = np.round(dalbedo_lcc_clean.max().values, 2)
df_stats.loc["\u0394Albedo_LCC",
             "Min"] = np.round(dalbedo_lcc_clean.min().values, 2)
df_stats.loc["\u0394Albedo_LCC",
             "Mean"] = np.round(dalbedo_lcc_clean.mean().values, 2)
df_stats.loc["\u0394Albedo_LCC",
             "StD"] = np.round(dalbedo_lcc_clean.std().values, 2)
df_stats.loc["\u0394Albedo_NV",
             "Max"] = np.round(dalbedo_nv_clean.max().values, 2)
df_stats.loc["\u0394Albedo_NV",
             "Min"] = np.round(dalbedo_nv_clean.min().values, 2)
df_stats.loc["\u0394Albedo_NV",
             "Mean"] = np.round(dalbedo_nv_clean.mean().values, 2)
df_stats.loc["\u0394Albedo_NV",
             "StD"] = np.round(dalbedo_nv_clean.std().values, 2)
df_stats.loc["\u0394Albedo_Total",
             "Max"] = np.round(dalbedo_clean_total.max().values, 2)
df_stats.loc["\u0394Albedo_Total",
             "Min"] = np.round(dalbedo_clean_total.min().values, 2)
df_stats.loc["\u0394Albedo_Total",
             "Mean"] = np.round(dalbedo_clean_total.mean().values, 2)
df_stats.loc["\u0394Albedo_Total",
             "StD"] = np.round(dalbedo_clean_total.std().values, 2)
df_stats.loc["\u0394ET_LCC[mm]", "Max"] = np.round(det_lcc_clean.max().values,
                                                   1)
df_stats.loc["\u0394ET_LCC[mm]", "Min"] = np.round(det_lcc_clean.min().values,
                                                   1)
df_stats.loc["\u0394ET_LCC[mm]",
             "Mean"] = np.round(det_lcc_clean.mean().values, 1)
df_stats.loc["\u0394ET_LCC[mm]", "StD"] = np.round(det_lcc_clean.std().values,
                                                   1)
df_stats.loc["\u0394ET_NV[mm]", "Max"] = np.round(det_nv_clean.max().values, 1)
df_stats.loc["\u0394ET_NV[mm]", "Min"] = np.round(det_nv_clean.min().values, 1)
df_stats.loc["\u0394ET_NV[mm]", "Mean"] = np.round(det_nv_clean.mean().values,
                                                   1)
df_stats.loc["\u0394ET_NV[mm]", "StD"] = np.round(det_nv_clean.std().values, 1)
df_stats.loc["\u0394ET_Total[mm]",
             "Max"] = np.round(det_clean_total.max().values, 1)
df_stats.loc["\u0394ET_Total[mm]",
             "Min"] = np.round(det_clean_total.min().values, 1)
df_stats.loc["\u0394ET_Total[mm]",
             "Mean"] = np.round(det_clean_total.mean().values, 1)
df_stats.loc["\u0394ET_Total[mm]",
             "StD"] = np.round(det_clean_total.std().values, 1)

# Change in increase/decrease in variables for different direction of
# variable_LCC and variable_NV
df_dlst = nv_vs_lcc(dlst_clean_total, dlst_nv_clean, dlst_lcc_clean)
df_dalbedo = nv_vs_lcc(dalbedo_clean_total, dalbedo_nv_clean,
                       dalbedo_lcc_clean)
df_det = nv_vs_lcc(det_clean_total, det_nv_clean, det_lcc_clean)
total_percent_changed = (changed.sum().values /
                         lst_mean.isel(year=0).notnull().sum().values) * 100

fname = "Annual_Stats.txt"
with open(out_dir + fname, "w") as f:
    f.write("This file contains the results of the annual data statistics\n")
    f.write("----------------------------------------------------------\n\n")
    f.write(f"Percent change: \
{np.round(total_percent_changed,3)}%\n\n")
    f.write(
        "\n-----------------------------------------------------------\n\n")
    f.write("Some stats:\n")
    f.write(df_stats.to_string(header=True, index=True))
    f.write(
        "\n-----------------------------------------------------------\n\n")
with open(out_dir + fname, "a") as f:
    f.write("Change in \u0394LST_LCC vs.\u0394LST_NV: \n")
    f.write(df_dlst.to_string(header=True, index=True))
    f.write(
        "\n-----------------------------------------------------------\n\n")
    f.write("Change in \u0394Albedo_LCC vs.\u0394Albedo_NV: \n")
    f.write(df_dalbedo.to_string(header=True, index=True))
    f.write(
        "\n-----------------------------------------------------------\n\n")
    f.write("Change in \u0394ET_LCC vs.\u0394ET_NV: \n")
    f.write(df_det.to_string(header=True, index=True))
    f.write(
        "\n-----------------------------------------------------------\n\n")
