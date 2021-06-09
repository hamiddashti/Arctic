""" ---------------------------------------------------------------------------
                             Annual data analyses

- This code produce the figures and data for the annual analyses. 

"""
"""----------------------------------------------------------------------------
Importing libraries used in this script
----------------------------------------------------------------------------"""
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
""" ---------------------------------------------------------------------------
defining functions used in this script
----------------------------------------------------------------------------"""


def outliers_index(data, m=3):
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


def binmean(data, var, weight_col):
    """
    Calculates the weighted mean of the bins. 
    
    The weights are based on area of each pixel. 

    :param int data: pandas group objects (refer to fit_bins_ols function)
    :param str var: name of the variable
    :param str weight_col: name of the column which contains weights  
    """
    # Weighted mean
    mean = np.nansum(data[var] * data[weight_col]) / np.nansum(
        data[weight_col])
    return mean


def fit_bins_ols(df, bins, var):
    """
    Fit OLS between variable of interest and DLCC

    :param dataframe df: pandas dataframe
    :param int bins: similar to bins argument in pandas.cut (bins intervals)
    :param str var: variable of interest   
    """
    # Binning data based on bins intervals
    df["bins"] = pd.cut(df["dlcc"], bins=bins, include_lowest=True)
    # group data based on bins and get the weighted mean of each bin
    bins_mean = df.groupby('bins').apply(binmean, var, 'w')

    # uncomment following linesto set the threshold on minimum number
    # of data in each bin since our bins are very small (0.001) we did not set
    # counts = df.groupby('bins').count()["dlcc"]
    # bins_mean = bins_mean.where(counts > 10)

    # When taking the regression of weighted means, the new weights are the
    # sum of all weights in each bin
    wsums = df.groupby("bins").apply(lambda d: d["w"].sum())

    # Get rid of bins when there is zero data
    x = bins[1:][bins_mean.notnull()]
    y = bins_mean[bins_mean.notnull()].values
    sample_weight = wsums[bins_mean.notnull()].values
    X = sm.add_constant(x)
    # Note it is weighted regression
    mod_wls = sm.WLS(y, X, weights=sample_weight)
    res_wls = mod_wls.fit()
    intercept, slope = np.round(res_wls.params, 3)
    intercept_bse, slope_bse = np.round(res_wls.bse, 3)
    predicts = res_wls.predict(X)
    pvalues = res_wls.pvalues
    out_list = [
        x, y, sample_weight, intercept, intercept_bse, slope, slope_bse,
        predicts, pvalues
    ]
    return out_list


class ScalarFormatterForceFormat(ScalarFormatter):
    def _set_format(self):  # Override function that finds format to use.
        self.format = "%1.1f"  # Give format here


yfmt = ScalarFormatterForceFormat()
yfmt.set_powerlimits((0, 0))

in_dir = ("/data/home/hamiddashti/nasa_above/outputs/")
out_dir = ("/data/home/hamiddashti/nasa_above/outputs/data_analyses/Annual/"
           "Geographics/Figures_MS1/")
lst_dir = "/data/ABOVE/Final_data/"
# out_dir = (
#     "/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/test/")

# The map of dLST due to LCC

lst_mean = xr.open_dataarray(lst_dir +
                             "LST_Final/LST/Annual_Mean/lst_mean_annual.nc")
dlst_lcc = xr.open_dataarray(
    in_dir + ("Natural_Variability/Natural_Variability_Annual_outputs/"
              "geographic/dlst_mean_lcc.nc"))
dlst_lcc = xr.open_dataarray(
    in_dir + ("Natural_Variability/Natural_Variability_Annual_outputs/"
              "geographic/dlst_mean_nv.nc"))
dlst_total = xr.open_dataarray(
    in_dir + ("Natural_Variability/Natural_Variability_Annual_outputs/"
              "geographic/02_percent/dlst_mean_changed.nc"))

# This is the results of confusion table script0
# This is the file with 2 percent change as threshold for calculating natural
# variability.
ct = xr.open_dataset(in_dir +
                     ("Sensitivity/EndPoints/Annual/Geographic/"
                      "02_percent/Confusion_Table_final_02precent.nc"))

# ct = xr.open_dataset(
#     in_dir +
#     "Sensitivity/EndPoints/Annual/Geographic/Confusion_Table_final.nc")
# ct = xr.open_dataset(in_dir +
#                      ("Sensitivity/EndPoints/Annual/Geographic/"
#                       "Confusion_Table_final_LC_Added_01precent_withnan2.nc"))

# ct2 = xr.open_dataset(
#     in_dir +
#     "Sensitivity/EndPoints/Annual/Geographic/Confusion_Table_final_LC_Added_01precent_withnan.nc"
# )

# ct = xr.open_dataset(
#     in_dir + "Sensitivity/EndPoints/Annual/Albers/Confusion_Table_Albers.nc")

weights = ct["WEIGHTS"]  # Weights based on area

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

I_dlst_lcc = outliers_index(dlst_lcc, 2)  # Outlier indices for dLST
I_dalbedo_lcc = outliers_index(dalbedo_lcc, 2)  # Outlier indices for dalbedo
I_det_lcc = outliers_index(det_lcc, 2)  # Outlier indices for det
I_dlst_nv = outliers_index(dlst_nv, 2)  # Outlier indices for dLST
I_dalbedo_nv = outliers_index(dalbedo_nv, 2)  # Outlier indices for dalbedo
I_det_nv = outliers_index(det_nv, 2)  # Outlier indices for det
I_dlst_total = outliers_index(dlst_total, 2)  # Outlier indices for dLST
I_dalbedo_total = outliers_index(dalbedo_total,
                                 2)  # Outlier indices for dalbedo
I_det_total = outliers_index(det_total, 2)  # Outlier indices for det

# Remove outliers based on indices
dlst_clean_lcc = dlst_lcc.where((I_dlst_lcc == False)
                                & (I_dalbedo_lcc == False)
                                & (I_det_lcc == False))
dalbedo_clean_lcc = dalbedo_lcc.where((I_dlst_lcc == False)
                                      & (I_dalbedo_lcc == False)
                                      & (I_det_lcc == False))
det_clean_lcc = det_lcc.where((I_dlst_lcc == False) & (I_dalbedo_lcc == False)
                              & (I_det_lcc == False))
dlst_clean_nv = dlst_nv.where((I_dlst_nv == False) & (I_dalbedo_nv == False)
                              & (I_det_nv == False))
dalbedo_clean_nv = dalbedo_nv.where((I_dlst_nv == False)
                                    & (I_dalbedo_nv == False)
                                    & (I_det_nv == False))
det_clean_nv = det_nv.where((I_dlst_nv == False) & (I_dalbedo_nv == False)
                            & (I_det_nv == False))
dlst_clean_total = dlst_total.where((I_dlst_total == False)
                                    & (I_dalbedo_total == False)
                                    & (I_det_total == False))
dalbedo_clean_total = dalbedo_total.where((I_dlst_total == False)
                                          & (I_dalbedo_total == False)
                                          & (I_det_total == False))
det_clean_total = det_total.where((I_dlst_total == False)
                                  & (I_dalbedo_total == False)
                                  & (I_det_total == False))

weights_clean = weights.where((I_dlst_lcc == False) & (I_dalbedo_lcc == False)
                              & (I_det_lcc == False))
# dlcc_clean = dlcc.where(I_dlst == False)
# normalized_confusion_clean = normalized_confusion.where(I_dlst == False)
dlcc_clean = dlcc
normalized_confusion_clean = normalized_confusion

palette = sns.color_palette("tab10")  # Color palette for plots
lc_names = [
    "EF", "DF", "Shrub", "Herbaceous", "Sparse", "Barren", "Fen", "Bog",
    "Shallow_Littoral", "Water"
]

# Calculate some statistics (max, min, std, percent change).
percent_changed = (dlst_lcc.notnull().sum() /
                   lst_mean.isel(year=0).notnull().sum()) * 100
df_stats = pd.DataFrame(data=None,
                        index=[
                            "\u0394LST_LCC[K]", "\u0394Albedo_LCC",
                            "\u0394ET_LCC[mm]", "\u0394LST_NV[K]",
                            "\u0394Albedo_NV", "\u0394ET_NV[mm]",
                            "\u0394LST_Total[K]", "\u0394Albedo_Total",
                            "\u0394ET_Total[mm]"
                        ],
                        columns=["Max", "Min", "StD"])
df_stats.loc["\u0394LST_LCC[K]", "Max"] = np.round(dlst_clean_lcc.max().values,
                                                   3)
df_stats.loc["\u0394LST_LCC[K]", "Min"] = np.round(dlst_clean_lcc.min().values,
                                                   3)
df_stats.loc["\u0394LST_LCC[K]", "StD"] = np.round(dlst_clean_lcc.std().values,
                                                   3)
df_stats.loc["\u0394Albedo_LCC",
             "Max"] = np.round(dalbedo_clean_lcc.max().values, 3)
df_stats.loc["\u0394Albedo_LCC",
             "Min"] = np.round(dalbedo_clean_lcc.min().values, 3)
df_stats.loc["\u0394Albedo_LCC",
             "StD"] = np.round(dalbedo_clean_lcc.std().values, 3)
df_stats.loc["\u0394ET_LCC[mm]", "Max"] = np.round(det_clean_lcc.max().values,
                                                   3)
df_stats.loc["\u0394ET_LCC[mm]", "Min"] = np.round(det_clean_lcc.min().values,
                                                   3)
df_stats.loc["\u0394ET_LCC[mm]", "StD"] = np.round(det_clean_lcc.std().values,
                                                   3)

df_stats.loc["\u0394LST_NV[K]", "Max"] = np.round(dlst_clean_nv.max().values,
                                                  3)
df_stats.loc["\u0394LST_NV[K]", "Min"] = np.round(dlst_clean_nv.min().values,
                                                  3)
df_stats.loc["\u0394LST_NV[K]", "StD"] = np.round(dlst_clean_nv.std().values,
                                                  3)
df_stats.loc["\u0394Albedo_NV",
             "Max"] = np.round(dalbedo_clean_nv.max().values, 3)
df_stats.loc["\u0394Albedo_NV",
             "Min"] = np.round(dalbedo_clean_nv.min().values, 3)
df_stats.loc["\u0394Albedo_NV",
             "StD"] = np.round(dalbedo_clean_nv.std().values, 3)
df_stats.loc["\u0394ET_NV[mm]", "Max"] = np.round(det_clean_nv.max().values, 3)
df_stats.loc["\u0394ET_NV[mm]", "Min"] = np.round(det_clean_nv.min().values, 3)
df_stats.loc["\u0394ET_NV[mm]", "StD"] = np.round(det_clean_nv.std().values, 3)
df_stats.loc["\u0394LST_Total[K]",
             "Max"] = np.round(dlst_clean_total.max().values, 3)
df_stats.loc["\u0394LST_Total[K]",
             "Min"] = np.round(dlst_clean_total.min().values, 3)
df_stats.loc["\u0394LST_Total[K]",
             "StD"] = np.round(dlst_clean_total.std().values, 3)
df_stats.loc["\u0394Albedo_Total",
             "Max"] = np.round(dalbedo_clean_total.max().values, 3)
df_stats.loc["\u0394Albedo_Total",
             "Min"] = np.round(dalbedo_clean_total.min().values, 3)
df_stats.loc["\u0394Albedo_Total",
             "StD"] = np.round(dalbedo_clean_total.std().values, 3)
df_stats.loc["\u0394ET_Total[mm]",
             "Max"] = np.round(det_clean_total.max().values, 3)
df_stats.loc["\u0394ET_Total[mm]",
             "Min"] = np.round(det_clean_total.min().values, 3)
df_stats.loc["\u0394ET_Total[mm]",
             "StD"] = np.round(det_clean_total.std().values, 3)

# Write the resutls in the results.txt (Note it will
# overwrite file with the same name)
with open(out_dir + "results.txt", "w") as f:
    f.write("This file contains the results of the annual data analyses\n")
    f.write("----------------------------------------------------------\n\n")
    f.write(f"Percent change: \
{np.round(percent_changed.values,3)}%\n\n")

with open(out_dir + "results.txt", "a") as f:
    f.write(df_stats.to_string(header=True, index=True))
    f.write(
        "\n-----------------------------------------------------------\n\n")
    f.write("Analyzing the LST data\n")
    f.write("-----------------------------------------------------------\n\n")

# Figure 2 of the manuscript (Main LST plot)
print("Working on LST:\n")
for i in range(len(lc_names)):
    # Skip the bog and shallow and litteral classes due to very sparse
    # data points
    if (i == 7) | (i == 8) | (i == 9):
        continue
    print(f"Analyzing {lc_names[i]}")
    dlcc_tmp_clean = dlcc_clean.isel(LC=i)
    df = pd.DataFrame({
        "dlst": dlst_clean_lcc,
        "dalbedo": dalbedo_clean_lcc,
        "det": det_clean_lcc,
        "w": weights_clean,
        "dlcc": dlcc_tmp_clean
    })
    df = df.dropna()

    # Bin data based on dLCC
    dlcc_bins = np.linspace(-1.001, 1, 2002)
    out = fit_bins_ols(df=df, bins=dlcc_bins, var="dlst")

    # The equation for the mean model (i.e. gain/loss)
    if out[5] > 0:
        eq_text = (f"\u0394LST = {out[3]}" + "(\u00B1" + f"{out[4]})" + "+" +
                   f"{out[5]}(\u00B1" + f"{out[6]})" + "X")
    else:
        eq_text = (f"\u0394LST = {out[3]}" + "(\u00B1" + f"{out[4]})" +
                   f"{out[5]}(\u00B1" + f"{out[6]})" + "X")

    with open(out_dir + "results.txt", "a") as f:
        f.write(f"\n\nResults of {lc_names[i]} analyses\n")
        f.write("------------------------------")
        f.write(f"\nLinear model fit to {lc_names[i]} gain/lost:\n")
        f.write(eq_text)
        f.write(
            f"\nn:{len(out[0])}. NOTE Data are binned, refer to the manuscript\n"
        )
        f.write(f"p-value intercept: {np.round(out[8][0],3)}\n")
        f.write(f"p-value slope: {np.round(out[8][1],3)}\n")
        f.write((f"Range of LCC:{np.round(np.min(out[0]),2)} - "
                 f"{np.round(np.max(out[0]),2)}"))

    # Now plot class transitions on top of the gain/loss
    # for k in range(1):
    for k in range(len(lc_names)):
        if (k == 7) | (k == 8) | (k == 9):
            continue
        if k == i:
            continue
        # transintion_loss is transition of class i to class k
        transintion_loss = normalized_confusion_clean[:, i, k]
        df_loss = pd.DataFrame({
            "dlst": dlst_clean_lcc,
            "w": weights_clean,
            "dlcc": -transintion_loss
        })
        df_loss = df_loss.dropna()
        bins_loss = np.linspace(-1, 0, 1001)
        df_loss["bins"] = pd.cut(df_loss["dlcc"],
                                 bins=bins_loss,
                                 include_lowest=True)
        out_loss = fit_bins_ols(df=df_loss, bins=bins_loss, var="dlst")
        # transintion_gain is transition of class k to class i
        transintion_gain = normalized_confusion_clean[:, k, i]
        df_gain = pd.DataFrame({
            "dlst": dlst_clean_lcc,
            "w": weights_clean,
            "dlcc": transintion_gain,
        })
        df_gain = df_gain.dropna()
        bins_gain = np.linspace(0, 1, 1001)
        out_gain = fit_bins_ols(df=df_gain, bins=bins_gain, var="dlst")

        # Concatenate the loss and gain transitions and fit a linear model
        X = np.append(out_loss[0], out_gain[0])
        Y = np.append(out_loss[1], out_gain[1])
        reg_weights = np.append(out_loss[2], out_gain[2])
        XX = sm.add_constant(X)
        mod_wls = sm.WLS(Y, XX, weights=reg_weights)
        res_wls = mod_wls.fit()
        predicts = res_wls.predict(XX)
        params = np.round(res_wls.params, 3)  # intercept, slope
        params_bse = np.round(res_wls.bse, 3)  # error on params
        pvalues = res_wls.pvalues  # pvalues of estimated params
        # if ((np.min(X) > -0.5) | (np.max(X) < 0.5) | (pvalues[1] > 0.05)):
        # if ((np.min(X) > -0.5) | (np.max(X) < 0.5) | (pvalues[1] > 0.05)):
        #     continue
        if params[1] > 0:
            eq_text_tmp = (f"\u0394LST={params[0]}" + "(\u00B1" +
                           f"{params_bse[0]})" + "+" + f"{params[1]}(\u00B1" +
                           f"{params_bse[1]})" + "X")
        else:
            eq_text_tmp = (f"\u0394LST={params[0]}" + "(\u00B1" +
                           f"{params_bse[0]})" + f"{params[1]}(\u00B1" +
                           f"{params_bse[1]})" + "X")

        with open(out_dir + "results.txt", "a") as f:
            f.write(f"\n\nLinear model between {lc_names[i]}-{lc_names[k]}:\n")
            f.write(eq_text_tmp)
            f.write(f"\nn: {len(XX)}\n")
            f.write(f"p-value intercept: {pvalues[0]}\n")
            f.write(f"p-value slope: {pvalues[1]}\n")
            f.write(
                f"Range of LCC:{np.round(np.min(X),2)} - {np.round(np.max(X),2)}"
            )

# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
print("Working on Albedo:\n")
with open(out_dir + "results.txt", "a") as f:
    f.write("\n-------------------------------------------------------\n\n")
    f.write("Analyzing the Alebedo data\n")
    f.write("\n------------------------\n")

for i in range(len(lc_names)):
    # Skip the bog and shallow and litteral classes due to very sparse
    # data points
    if (i == 7) | (i == 8) | (i == 9):
        continue
    print(f"Analyzing {lc_names[i]}")
    dlcc_tmp_clean = dlcc_clean.isel(LC=i)
    df = pd.DataFrame({
        "dlst": dlst_clean_lcc,
        "dalbedo": dalbedo_clean_lcc,
        "det": det_clean_lcc,
        "w": weights_clean,
        "dlcc": dlcc_tmp_clean
    })
    df = df.dropna()
    # Bin data based on dLCC
    dlcc_bins = np.linspace(-1.001, 1, 2002)
    out = fit_bins_ols(df=df, bins=dlcc_bins, var="dalbedo")
    # The equation for the mean model (i.e. gain/loss)
    if out[5] > 0:
        eq_text = (f"\u0394Albedo = {out[3]}" + "(\u00B1" + f"{out[4]})" +
                   "+" + f"{out[5]}(\u00B1" + f"{out[6]})" + "X")
    else:
        eq_text = (f"\u0394Albedo = {out[3]}" + "(\u00B1" + f"{out[4]})" +
                   f"{out[5]}(\u00B1" + f"{out[6]})" + "X")

    with open(out_dir + "results.txt", "a") as f:
        f.write(f"\n\nResults of {lc_names[i]} analyses\n")
        f.write("------------------------------")
        f.write(f"\nLinear model fit to {lc_names[i]} gain/lost Albedo:\n")
        f.write(eq_text)
        f.write(
            f"\nn:{len(out[0])}. NOTE Data are binned, refer to the manuscript\n"
        )
        f.write(f"p-value intercept: {np.round(out[8][0],3)}\n")
        f.write(f"p-value slope: {np.round(out[8][1],3)}\n")
        f.write((f"Range of LCC:{np.round(np.min(out[0]),2)} - "
                 f"{np.round(np.max(out[0]),2)}"))
    # Now plot class transitions on top of the gain/loss
    # for k in range(1):
    for k in range(len(lc_names)):
        if (k == 7) | (k == 8) | (k == 9):
            continue
        if k == i:
            continue
        # transintion_loss is transition of class i to class k
        transintion_loss = normalized_confusion_clean[:, i, k]
        df_loss = pd.DataFrame({
            "dlst": dlst_clean_lcc,
            "dalbedo": dalbedo_clean_lcc,
            "det": det_clean_lcc,
            "w": weights_clean,
            "dlcc": -transintion_loss
        })
        df_loss = df_loss.dropna()
        bins_loss = np.linspace(-1, 0, 1001)
        df_loss["bins"] = pd.cut(df_loss["dlcc"],
                                 bins=bins_loss,
                                 include_lowest=True)
        out_loss = fit_bins_ols(df=df_loss, bins=bins_loss, var="dalbedo")
        # transintion_gain is transition of class k to class i
        transintion_gain = normalized_confusion_clean[:, k, i]
        df_gain = pd.DataFrame({
            "dlst": dlst_clean_lcc,
            "dalbedo": dalbedo_clean_lcc,
            "det": det_clean_lcc,
            "w": weights_clean,
            "dlcc": transintion_gain,
        })
        df_gain = df_gain.dropna()
        bins_gain = np.linspace(0, 1, 1001)
        out_gain = fit_bins_ols(df=df_gain, bins=bins_gain, var="dalbedo")

        # Concatenate the loss and gain transitions and fit a linear model
        X = np.append(out_loss[0], out_gain[0])
        # if ((np.min(X) > -0.5) | (np.max(X) < 0.5)):
        # continue
        Y = np.append(out_loss[1], out_gain[1])
        reg_weights = np.append(out_loss[2], out_gain[2])
        XX = sm.add_constant(X)
        mod_wls = sm.WLS(Y, XX, weights=reg_weights)
        res_wls = mod_wls.fit()
        predicts = res_wls.predict(XX)
        params = np.round(res_wls.params, 3)  # intercept, slope
        params_bse = np.round(res_wls.bse, 3)  # error on params
        pvalues = res_wls.pvalues  # pvalues of estimated params
        if params[1] > 0:
            eq_text_tmp = (f"\u0394Albedo={params[0]}" + "(\u00B1" +
                           f"{params_bse[0]})" + "+" + f"{params[1]}(\u00B1" +
                           f"{params_bse[1]})" + "X")
        else:
            eq_text_tmp = (f"\u0394Albedo={params[0]}" + "(\u00B1" +
                           f"{params_bse[0]})" + f"{params[1]}(\u00B1" +
                           f"{params_bse[1]})" + "X")
        with open(out_dir + "results.txt", "a") as f:
            f.write(f"\n\nLinear model between {lc_names[i]}-{lc_names[k]}:\n")
            f.write(eq_text_tmp)
            f.write(f"\nn: {len(XX)}\n")
            f.write(f"p-value intercept: {pvalues[0]}\n")
            f.write(f"p-value slope: {pvalues[1]}\n")
            f.write(
                f"Range of LCC:{np.round(np.min(X),2)} - {np.round(np.max(X),2)}"
            )
# -------------------------------------------------------------------------
print("Working on ET:\n")
with open(out_dir + "results.txt", "a") as f:
    f.write("\n-------------------------------------------------------\n\n")
    f.write("Analyzing the ET data\n")
    f.write("\n------------------------\n")

for i in range(len(lc_names)):
    # Skip the bog and shallow and litteral classes due to very sparse
    # data points
    if (i == 7) | (i == 8) | (i == 9):
        continue
    print(f"Analyzing {lc_names[i]}")
    dlcc_tmp_clean = dlcc_clean.isel(LC=i)
    df = pd.DataFrame({
        "dlst": dlst_clean_lcc,
        "dalbedo": dalbedo_clean_lcc,
        "det": det_clean_lcc,
        "w": weights_clean,
        "dlcc": dlcc_tmp_clean
    })
    df = df.dropna()

    # Bin data based on dLCC
    dlcc_bins = np.linspace(-1.001, 1, 2002)
    out = fit_bins_ols(df=df, bins=dlcc_bins, var="det")
    # The equation for the mean model (i.e. gain/loss)
    if out[5] > 0:
        eq_text = (f"\u0394ET = {out[3]}" + "(\u00B1" + f"{out[4]})" + "+" +
                   f"{out[5]}(\u00B1" + f"{out[6]})" + "X")
    else:
        eq_text = (f"\u0394ET = {out[3]}" + "(\u00B1" + f"{out[4]})" +
                   f"{out[5]}(\u00B1" + f"{out[6]})" + "X")

    with open(out_dir + "results.txt", "a") as f:
        f.write(f"\n\nResults of {lc_names[i]} analyses\n")
        f.write("------------------------------")
        f.write(f"\nLinear model fit to {lc_names[i]} gain/lost ET:\n")
        f.write(eq_text)
        f.write(
            f"\nn:{len(out[0])}. NOTE Data are binned, refer to the manuscript\n"
        )
        f.write(f"p-value intercept: {np.round(out[8][0],3)}\n")
        f.write(f"p-value slope: {np.round(out[8][1],3)}\n")
        f.write((f"Range of LCC:{np.round(np.min(out[0]),2)} - "
                 f"{np.round(np.max(out[0]),2)}"))

    # Now plot class transitions on top of the gain/loss
    for k in range(len(lc_names)):
        if (k == 7) | (k == 8) | (k == 9):
            continue
        if k == i:
            continue
        # transintion_loss is transition of class i to class k
        transintion_loss = normalized_confusion_clean[:, i, k]
        df_loss = pd.DataFrame({
            "dlst": dlst_clean_lcc,
            "dalbedo": dalbedo_clean_lcc,
            "det": det_clean_lcc,
            "w": weights_clean,
            "dlcc": -transintion_loss
        })
        df_loss = df_loss.dropna()
        bins_loss = np.linspace(-1, 0, 1001)
        df_loss["bins"] = pd.cut(df_loss["dlcc"],
                                 bins=bins_loss,
                                 include_lowest=True)
        out_loss = fit_bins_ols(df=df_loss, bins=bins_loss, var="det")
        # transintion_gain is transition of class k to class i
        transintion_gain = normalized_confusion_clean[:, k, i]
        df_gain = pd.DataFrame({
            "dlst": dlst_clean_lcc,
            "dalbedo": dalbedo_clean_lcc,
            "det": det_clean_lcc,
            "w": weights_clean,
            "dlcc": transintion_gain,
        })
        df_gain = df_gain.dropna()
        bins_gain = np.linspace(0, 1, 1001)
        out_gain = fit_bins_ols(df=df_gain, bins=bins_gain, var="det")

        # Concatenate the loss and gain transitions and fit a linear model
        X = np.append(out_loss[0], out_gain[0])
        Y = np.append(out_loss[1], out_gain[1])
        reg_weights = np.append(out_loss[2], out_gain[2])
        XX = sm.add_constant(X)
        mod_wls = sm.WLS(Y, XX, weights=reg_weights)
        res_wls = mod_wls.fit()
        predicts = res_wls.predict(XX)
        params = np.round(res_wls.params, 3)  # intercept, slope
        params_bse = np.round(res_wls.bse, 3)  # error on params
        pvalues = res_wls.pvalues  # pvalues of estimated params
        if params[1] > 0:
            eq_text_tmp = (f"\u0394ET={params[0]}" + "(\u00B1" +
                           f"{params_bse[0]})" + "+" + f"{params[1]}(\u00B1" +
                           f"{params_bse[1]})" + "X")
        else:
            eq_text_tmp = (f"\u0394ET={params[0]}" + "(\u00B1" +
                           f"{params_bse[0]})" + f"{params[1]}(\u00B1" +
                           f"{params_bse[1]})" + "X")
        with open(out_dir + "results.txt", "a") as f:
            f.write(f"\n\nLinear model between {lc_names[i]}-{lc_names[k]}:\n")
            f.write(eq_text_tmp)
            f.write(f"\nn: {len(XX)}\n")
            f.write(f"p-value intercept: {pvalues[0]}\n")
            f.write(f"p-value slope: {pvalues[1]}\n")
            f.write(
                f"Range of LCC:{np.round(np.min(X),2)} - {np.round(np.max(X),2)}"
            )
# -------------------------------------------------------

# # Figure 1 box plot of ET/ALBEDO/LST for different land cover with >98 cover

# in_dir = "/data/ABOVE/Final_data/"
# out_dir = ("/data/home/hamiddashti/nasa_above/outputs/Natural_Variability/"
#            "Natural_Variability_Annual_outputs/EndPoints/all_bands/")

# luc = xr.open_dataarray(in_dir + "LUC/LUC_10/LULC_10_2003_2014.nc")
# lst_mean = xr.open_dataarray(in_dir +
#                              "LST_Final/LST/Annual_Mean/lst_mean_annual.nc")
# albedo = xr.open_dataarray(in_dir +
#                            "ALBEDO_Final/Annual_Albedo/Albedo_annual.nc")
# et = xr.open_dataarray(in_dir + "ET_Final/Annual_ET/ET_Annual.nc")

# luc_2003 = luc.loc[2003]
# lst_mean_2003 = lst_mean.loc[2003]
# albedo_2003 = albedo.loc[2003]
# et_2003 = et.loc[2003]
# lc_names = [
#     "EF", "DF", "Shrub", "Herbaceous", "Sparse", "Barren", "Fen", "Water"
# ]
# df = pd.DataFrame(columns=[
#     "EF",
# ])
# for i in range(len(lc_names)):
#     lst_tmp = lst_mean_2003.where(luc_2003.isel(band=i) > 0.98).values
#     df[lc_names[i]] = lst_tmp.ravel()
# df1 = df.dropna(how="all")
# len(df1)
# plt.close()
# df.boxplot()
