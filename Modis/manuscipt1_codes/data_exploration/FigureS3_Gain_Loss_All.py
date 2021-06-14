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
import matplotlib.gridspec as gridspec
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

# out_dir = (
#     "/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/test/")

# The map of dLST due to LCC
dlst_lcc = xr.open_dataarray(
    in_dir + ("Natural_Variability/Natural_Variability_Annual_outputs/"
              "geographic/02_percent/dlst_mean_lcc.nc"))
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
dlst = ct["DLST_MEAN_LCC"]  # Changed LST due to LCC
det = ct["DET_LCC"]  # Changed ET due to LCC
dalbedo = ct["DALBEDO_LCC"]  # Changed albedo due to LCC
dlcc = ct["DLCC"]  # Fractional change in land cover
normalized_confusion = ct["NORMALIZED_CONFUSION"]  # Normalized confusion
I_dlst = outliers_index(dlst, 2)  # Outlier indices for dLST
I_dalbedo = outliers_index(dalbedo, 2)  # Outlier indices for dalbedo
I_det = outliers_index(det, 2)  # Outlier indices for det

# Remove outliers based on indices
dlst_clean = dlst.where((I_dlst == False) & (I_dalbedo == False)
                        & (I_det == False))
dalbedo_clean = dalbedo.where((I_dlst == False) & (I_dalbedo == False)
                              & (I_det == False))
det_clean = det.where((I_dlst == False) & (I_dalbedo == False)
                      & (I_det == False))
weights_clean = weights.where((I_dlst == False) & (I_dalbedo == False)
                              & (I_det == False))
# dlcc_clean = dlcc.where(I_dlst == False)
# normalized_confusion_clean = normalized_confusion.where(I_dlst == False)
dlcc_clean = dlcc
normalized_confusion_clean = normalized_confusion

palette = sns.color_palette("tab10")  # Color palette for plots
lc_names = [
    "EF", "DF", "Shrub", "Herbaceous", "Sparse", "Barren", "Fen", "Bog",
    "Shallow_Littoral", "Water"
]

out_dir = ("/data/home/hamiddashti/nasa_above/outputs/data_analyses/Annual/"
           "Geographics/Figures_MS1/")
# Figure 2 of the manuscript (Main LST plot)
plt.close()
fig, axs = plt.subplots(7, 7, figsize=(15, 7), facecolor='w', edgecolor='k')
for i in range(len(lc_names)):
    # for i in range(4):
    # Skip the bog and shallow and litteral classes due to very sparse
    # data points
    if (i == 7) | (i == 8) | (i == 9):
        continue
    for k in range(len(lc_names)):
        # for k in range(4):
        if (i == k) | (i > k):
            axs[i, k].text(0.5, 0.5, "---")
            axs[i, k].axis("off")
            continue
        if (k == 7) | (k == 8) | (k == 9) | (k == i):
            continue

        # transintion_loss is transition of class i to class k
        transintion_loss = normalized_confusion_clean[:, i, k]
        df_loss = pd.DataFrame({
            "dlst": dlst_clean,
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
            "dlst": dlst_clean,
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
        pvalues = np.round(res_wls.pvalues, 3)  # pvalues of estimated params
        axs[i, k].scatter(X, Y, color="gray", s=0.5)
        axs[i, k].plot(X, predicts, linewidth=3, color="black")
        axs[i, k].set_xlim(-1, 1)

pltfont = {'fontname': 'Times New Roman'}
fig.text(0.01, 0.98, "A", fontsize=26, **pltfont)
fig.text(0.02, 0.93, "EF", fontsize=16, **pltfont)
fig.text(0.02, 0.79, "DF", fontsize=16, **pltfont)
fig.text(0.02, 0.65, "Shrub", fontsize=16, **pltfont)
fig.text(0.02, 0.505, "Herb", fontsize=16, **pltfont)
fig.text(0.02, 0.36, "Sparse", fontsize=16, **pltfont)
fig.text(0.02, 0.215, "Barren", fontsize=16, **pltfont)
fig.text(0.02, 0.072, "Fen", fontsize=16, **pltfont)
fig.text(0.06, 0.99, "EF", fontsize=16, **pltfont)
fig.text(0.20, 0.99, "DF", fontsize=16, **pltfont)
fig.text(0.33, 0.99, "Shrub", fontsize=16, **pltfont)
fig.text(0.47, 0.99, "Herb", fontsize=16, **pltfont)
fig.text(0.62, 0.99, "Sparse", fontsize=16, **pltfont)
fig.text(0.76, 0.99, "Barren", fontsize=16, **pltfont)
fig.text(0.91, 0.99, "Fen", fontsize=16, **pltfont)
plt.tight_layout()
save(out_dir + "FigS4_LST_transitions_subplots.png", bbox_inches='tight')

# Figure 2 of the manuscript (Main LST plot)
plt.close()
fig, axs = plt.subplots(7, 7, figsize=(15, 7), facecolor='w', edgecolor='k')
for i in range(len(lc_names)):
    # for i in range(4):
    # Skip the bog and shallow and litteral classes due to very sparse
    # data points
    if (i == 7) | (i == 8) | (i == 9):
        continue
    for k in range(len(lc_names)):
        # for k in range(4):
        if (i == k) | (i > k):
            axs[i, k].text(0.5, 0.5, "---")
            axs[i, k].axis("off")
            continue
        if (k == 7) | (k == 8) | (k == 9) | (k == i):
            continue

        # transintion_loss is transition of class i to class k
        transintion_loss = normalized_confusion_clean[:, i, k]
        df_loss = pd.DataFrame({
            "dlst": dlst_clean,
            "dalbedo": dalbedo_clean,
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
            "dlst": dlst_clean,
            "dalbedo": dalbedo_clean,
            "w": weights_clean,
            "dlcc": transintion_gain,
        })
        df_gain = df_gain.dropna()
        bins_gain = np.linspace(0, 1, 1001)
        out_gain = fit_bins_ols(df=df_gain, bins=bins_gain, var="dalbedo")

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
        pvalues = np.round(res_wls.pvalues, 3)  # pvalues of estimated params
        axs[i, k].scatter(X, Y, color="gray", s=0.5)
        axs[i, k].plot(X, predicts, linewidth=3, color="black")
        axs[i, k].set_xlim(-1, 1)
pltfont = {'fontname': 'Times New Roman'}
fig.text(0.01, 0.98, "B", fontsize=26, **pltfont)
fig.text(0.016, 0.93, "EF", fontsize=16, **pltfont)
fig.text(0.016, 0.79, "DF", fontsize=16, **pltfont)
fig.text(0.016, 0.65, "Shrub", fontsize=16, **pltfont)
fig.text(0.016, 0.505, "Herb", fontsize=16, **pltfont)
fig.text(0.016, 0.36, "Sparse", fontsize=16, **pltfont)
fig.text(0.016, 0.215, "Barren", fontsize=16, **pltfont)
fig.text(0.016, 0.072, "Fen", fontsize=16, **pltfont)
fig.text(0.06, 0.99, "EF", fontsize=16, **pltfont)
fig.text(0.20, 0.99, "DF", fontsize=16, **pltfont)
fig.text(0.33, 0.99, "Shrub", fontsize=16, **pltfont)
fig.text(0.47, 0.99, "Herb", fontsize=16, **pltfont)
fig.text(0.62, 0.99, "Sparse", fontsize=16, **pltfont)
fig.text(0.76, 0.99, "Barren", fontsize=16, **pltfont)
fig.text(0.91, 0.99, "Fen", fontsize=16, **pltfont)
plt.tight_layout()
save(out_dir + "FigS4_Albedo_transitions_subplots.png", bbox_inches='tight')

plt.close()
fig, axs = plt.subplots(7, 7, figsize=(15, 7), facecolor='w', edgecolor='k')
for i in range(len(lc_names)):
    # for i in range(4):
    # Skip the bog and shallow and litteral classes due to very sparse
    # data points
    if (i == 7) | (i == 8) | (i == 9):
        continue
    for k in range(len(lc_names)):
        # for k in range(4):
        if (i == k) | (i > k):
            axs[i, k].text(0.5, 0.5, "---")
            axs[i, k].axis("off")
            continue
        if (k == 7) | (k == 8) | (k == 9) | (k == i):
            continue

        # transintion_loss is transition of class i to class k
        transintion_loss = normalized_confusion_clean[:, i, k]
        df_loss = pd.DataFrame({
            "dlst": dlst_clean,
            "dalbedo": dalbedo_clean,
            "det": det_clean,
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
            "dlst": dlst_clean,
            "dalbedo": dalbedo_clean,
            "det": det_clean,
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
        pvalues = np.round(res_wls.pvalues, 3)  # pvalues of estimated params
        axs[i, k].scatter(X, Y, color="gray", s=0.5)
        axs[i, k].plot(X, predicts, linewidth=3, color="black")
        axs[i, k].set_xlim(-1, 1)
pltfont = {'fontname': 'Times New Roman'}
fig.text(0.01, 0.98, "C", fontsize=26, **pltfont)
fig.text(0.016, 0.93, "EF", fontsize=16, **pltfont)
fig.text(0.016, 0.79, "DF", fontsize=16, **pltfont)
fig.text(0.016, 0.65, "Shrub", fontsize=16, **pltfont)
fig.text(0.016, 0.505, "Herb", fontsize=16, **pltfont)
fig.text(0.016, 0.36, "Sparse", fontsize=16, **pltfont)
fig.text(0.016, 0.215, "Barren", fontsize=16, **pltfont)
fig.text(0.016, 0.072, "Fen", fontsize=16, **pltfont)
fig.text(0.06, 0.99, "EF", fontsize=16, **pltfont)
fig.text(0.20, 0.99, "DF", fontsize=16, **pltfont)
fig.text(0.33, 0.99, "Shrub", fontsize=16, **pltfont)
fig.text(0.47, 0.99, "Herb", fontsize=16, **pltfont)
fig.text(0.62, 0.99, "Sparse", fontsize=16, **pltfont)
fig.text(0.76, 0.99, "Barren", fontsize=16, **pltfont)
fig.text(0.91, 0.99, "Fen", fontsize=16, **pltfont)
plt.tight_layout()
save(out_dir + "FigS4_ET_transitions_subplots.png", bbox_inches='tight')
