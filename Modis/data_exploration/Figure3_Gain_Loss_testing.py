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
from statsmodels.stats.outliers_influence import summary_table
""" ---------------------------------------------------------------------------
defining functions used in this script
----------------------------------------------------------------------------"""


def outliers_index(data, m=3.5):
    """
    Returns true if a value is outlier
    https://www.itl.nist.gov/div898/handbook/eda/section3/eda356.htm#MAD
    https://stats.stackexchange.com/questions/123895/
    mad-formula-for-outlier-detection
    :param int data: numpy array
    :param int m: # of std to include data 
    """
    import numpy as np
    d = (np.abs(data - np.nanmedian(data)))
    mdev = np.nanmedian(d)
    # s = (0.6745 * d) / mdev if mdev else 0.
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
    counts = df.groupby('bins').count()["dlcc"]
    # wsums = df.groupby('bins').count()["dlcc"]
    # bins_mean = bins_mean.where(counts >=3)

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

    # mod_ols = sm.OLS(y, X)
    # res_ols = mod_ols.fit()
    intercept, slope = np.round(res_wls.params, 3)
    intercept_bse, slope_bse = np.round(res_wls.bse, 3)
    predicts = res_wls.predict(X)
    pvalues = res_wls.pvalues
    st, data, ss2 = summary_table(res_wls, alpha=0.05)
    predict_mean_ci_low, predict_mean_ci_upp = data[:, 4:6].T
    out_list = [
        x, y, intercept, intercept_bse, slope, slope_bse, predicts, pvalues,
        predict_mean_ci_upp, predict_mean_ci_low,sample_weight
    ]
    return out_list


class ScalarFormatterForceFormat(ScalarFormatter):
    def _set_format(self):  # Override function that finds format to use.
        self.format = "%1.1f"  # Give format here

yfmt = ScalarFormatterForceFormat()
yfmt.set_powerlimits((0, 0))

in_dir = ("/data/home/hamiddashti/nasa_above/outputs/")
# out_dir = ("/data/home/hamiddashti/nasa_above/outputs/data_analyses/Annual/"
#            "Geographics/Figures_MS1/")

out_dir = (
    "/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/test/")

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
ct = xr.open_dataset(
    in_dir + ("Sensitivity/EndPoints/Annual/Geographic/"
              "02_percent/Confusion_Table_final_02precent_new_albedo.nc"))

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
I_dlst = outliers_index(dlst)  # Outlier indices for dLST
I_dalbedo = outliers_index(dalbedo)  # Outlier indices for dalbedo
I_det = outliers_index(det)  # Outlier indices for det

# Remove outliers based on indices
dlst_clean = dlst.where((I_dlst == False) & (I_dalbedo == False)
                        & (I_det == False))
dalbedo_clean = dalbedo.where((I_dlst == False) & (I_dalbedo == False)
                              & (I_det == False))
det_clean = det.where((I_dlst == False) & (I_dalbedo == False)
                      & (I_det == False))

# dlcc_clean = dlcc.where(I_dlst == False)
# normalized_confusion_clean = normalized_confusion.where(I_dlst == False)
weights_clean = weights
dlcc_clean = dlcc
normalized_confusion_clean = normalized_confusion

palette = sns.color_palette("tab10")  # Color palette for plots
lc_names = [
    "EF", "DF", "Shrub", "Herbaceous", "Sparse", "Barren", "Fen", "Bog",
    "Shallow_Littoral", "Water"
]


# Figure 2 of the manuscript (Main LST plot)
print("\nWorking on LST:\n")
plt.close()
fig, axs = plt.subplots(2, 4, figsize=(15, 7), facecolor='w', edgecolor='k')
axs = axs.ravel()
axs_counter = 0
for i in range(len(lc_names)):
    # Skip the bog and shallow and litteral classes due to very sparse
    # data points
    if (i == 7) | (i == 8) | (i == 9):
        continue
    print(f"Analyzing {lc_names[i]}")
    dlcc_tmp_clean = dlcc_clean.isel(LC=i)
    df = pd.DataFrame({
        "dlst": dlst_clean,
        "dalbedo": dalbedo_clean,
        "det": det_clean,
        "w": weights_clean,
        "dlcc": dlcc_tmp_clean
    })
    df = df.dropna()
    # Bin data based on dLCC
    dlcc_bins = np.linspace(-1.01, 1, 2002)
    out = fit_bins_ols(df=df, bins=dlcc_bins, var="dlst")
    # The equation for the mean model (i.e. gain/loss)
    if out[5] > 0:
        eq_text = (f"\u0394LST = {out[2]}" + "(\u00B1" + f"{out[3]})" + "+" +
                   f"{out[4]}(\u00B1" + f"{out[5]})" + "X")
    else:
        eq_text = (f"\u0394LST = {out[2]}" + "(\u00B1" + f"{out[3]})" +
                   f"{out[4]}(\u00B1" + f"{out[5]})" + "X")

    # Scatter plot of the gain/loss
    # axs[axs_counter].scatter(x=out[0], y=out[1], color="gray")
    axs[axs_counter].plot(out[0],
                          out[6],
                          color='black',
                          linestyle="--",
                          linewidth=3,
                          label='Total gain and loss')
    axs[axs_counter].fill_between(out[0],
                                  out[8],
                                  out[9],
                                  color="gray",
                                  alpha=0.3)

    axs[axs_counter].set_title(lc_names[i], fontsize=16)
    axs[axs_counter].tick_params(labelsize=16)
    axs[axs_counter].axvline(0, ls='--', c="k", linewidth=1)
    axs[axs_counter].text(0,
                          1.4,
                          eq_text,
                          horizontalalignment='center',
                          verticalalignment='center',
                          fontsize=8,
                          fontweight="bold",
                          color="black")
    axs[axs_counter].text(0.18,
                          -1.35,
                          "Gain",
                          ha="center",
                          va="center",
                          size=10,
                          color="black",
                          bbox=dict(boxstyle="rarrow,pad=0.3",
                                    fc="gray",
                                    ec="gray",
                                    alpha=0.5,
                                    lw=2))
    axs[axs_counter].text(-0.17,
                          -1.35,
                          "Loss",
                          ha="center",
                          va="center",
                          size=10,
                          color="black",
                          bbox=dict(boxstyle="larrow,pad=0.3",
                                    fc="gray",
                                    ec="gray",
                                    alpha=0.5,
                                    lw=2))
    axs[axs_counter].set_xlim(-1, 1)
    axs[axs_counter].set_ylim(-2.5, 2.5)

    # Now plot class transitions on top of the gain/loss
    # for k in range(1):
    for k in range(len(lc_names)):
        if (k == 7) | (k == 8) | (k == 9):
            continue
        if k == i:
            continue
        print(f"Transition to --> {lc_names[k]}")
        # transintion_loss is transition of class i to class k
        transintion_loss = normalized_confusion_clean[:, i, k]
        df_loss = pd.DataFrame({
            "dlst": dlst_clean,
            "w": weights_clean,
            "dlcc": -transintion_loss
        })
        df_loss = df_loss.dropna()
        bins_loss = np.linspace(-1, 0, 101)
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
        bins_gain = np.linspace(0, 1, 101)
        out_gain = fit_bins_ols(df=df_gain, bins=bins_gain, var="dlst")

        # Concatenate the loss and gain transitions and fit a linear model
        X = np.append(out_loss[0], out_gain[0])
        Y = np.append(out_loss[1], out_gain[1])
        sample_weights = np.append(out_loss[10], out_gain[10])
        XX = sm.add_constant(X)
        mod_wls = sm.WLS(Y, XX,weights=sample_weights)
        res_wls = mod_wls.fit()
        predicts = res_wls.predict(XX)
plt.close()
        # axs[axs_counter].plot(X,
        #                       predicts,
        #                       color=palette[k],
        #                       linewidth=3,
        #                       label=lc_names[k])
plt.plot(X,predicts)
        st, data, ss2 = summary_table(res_wls, alpha=0.05)
        predict_mean_ci_low, predict_mean_ci_upp = data[:, 6:8].T
plt.fill_between(X,predict_mean_ci_upp,predict_mean_ci_low,alpha=0.5)
        # axs[axs_counter].plot(X, predict_mean_ci_low, 'r--', lw=2)
        # axs[axs_counter].plot(X, predict_mean_ci_upp, 'r--', lw=2)
        # axs[axs_counter].fill_between(X,
        #                               predict_mean_ci_upp,
        #                               predict_mean_ci_low,
        #                               color=palette[k],
        #                               alpha=0.5)
plt.savefig(out_dir+"test3.png")


        
        params = np.round(res_wls.params, 3)  # intercept, slope
        params_bse = np.round(res_wls.bse, 3)  # error on params
        pvalues = np.round(res_wls.pvalues, 3)  # pvalues of estimated params
        if ((np.min(X) > -0.5) | (np.max(X) < 0.5) | (pvalues[1] > 0.05) |
            (pvalues[0] > 0.05)):
            continue
        if params[1] > 0:
            eq_text_tmp = (f"\u0394LST={params[0]}" + "(\u00B1" +
                           f"{params_bse[0]})" + "+" + f"{params[1]}(\u00B1" +
                           f"{params_bse[1]})" + "X")
        else:
            eq_text_tmp = (f"\u0394LST={params[0]}" + "(\u00B1" +
                           f"{params_bse[0]})" + f"{params[1]}(\u00B1" +
                           f"{params_bse[1]})" + "X")

        axs[axs_counter].plot(X,
                              predicts,
                              color=palette[k],
                              linewidth=3,
                              label=lc_names[k])
        st, data, ss2 = summary_table(res_wls, alpha=0.05)
        predict_mean_ci_low, predict_mean_ci_upp = data[:, 4:6].T
        # axs[axs_counter].plot(X, predict_mean_ci_low, 'r--', lw=2)
        # axs[axs_counter].plot(X, predict_mean_ci_upp, 'r--', lw=2)
        axs[axs_counter].fill_between(X,
                                      predict_mean_ci_upp,
                                      predict_mean_ci_low,
                                      color=palette[k],
                                      alpha=0.5)

    axs_counter += 1



# Sorting out the legend
labels_handles = {
    label: handle
    for ax in fig.axes
    for handle, label in zip(*ax.get_legend_handles_labels())
}
labels_handles["Evergreen Forest (EF)"] = labels_handles["EF"]
labels_handles["Deciduous Forest (DF)"] = labels_handles["DF"]
del labels_handles["EF"]
del labels_handles["DF"]
desired_legend_order = [
    "Total gain and loss", "Evergreen Forest (EF)", "Deciduous Forest (DF)",
    "Shrub", "Herbaceous", "Sparse", "Barren", "Fen"
]
reordered_labels_handles = {k: labels_handles[k] for k in desired_legend_order}
bottom_right_ax = axs[-1]
bottom_right_ax.clear()  # clears the random data I plotted previously
bottom_right_ax.set_axis_off()  # removes the XY axes
plt.legend(
    reordered_labels_handles.values(),
    reordered_labels_handles.keys(),
    # bbox_to_anchor=(1, 1),
    loc='center',
    # fontsize=30,
    prop={
        "family": "Times New Roman",
        # "weight": "bold"
        "size": 14
    },
)
pltfont = {'fontname': 'Times New Roman'}

# fig.supxlabel('Fractional change in land cover', fontsize=16, **pltfont)
fig.supylabel("$\Delta LST_{LCC}$ [k]", fontsize=16, **pltfont)
fig.suptitle("A", x=0.02, y=0.85, size=30, **pltfont)
plt.tight_layout()
save(out_dir + "Figures/Fig3_LST_Gain_Loss_geographic.png")

# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
print("\nWorking on Albedo:\n")
plt.close()
fig, axs = plt.subplots(2, 4, figsize=(15, 7), facecolor='w', edgecolor='k')
axs = axs.ravel()
axs_counter = 0
for i in range(len(lc_names)):
    # Skip the bog and shallow and litteral classes due to very sparse
    # data points
    if (i == 7) | (i == 8) | (i == 9):
        continue
    print(f"Analyzing {lc_names[i]}")
    dlcc_tmp_clean = dlcc_clean.isel(LC=i)
    df = pd.DataFrame({
        "dlst": dlst_clean,
        "dalbedo": dalbedo_clean,
        "det": det_clean,
        "w": weights_clean,
        "dlcc": dlcc_tmp_clean
    })
    df = df.dropna()
    # Bin data based on dLCC
    dlcc_bins = np.linspace(-1.001, 1, 2002)
    out = fit_bins_ols(df=df, bins=dlcc_bins, var="dalbedo")
    # The equation for the mean model (i.e. gain/loss)
    if out[5] > 0:
        eq_text = (f"\u0394Albedo = {out[2]}" + "(\u00B1" + f"{out[3]})" +
                   "+" + f"{out[4]}(\u00B1" + f"{out[5]})" + "X")
    else:
        eq_text = (f"\u0394Albedo = {out[2]}" + "(\u00B1" + f"{out[3]})" +
                   f"{out[4]}(\u00B1" + f"{out[5]})" + "X")

    # Scatter plot of the gain/loss
    axs[axs_counter].scatter(x=out[0], y=out[1], color="gray")
    axs[axs_counter].plot(out[0],
                          out[6],
                          color='black',
                          linestyle="--",
                          linewidth=3,
                          label='Total gain and loss')
    axs[axs_counter].set_title(lc_names[i], fontsize=16)
    axs[axs_counter].tick_params(labelsize=16)
    axs[axs_counter].axvline(0, ls='--', c="k", linewidth=1)
    axs[axs_counter].text(0,
                          0.045,
                          eq_text,
                          horizontalalignment='center',
                          verticalalignment='center',
                          fontsize=8,
                          fontweight="bold",
                          color="black")
    axs[axs_counter].text(0.18,
                          -0.045,
                          "Gain",
                          ha="center",
                          va="center",
                          size=10,
                          color="black",
                          bbox=dict(boxstyle="rarrow,pad=0.03",
                                    fc="gray",
                                    ec="gray",
                                    alpha=0.5,
                                    lw=2))
    axs[axs_counter].text(-0.17,
                          -0.045,
                          "Loss",
                          ha="center",
                          va="center",
                          size=10,
                          color="black",
                          bbox=dict(boxstyle="larrow,pad=0.03",
                                    fc="gray",
                                    ec="gray",
                                    alpha=0.5,
                                    lw=2))
    axs[axs_counter].set_xlim(-1, 1)
    axs[axs_counter].set_ylim(-0.05, 0.05)
    axs[axs_counter].yaxis.set_major_formatter(yfmt)

    # Now plot class transitions on top of the gain/loss
    # for k in range(1):
    for k in range(len(lc_names)):
        if (k == 7) | (k == 8) | (i == 9):
            continue
        if k == i:
            continue
        print(f"Transition to --> {lc_names[k]}")
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
        out_loss = fit_bins_ols(df=df_loss, bins=bins_loss, var="dalbedo")
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
        out_gain = fit_bins_ols(df=df_gain, bins=bins_gain, var="dalbedo")

        # Concatenate the loss and gain transitions and fit a linear model
        X = np.append(out_loss[0], out_gain[0])
        Y = np.append(out_loss[1], out_gain[1])
        # reg_weights = np.append(out_loss[2], out_gain[2])
        XX = sm.add_constant(X)
        mod_ols = sm.OLS(Y, XX)
        res_ols = mod_ols.fit()
        predicts = res_ols.predict(XX)
        params = np.round(res_ols.params, 3)  # intercept, slope
        params_bse = np.round(res_ols.bse, 3)  # error on params
        pvalues = np.round(res_ols.pvalues, 3)  # pvalues of estimated params
        if ((np.min(X) > -0.5) | (np.max(X) < 0.5) | (pvalues[1] > 0.05)):
            continue
        if params[1] > 0:
            eq_text_tmp = (f"\u0394Albedo={params[0]}" + "(\u00B1" +
                           f"{params_bse[0]})" + "+" + f"{params[1]}(\u00B1" +
                           f"{params_bse[1]})" + "X")
        else:
            eq_text_tmp = (f"\u0394Albedo={params[0]}" + "(\u00B1" +
                           f"{params_bse[0]})" + f"{params[1]}(\u00B1" +
                           f"{params_bse[1]})" + "X")

        axs[axs_counter].plot(X,
                              predicts,
                              color=palette[k],
                              linewidth=3,
                              label=lc_names[k])

    axs_counter += 1

# Sorting out the legend
labels_handles = {
    label: handle
    for ax in fig.axes
    for handle, label in zip(*ax.get_legend_handles_labels())
}
labels_handles["Evergreen Forest (EF)"] = labels_handles["EF"]
labels_handles["Deciduous Forest (DF)"] = labels_handles["DF"]
del labels_handles["EF"]
del labels_handles["DF"]
desired_legend_order = [
    "Total gain and loss", "Evergreen Forest (EF)", "Deciduous Forest (DF)",
    "Shrub", "Herbaceous", "Sparse", "Barren", "Fen"
]
reordered_labels_handles = {k: labels_handles[k] for k in desired_legend_order}
bottom_right_ax = axs[-1]
bottom_right_ax.clear()  # clears the random data I plotted previously
bottom_right_ax.set_axis_off()  # removes the XY axes
plt.legend(
    reordered_labels_handles.values(),
    reordered_labels_handles.keys(),
    # bbox_to_anchor=(1, 1),
    loc='center',
    prop={
        "family": "Times New Roman",
        "size": 14
        # "weight": "bold"
    },
)
pltfont = {'fontname': 'Times New Roman'}
fig.suptitle("B", x=0.02, y=0.85, size=30, **pltfont)
# fig.supxlabel('Fractional change in land cover', fontsize=16, **pltfont)
fig.supylabel("$\Delta Albedo$", fontsize=16, **pltfont)
plt.tight_layout()
save(out_dir + "Fig3_Albedo_Gain_Loss.png")
# -------------------------------------------------------------------------
print("\nWorking on ET:\n")
plt.close()
fig, axs = plt.subplots(2, 4, figsize=(15, 7), facecolor='w', edgecolor='k')
axs = axs.ravel()
axs_counter = 0

for i in range(len(lc_names)):
    # Skip the bog and shallow and litteral classes due to very sparse
    # data points
    if (i == 7) | (i == 8) | (i == 9):
        continue
    print(f"Analyzing {lc_names[i]}")
    dlcc_tmp_clean = dlcc_clean.isel(LC=i)
    df = pd.DataFrame({
        "dlst": dlst_clean,
        "dalbedo": dalbedo_clean,
        "det": det_clean,
        "w": weights_clean,
        "dlcc": dlcc_tmp_clean
    })
    df = df.dropna()

    # Bin data based on dLCC
    dlcc_bins = np.linspace(-1.001, 1, 2002)
    out = fit_bins_ols(df=df, bins=dlcc_bins, var="det")

    # The equation for the mean model (i.e. gain/loss)
    if out[5] > 0:
        eq_text = (f"\u0394ET = {out[2]}" + "(\u00B1" + f"{out[3]})" + "+" +
                   f"{out[4]}(\u00B1" + f"{out[5]})" + "X")
    else:
        eq_text = (f"\u0394ET = {out[2]}" + "(\u00B1" + f"{out[3]})" +
                   f"{out[4]}(\u00B1" + f"{out[5]})" + "X")

    # Scatter plot of the gain/loss
    axs[axs_counter].scatter(x=out[0], y=out[1], color="gray")
    axs[axs_counter].plot(out[0],
                          out[6],
                          color='black',
                          linestyle="--",
                          linewidth=3,
                          label='Total gain and loss')
    axs[axs_counter].set_title(lc_names[i], fontsize=16)
    axs[axs_counter].tick_params(labelsize=16)
    axs[axs_counter].axvline(0, ls='--', c="k", linewidth=1)
    axs[axs_counter].text(0,
                          22,
                          eq_text,
                          horizontalalignment='center',
                          verticalalignment='center',
                          fontsize=8,
                          fontweight="bold",
                          color="black")
    axs[axs_counter].text(0.18,
                          -22,
                          "Gain",
                          ha="center",
                          va="center",
                          size=10,
                          color="black",
                          bbox=dict(boxstyle="rarrow,pad=0.03",
                                    fc="gray",
                                    ec="gray",
                                    alpha=0.5,
                                    lw=2))
    axs[axs_counter].text(-0.17,
                          -22,
                          "Loss",
                          ha="center",
                          va="center",
                          size=10,
                          color="black",
                          bbox=dict(boxstyle="larrow,pad=0.03",
                                    fc="gray",
                                    ec="gray",
                                    alpha=0.5,
                                    lw=2))
    axs[axs_counter].set_xlim(-1, 1)
    axs[axs_counter].set_ylim(-40, 40)

    # Now plot class transitions on top of the gain/loss
    for k in range(len(lc_names)):
        if (k == 7) | (k == 8) | (i == 9):
            continue
        if k == i:
            continue
        print(f"Transition to --> {lc_names[k]}")
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
        reg_weights = np.append(out_loss[10], out_gain[10])
        XX = sm.add_constant(X)
        mod_wls = sm.WLS(Y, XX, weights=reg_weights)
        res_wls = mod_wls.fit()



mod_wls = sm.OLS(Y, XX)
res_wls = mod_wls.fit()
res_wls.params

mod_wls = sm.WLS(Y, XX,weights=reg_weights)
res_wls = mod_wls.fit()
res_wls.params

plt.close()
plt.scatter(X,Y)
sns.regplot(x=X,y=Y)
plt.savefig(out_dir+"test.png")
plt.plot(reg_weights)
np.argmax(reg_weights)
min(reg_weights)
res_wls.params
        # mod_ols = sm.OLS(Y, XX)
        # res_ols = mod_ols.fit()
        predicts = res_ols.predict(XX)
        params = np.round(res_ols.params, 3)  # intercept, slope
        params_bse = np.round(res_ols.bse, 3)  # error on params
        pvalues = np.round(res_ols.pvalues, 3)  # pvalues of estimated params

        if ((np.min(X) > -0.5) | (np.max(X) < 0.5) | (pvalues[1] > 0.05) |
            (pvalues[0] > 0.05)):
            continue
        if params[1] > 0:
            eq_text_tmp = (f"\u0394ET={params[0]}" + "(\u00B1" +
                           f"{params_bse[0]})" + "+" + f"{params[1]}(\u00B1" +
                           f"{params_bse[1]})" + "X")
        else:
            eq_text_tmp = (f"\u0394ET={params[0]}" + "(\u00B1" +
                           f"{params_bse[0]})" + f"{params[1]}(\u00B1" +
                           f"{params_bse[1]})" + "X")
        axs[axs_counter].plot(X,
                              predicts,
                              color=palette[k],
                              linewidth=3,
                              label=lc_names[k])

    axs_counter += 1

# Sorting out the legend
labels_handles = {
    label: handle
    for ax in fig.axes
    for handle, label in zip(*ax.get_legend_handles_labels())
}
# Renaming legened items
labels_handles["Evergreen Forest (EF)"] = labels_handles["EF"]
labels_handles["Deciduous Forest (DF)"] = labels_handles["DF"]
del labels_handles["EF"]
del labels_handles["DF"]
# Reordering legend items
desired_legend_order = [
    "Total gain and loss", "Evergreen Forest (EF)", "Deciduous Forest (DF)",
    "Shrub", "Herbaceous", "Sparse", "Barren", "Fen"
]
reordered_labels_handles = {k: labels_handles[k] for k in desired_legend_order}
bottom_right_ax = axs[-1]
bottom_right_ax.clear()  # clears the random data I plotted previously
bottom_right_ax.set_axis_off()  # removes the XY axes
plt.legend(
    reordered_labels_handles.values(),
    reordered_labels_handles.keys(),
    # bbox_to_anchor=(1, 1),
    loc='center',
    prop={
        "family": "Times New Roman",
        "size": 14
    },
)
pltfont = {'fontname': 'Times New Roman'}
fig.supxlabel('Fractional change in land cover', fontsize=16, **pltfont)
fig.suptitle("C", x=0.02, y=0.85, size=30, **pltfont)
fig.supylabel("$\Delta ET$ [mm]", fontsize=16, **pltfont)
plt.tight_layout()
save(out_dir + "Fig3_ET_Gain_Loss.png")
# -------------------------------------------------------
