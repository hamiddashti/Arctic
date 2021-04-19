import xarray as xr
import matplotlib.pylab as plt
import numpy as np
from matplotlib.pylab import savefig as save
import pandas as pd
from sklearn.linear_model import LinearRegression
import sklearn
import statsmodels.api as sm
import seaborn as sns


def outliers_index(data, m=3):
    # Return true if a value is outlier
    d = np.abs(data - np.nanmedian(data))
    mdev = np.nanmedian(d)
    s = d / mdev if mdev else 0.
    return ~(s < m)


def binmean(data, val_col, weight_col):
    mean = np.nansum(data[val_col] * data[weight_col]) / np.nansum(
        data[weight_col])
    return mean


def weighted_mean(lc, var, thresh, weights, type):
    w = weights.where(var.notnull())
    if type == "gain":
        var_tmp = var.where(lc > thresh)
        w = w.where(lc > thresh)
    if type == "lost":
        var_tmp = var.where(lc < -thresh)
        w = w.where(lc < -thresh)
    mean = (var_tmp * w).sum() / w.sum()
    n = w.notnull().sum()
    return n, mean


def bins_mean(dvar, dlc, thresh, w, type):
    ''' 
    Calculate weighted mean for gain/lost and extremes
    '''
    import pandas as np
    import numpy as np
    if type == "lost":
        dvar_tmp = dvar.where(dlc < -thresh)
        w_tmp = w.where(dlc < -thresh)
        dlc_tmp = dlc.where(dlc < -thresh)
        dlc_bins = np.linspace(-1, -0.5, 51)
    if type == "gain":
        dvar_tmp = dvar.where(dlc > thresh)
        w_tmp = w.where(dlc > thresh)
        dlc_tmp = dlc.where(dlc > thresh)
        dlc_bins = np.linspace(0.5, 1, 51)
    df = pd.DataFrame({
        "dvar_tmp": dvar_tmp,
        "w_tmp": w_tmp,
        "dlc_tmp": dlc_tmp
    })
    df = df.dropna()
    df["bins"] = pd.cut(df["dlc_tmp"], bins=dlc_bins, right=False)
    bins_mean = df.groupby('bins').apply(myfunc, 'dvar_tmp', 'w_tmp')
    wsums = df.groupby("bins").apply(lambda d: d["w_tmp"].sum())
    counts = df.groupby("bins").count()["dlc_tmp"]
    total_mean = (bins_mean * wsums).sum() / wsums.sum()
    return counts, total_mean, bins_mean


def fit_bins_ols(df, bins, var):
    df["bins"] = pd.cut(df["dlcc"], bins=bins, include_lowest=True)
    bins_mean = df.groupby('bins').apply(binmean, var, 'w')
    # counts = df.groupby('bins').count()["dlcc"]
    # bins_mean = bins_mean.where(counts > 10)
    wsums = df.groupby("bins").apply(lambda d: d["w"].sum())
    x = bins[1:][bins_mean.notnull()]
    y = bins_mean[bins_mean.notnull()].values
    sample_weight = wsums[bins_mean.notnull()].values
    X = sm.add_constant(x)
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


in_dir = ("/data/home/hamiddashti/nasa_above/outputs/")
out_dir = "/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/test/"

dlst_xr = xr.open_dataarray(
    in_dir + "Natural_Variability/"
    "Natural_Variability_Annual_outputs/EndPoints/all_bands/dlst_mean_lcc.nc")

ct = xr.open_dataset(
    in_dir + "Sensitivity/EndPoints/Annual/all_bands/Confusion_Table_final.nc")
weights = ct["WEIGHTS"]
dlst = ct["DLST_MEAN_LCC"]
det = ct["DET_LCC"]
dalbedo = ct["DALBEDO_LCC"]
dlcc = ct["DLCC"]
confusion = ct["CONFUSION"]
idx = ct["PIX_INDEX"]
normalized_confusion = ct["NORMALIZED_CONFUSION"]
I_dlst = outliers_index(dlst, 3)
I_dalbedo = outliers_index(dalbedo, 3)
I_det = outliers_index(det, 3)

dlst_clean = dlst.where(I_dlst == False)
dalbedo_clean = dalbedo.where(I_dalbedo == False)
det_clean = det.where(I_det == False)
weights_clean = weights.where(I_dlst == False)
dlcc_clean = dlcc.where(I_dlst == False)
normalized_confusion_clean = normalized_confusion.where(I_dlst == False)

palette = sns.color_palette("tab10")
Total_Number_Pixels = dlst_xr.shape[0] * dlst_xr.shape[1]

lc_names = [
    "EF", "DF", "Shrub", "Herbaceous", "Sparse", "Barren", "Fen", "Bog",
    "Shallow_Littoral", "water"
]

df_stats = pd.DataFrame(data=None,
                        index=["\u0394LST[K]", "\u0394Albedo", "\u0394ET[mm]"],
                        columns=["Max", "Min", "StD"])
df_stats.loc["\u0394LST[K]", "Max"] = np.round(dlst_clean.max().values, 3)
df_stats.loc["\u0394LST[K]", "Min"] = np.round(dlst_clean.min().values, 3)
df_stats.loc["\u0394LST[K]", "StD"] = np.round(dlst_clean.std().values, 3)
df_stats.loc["\u0394Albedo", "Max"] = np.round(dalbedo_clean.max().values, 3)
df_stats.loc["\u0394Albedo", "Min"] = np.round(dalbedo_clean.min().values, 3)
df_stats.loc["\u0394Albedo", "StD"] = np.round(dalbedo_clean.std().values, 3)
df_stats.loc["\u0394ET[mm]", "Max"] = np.round(det_clean.max().values, 3)
df_stats.loc["\u0394ET[mm]", "Min"] = np.round(det_clean.min().values, 3)
df_stats.loc["\u0394ET[mm]", "StD"] = np.round(det_clean.std().values, 3)

with open(out_dir + "results.txt", "w") as f:
    f.write("This file contains the results of the annual data analyses\n")
    f.write("----------------------------------------------------------\n\n")
    f.write(
        f"Total number of changed pixels (modis scale): {len(dlst_clean)}\n")
    f.write(f"Total number of pixels (modis scale) covering the study area: \
{Total_Number_Pixels}\n")
    f.write(f"Percent change: \
{np.round((len(dlst_clean)/Total_Number_Pixels)*100,3)}%\n\n")

with open(out_dir + "results.txt", "a") as f:
    f.write(df_stats.to_string(header=True, index=True))

plt.close()
fig, axs = plt.subplots(2, 4, figsize=(15, 6), facecolor='w', edgecolor='k')
axs = axs.ravel()
axs_counter = 0
for i in range(10):
    # Skip the bog and shallow and litteral classes
    if (i == 7) | (i == 8):
        continue
    print(i)
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
    out = fit_bins_ols(df=df, bins=dlcc_bins, var="dlst")
    eq_text = (f"\u0394LST = {out[3]}" + "(\u00B1" + f"{out[4]})" + "\u00D7" +
               f"{out[5]}(\u00B1" + f"{out[6]})" + "X")

    percent_change_subclass = np.round((len(df) / Total_Number_Pixels) * 100,
                                       3)

    with open(out_dir + "results.txt", "a") as f:
        f.write(f"\n\nResults of {lc_names[i]} analyses\n")
        f.write("------------------------------")
        f.write(
            f"\n\nPercent change in {lc_names[i]}:{percent_change_subclass}%\n")
        f.write(f"\nLinear model fit to {lc_names[i]} gain/lost:\n")
        f.write(eq_text)
        f.write(
            f"\nn:{len(out[0])}. NOTE Data are binned, refer to the manuscript\n"
        )
        f.write(f"p-value intercept: {np.round(out[8][0],3)}\n")
        f.write(f"p-value slope: {np.round(out[8][1],3)}\n")

    axs[axs_counter].scatter(x=out[0], y=out[1], color="gray")
    axs[axs_counter].plot(out[0],
                          out[7],
                          color='black',
                          linestyle="--",
                          linewidth=3,
                          label='Mean model')
    axs[axs_counter].title.set_text(lc_names[i])
    axs[axs_counter].tick_params(labelsize=14)
    axs[axs_counter].axvline(0, ls='--', c="k", linewidth=1)
    axs[axs_counter].text(0,
                          max(out[1]) - 0.1,
                          eq_text,
                          horizontalalignment='center',
                          verticalalignment='center',
                          fontsize=8,
                          fontweight="bold",
                          color="black")

    if i == 0:
        trans_list = [2, 4]
    elif i == 1:
        trans_list = [3, 6]
    elif i == 2:
        trans_list = [1, 3, 4, 6]
    elif i == 3:
        trans_list = [0, 9]
    elif i == 4:
        trans_list = [0, 2, 4, 5, 6]
    elif i == 5:
        trans_list = [0, 4, 9]
    elif i == 6:
        trans_list = [0, 1]
    elif i == 9:
        trans_list = [2, 3, 5]

    # for k in trans_list:
    for k in range(10):
        if (k == 7) | (k == 8):
            continue
        print(lc_names[k])
        if k == i:
            continue

        transintion_lost = normalized_confusion_clean[:, i, k]
        df_lost = pd.DataFrame({
            "dlst": dlst_clean,
            "w": weights_clean,
            "dlcc": -transintion_lost
        })
        df_lost = df_lost.dropna()
        bins_lost = np.linspace(-1, 0, 1001)
        df_lost["bins"] = pd.cut(df_lost["dlcc"],
                                 bins=bins_lost,
                                 include_lowest=True)
        out_lost = fit_bins_ols(df=df_lost, bins=bins_lost, var="dlst")
        transintion_gain = normalized_confusion_clean[:, k, i]
        df_gain = pd.DataFrame({
            "dlst": dlst_clean,
            "w": weights_clean,
            "dlcc": transintion_gain,
        })
        df_gain = df_gain.dropna()
        bins_gain = np.linspace(0, 1, 1001)
        out_gain = fit_bins_ols(df=df_gain, bins=bins_gain, var="dlst")
        X = np.append(out_lost[0], out_gain[0])
        Y = np.append(out_lost[1], out_gain[1])
        reg_weights = np.append(out_lost[2], out_gain[2])
        XX = sm.add_constant(X)
        mod_wls = sm.WLS(Y, XX, weights=reg_weights)
        res_wls = mod_wls.fit()
        predicts = res_wls.predict(XX)
        params = np.round(res_wls.params, 3)
        params_bse = np.round(res_wls.bse, 3)
        pvalues = np.round(res_wls.pvalues, 3)
        eq_text_tmp = (f"\u0394LST={params[0]}" + "(\u00B1" +
                       f"{params_bse[0]})" + "\u00D7" + f"{params[1]}(\u00B1" +
                       f"{params_bse[1]})" + "X")
        with open(out_dir+"results.txt","a") as f:
            f.write(f"\n\nLinear model between {lc_names[i]}-{lc_names[k]}:\n")
            f.write(eq_text_tmp)
            f.write(f"\nn: {len(XX)}\n")
            f.write(f"p-value intercept: {pvalues[0]}\n")
            f.write(f"p-value slope: {pvalues[1]}\n")
            
        axs[axs_counter].plot(X,
                              predicts,
                              color=palette[k],
                              linewidth=3,
                              label=lc_names[k])

        axs[axs_counter].text(0.18,
                              out[1].min() + 0.2,
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
                              out[1].min() + 0.2,
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
    axs_counter += 1

handles, labels = axs[0].get_legend_handles_labels()
plt.legend(handles, labels, bbox_to_anchor=(1, 1))
fig.supxlabel('Fractional change in land cover', fontsize=16)
fig.supylabel("$\Delta LST$ [k]", fontsize=16)
plt.tight_layout()
save(out_dir + "test.png")