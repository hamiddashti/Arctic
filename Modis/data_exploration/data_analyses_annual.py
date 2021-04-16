import xarray as xr
import matplotlib.pylab as plt
import numpy as np
from matplotlib.pylab import savefig as save
import pandas as pd
from sklearn.linear_model import LinearRegression
import sklearn
import statsmodels.api as sm


def outliers_index(data, m=3):
    # Return true if a value is outlier
    d = np.abs(data - np.nanmedian(data))
    mdev = np.nanmedian(d)
    s = d / mdev if mdev else 0.
    return ~(s < m)


def myfunc(data, val_col, weight_col):
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


in_dir = ("/data/home/hamiddashti/nasa_above/outputs/")
out_dir = "/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/test/"

dlcc = xr.open_dataarray(
    in_dir + "Natural_Variability/"
    "Natural_Variability_Annual_outputs/EndPoints/all_bands/dlcc.nc")
dlst_xr = xr.open_dataarray(
    in_dir + "Natural_Variability/"
    "Natural_Variability_Annual_outputs/EndPoints/all_bands/dlst_mean_lcc.nc")
# I = outliers_index(dlst, 5)
# dlst_clean = dlst.where(I == False)
ct = xr.open_dataset(
    in_dir + "Sensitivity/EndPoints/Annual/all_bands/Confusion_Table_final.nc")

lc = dlcc.isel(band=0)
I = outliers_index(dlst_xr, 3)
dlst_clean = dlst_xr.where(I == False)
lc_clean = lc.where(I == False)
dlst_clean2 = dlst_clean.where(lc.notnull() == True)
#Xarray weighting
weights = np.cos(np.deg2rad(dlst_clean2.lat))
weights.name = "weights"
dlst_clean_weighted = dlst_clean.weighted(weights)
dlst_clean_weighted.mean(("lat", "lon"))

# Binning
weights = ct["WEIGHTS"]
dlst = ct["DLST_MEAN_LCC"]
I = outliers_index(dlst, 3)
dlst_clean = dlst.where(I == False)
weights_clean = weights.where(I == False)
total_mean_all_categories = np.round(
    (dlst_clean * weights_clean).sum() / weights_clean.sum(), 3)

lc_names = [
    "EF", "DF", "Shrub", "Herb", "Sparse", "Barren", "Fen", "Bog",
    "Shallow_Littoral", "water"
]

for i in range(10):
    print(i)
    weights = ct["WEIGHTS"]
    dlst = ct["DLST_MEAN_LCC"]
    dlc_tmp = ct["DLCC"].isel(LC=i)
    I = outliers_index(dlst, 5)
    dlst_clean = dlst.where(I == False)
    weights_clean = weights.where(I == False)
    dlc_tmp_clean = dlc_tmp.where(I == False)

    # dlc_bins = np.linspace(-1.0001, 1, 20002)
    dlc_bins = np.linspace(-1.001, 1, 2002)

    df = pd.DataFrame({
        "dlst": dlst_clean,
        "w": weights_clean,
        "dlc": dlc_tmp_clean
    })
    df = df.dropna()
    df["bins"] = pd.cut(df["dlc"], bins=dlc_bins)
    bins_mean = df.groupby('bins').apply(myfunc, 'dlst', 'w')
    # counts = df.groupby("bins").count()["dlc"]
    wsums = df.groupby("bins").apply(lambda d: d["w"].sum())
    total_mean_each_lc = (bins_mean * wsums).sum() / wsums.sum()

    x = dlc_bins[:-1][bins_mean.notnull()]
    y = bins_mean[bins_mean.notnull()].values
    sample_weight = wsums[bins_mean.notnull()].values

    X = sm.add_constant(x)
    mod_wls = sm.WLS(y, X, weights=sample_weight)
    res_wls = mod_wls.fit()
    intercept, slope = np.round(res_wls.params, 3)
    intercept_std, slope_std = np.round(res_wls.bse, 3)
    eq_text = (f"$\Delta LST$ = {intercept}" + "\u00B1" + f"{intercept_std}" +
               r"$\times$" + f"{slope} \u00B1" + f"{slope_std}" + "X")
    plt.close()
    fig, ax = plt.subplots(figsize=(5, 4))
    # plt.plot(x, regr.predict(x), color='red', linewidth=1, label='Weighted model')
    plt.plot(x,
             res_wls.predict(X),
             color='black',
             linewidth=3,
             label='Weighted model')
    # plt.plot(x, m*x + b,color="r")
    plt.scatter(dlc_bins[:-1], bins_mean, color="gray")
    plt.title(lc_names[i], fontsize=14, fontweight="bold")
    ax.tick_params(labelsize=12)
    ax.axvline(0, ls='--', c="k", linewidth=1)
    plt.text(0,
             2.2,
             eq_text,
             horizontalalignment='center',
             verticalalignment='center',
             fontsize=12,
             fontweight="bold",
             color="black")
    plt.tight_layout()
    outname = str(lc_names[i]) + "_5.png"
    save(out_dir + outname)

weights = ct["WEIGHTS"]
dlst = ct["DLST_MEAN_LCC"]
I = outliers_index(dlst, 5)
dlst_clean = dlst.where(I == False)
weights_clean = weights.where(I == False)

# Calculate gain lost mean
total_mean_lost_list = []
total_mean_lost_list_n = []
total_mean_gain_list = []
total_mean_gain_list_n = []
for i in range(10):
    print(i)
    dlc_tmp = ct["DLCC"].isel(LC=i)
    dlc_tmp_clean = dlc_tmp.where(I == False)

    n_lost, total_mean_lost = weighted_mean(dlc_tmp_clean, dlst_clean, 0.5,
                                            weights_clean, "lost")
    n_gain, total_mean_gain = weighted_mean(dlc_tmp_clean, dlst_clean, 0.5,
                                            weights_clean, "gain")
    total_mean_lost_list.append(total_mean_lost)
    total_mean_gain_list.append(total_mean_gain)
    total_mean_lost_list_n.append(n_lost)
    total_mean_gain_list_n.append(n_gain)

lost_counts, total_lost_mean, bins_lost_mean = bins_mean(dvar=dlst_clean,
                                                         dlc=lc_tmp_clean,
                                                         thresh=0.5,
                                                         w=weights_clean,
                                                         type="lost")
gain_counts, total_gain_mean, bins_gain_mean = bins_mean(dvar=dlst_clean,
                                                         dlc=lc_tmp_clean,
                                                         thresh=0.5,
                                                         w=weights_clean,
                                                         type="gain")

np.round(np.array(total_mean_lost_list), 3)
np.round(np.array(total_mean_gain_list), 3)
np.array(total_mean_lost_list_n)
np.array(total_mean_gain_list_n)
# Calculate the gain lost based on binning

bins_lost_mean.reset_index(drop=True, inplace=True)
bins_gain_mean.reset_index(drop=True, inplace=True)
df_concat = pd.concat([bins_lost_mean, bins_gain_mean], axis=1)
# ----------------------------------

weights = ct["WEIGHTS"]
dlst = ct["DLST_MEAN_LCC"]
dlc_tmp = ct["DLCC"].isel(LC=5)
I = outliers_index(dlst, 3)
dlst_clean = dlst.where(I == False)
weights_clean = weights.where(I == False)
dlc_tmp_clean = dlc_tmp.where(I == False)

var_tmp = dlst_clean.where(dlc_tmp_clean > 0.5)
w = weights_clean.where(dlc_tmp_clean > 0.5)
(var_tmp * w).sum() / w.sum()

var_tmp = dlst_clean.where(dlc_tmp_clean < -0.5)
w = weights_clean.where(dlc_tmp_clean < -0.5)
(var_tmp * w).sum() / w.sum()

w = weights_clean.where(dlst_clean.notnull())
if type == "gain":
    var_tmp = dlst_clean.where(dlc_tmp_clean > 0.5)
    w = w.where(dlc_tmp_clean > 0.5)
if type == "lost":
    var_tmp = dlst_clean.where(dlc_tmp_clean < -0.5)
    w = w.where(dlc_tmp_clean < -0.5)

(var_tmp * w).sum() / w.sum()
w.notnull().sum()
