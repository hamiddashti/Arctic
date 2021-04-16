import xarray as xr
import matplotlib.pylab as plt
import numpy as np
from pylab import savefig as save
import pandas as pd
import seaborn as sns

# def reject_outliers(data, m):
# 	# m is number of std
# 	import numpy as np
# 	data = data.astype(float)
# 	data[abs(data - np.nanmean(data)) > m * np.nanstd(data)] = np.nan
# 	return data

def outliers_index(data, m=3):
    # Return true if a value is outlier
    d = np.abs(data - np.nanmedian(data))
    mdev = np.nanmedian(d)
    s = d / mdev if mdev else 0.
    return ~(s < m)


in_dir = ("/data/home/hamiddashti/nasa_above/outputs/Sensitivity/EndPoints/"
"Annual/all_bands/")
out_dir = "/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/test/"

ECO_NAMES = [
    "EF", "DF", "Shrub", "Herb", "Sparse", "Barren", "Fen", "Bog", "SL",
    "water"
]
MIN_THRESH = 15  # Minimum number of pixels for plotting boxplots
EXT_THRESH = 0.5
ds = xr.open_dataset(in_dir + 'Confusion_Table.nc')
conversions = ds['Conversion'].values

gain_list = []
for k in range(0, len(ECO_NAMES)):
    gain = [i for i in conversions if i.endswith(ECO_NAMES[k])]
    del gain[k]
    gain_list.append(gain)

lost_list = []
for k in range(0, len(ECO_NAMES)):
    lost = [i for i in conversions if i.startswith(ECO_NAMES[k])]
    del lost[k]
    lost_list.append(lost)


cnt = ds["NORMALIZED_CONFUSION"]
dlst = ds['DLST_MEAN_LCC']
I = outliers_index(dlst,5)
dlst_clean = dlst[~I]
cnt_clean = cnt[~I,:]

ef_lost = cnt_clean.sel(Conversion=lost_list[3])
ef_gain = cnt_clean.sel(Conversion=gain_list[3])


(ef_gain>0.5).any(axis=1)


dlst_clean[(ef_lost>0.5).any(axis=1)].count()
dlst_clean[(ef_gain>0.5).any(axis=1)].count()


dlst_clean[(ef_gain>0.5).any(axis=1)]
a=ef_gain.where(ef_gain>0.5).any("Conversion")



I_zeros = (ef_lost == 0).all(axis=1)
ef_lost_nozero = ef_lost[~I_zeros,:]

a=ef_lost.where(ef_lost>0.5).any("Conversion")

ef_lost[0]


I_extreme = np.vstack(np.argwhere(ef_lost.values > EXT_THRESH))


plt.close()
plt.hist(ef_lost[])
save(out_dir+"test1.png")









l_list = []
for k in range(0, len(ECO_NAMES)):
    print(ECO_NAMES[k])
    # Get the name of all classes rather than class[k]
    tmp_lc = ECO_NAMES[:k] + ECO_NAMES[k + 1:]
    cnt_lost = cnt.sel(Conversion=lost_list[k])
    #Remove rows where all columns are zero
    I_zeros = (cnt_lost == 0).all(axis=1)
    cnt_lost_nonzero = cnt_lost[~I_zeros, :]

    # Find the extreme ecosystem transition (LCC>EXT_THRESH)
    I_extreme = np.vstack(np.argwhere(cnt_lost_nonzero.values > EXT_THRESH))
    cnt_lost_extreme = cnt_lost_nonzero[I_extreme[:, 0], :]
    lc_lost_extreme = []
    # find the class after lcc
    for i_ext in I_extreme:
        lc_lost_extreme.append(tmp_lc[i_ext[1]])
    lc_lost_extreme = np.array(lc_lost_extreme)

    # Find the asosciated dLST
    dlst_lost_nonzero = dlst[~I_zeros]
    dlst_lost_extreme = dlst_lost_nonzero[I_extreme[:, 0]]

    # Remove outliers based on dLST
    I_outliers = outliers_index(dlst_lost_extreme)
    cnt_lost_clean = cnt_lost_extreme[~I_outliers, :]
    dlst_lost_clean = dlst_lost_extreme[~I_outliers]
    lc_lost_clean = lc_lost_extreme[~I_outliers.values]

    tmp_lost_df = pd.DataFrame({
        "dlst": dlst_lost_clean,
        "changed_cover": lc_lost_clean,
        "gain_lost": ["lost"] * len(dlst_lost_clean),
        "LC": [ECO_NAMES[k]] * len(dlst_lost_clean)
    })
    cnt_gain = cnt.sel(Conversion=gain_list[k])
    #Remove rows where all columns are zero
    I_zeros = (cnt_gain == 0).all(axis=1)
    cnt_gain_nonzero = cnt_gain[~I_zeros, :]
    # Find the extreme ecosystem transition
    I_extreme = np.vstack(np.argwhere(cnt_gain_nonzero.values > EXT_THRESH))
    cnt_gain_extreme = cnt_gain_nonzero[I_extreme[:, 0], :]

    lc_gain_extreme = []
    for i_ext in I_extreme:
        tmp_lc_gain_extreme = tmp_lc[i_ext[1]]
        lc_gain_extreme.append(tmp_lc_gain_extreme)
    lc_gain_extreme = np.array(lc_gain_extreme)

    # Find the asosciated dLST
    dlst_gain_nonzero = dlst[~I_zeros]
    dlst_gain_extreme = dlst_gain_nonzero[I_extreme[:, 0]]
    # Remove outliers based on LST
    I_outliers = outliers_index(dlst_gain_extreme)
    cnt_gain_clean = cnt_gain_extreme[~I_outliers, :]
    dlst_gain_clean = dlst_gain_extreme[~I_outliers]
    lc_gain_clean = lc_gain_extreme[~I_outliers.values]
    tmp_gain_df = pd.DataFrame({
        "dlst": dlst_gain_clean,
        "changed_cover": lc_gain_clean,
        "gain_lost": ["gain"] * len(dlst_gain_clean),
        "LC": [ECO_NAMES[k]] * len(dlst_gain_clean)
    })

    # Make sure data has at least 10 points
    if (len(tmp_lost_df) >= 10):
        l_list.append(tmp_lost_df)
    if (len(tmp_gain_df) >= 10):
        l_list.append(tmp_gain_df)

df_concat = pd.concat(l_list)
final_pd = df_concat.groupby(["LC", "gain_lost",
                              "changed_cover"]).filter(lambda x: len(x) > 10)

gain_lost_count = final_pd.groupby("gain_lost").count()
gain_lost_mean = final_pd.groupby("gain_lost").mean()
gain_count = gain_lost_count.loc["gain"]["dlst"]
lost_count = gain_lost_count.loc["lost"]["dlst"]

total_mean = np.round(final_pd['dlst'].mean(), 4)
gain_mean = np.round(gain_lost_mean.loc["gain"]["dlst"], 4)
lost_mean = np.round(gain_lost_mean.loc["lost"]["dlst"], 4)

with open(out_dir + "restults2.txt", "w") as txt_f:
    txt_f.write(
        "Results of annual extreme LCC (0.5> land cover change) and LST:\n")
    txt_f.write("--------------------------------\n")
    txt_f.write(f"Total number of gained pixels: {gain_count}\n")
    txt_f.write(f"Total number of lost pixels: {lost_count}\n")
    txt_f.write(f'Mean of \u0394LST over all changed pixels: {total_mean}\n')
    txt_f.write(f'Mean of \u0394LST in gained pixels: {gain_mean}\n')
    txt_f.write(f'Mean of \u0394LST lost pixels: {lost_mean}\n\n')

lost_gain_groups_mean = final_pd.groupby(["LC", "gain_lost"],
                                         sort=False).mean()
meanL1 = np.squeeze(np.round(lost_gain_groups_mean.values, 2))

lost_gain_groups_median = final_pd.groupby(["LC", "gain_lost"],
                                           sort=False).median()
medianL1 = np.squeeze(np.round(lost_gain_groups_median.values, 2))

lost_gain_groups_std = final_pd.groupby(["LC", "gain_lost"], sort=False).std()
stdL1 = np.squeeze(np.round(lost_gain_groups_std.values, 2))

with open(out_dir + "restults2.txt", "a") as txt_f:
    txt_f.write("Mean of lost and gain for each groups\n")
    txt_f.write(lost_gain_groups_mean.to_string())

mean_labels = []
for i in range(0, len(meanL1)):
    mean_labels.append(str(meanL1[i]) + u"\u00B1" + str(stdL1[i]))

# Boxplot of LST changes based on net gain and lost
plt.close()
fsize = 6
fig, ax = plt.subplots(figsize=(3.5, 2.4))
sns.set_style("white")
box_plot = sns.boxplot(ax=ax,
                       x="LC",
                       y="dlst",
                       linewidth=1,
                       hue="gain_lost",
                       data=final_pd,
                       palette="husl")
ind = 0
for tick in range(0, len(box_plot.get_xticklabels())):
    box_plot.text(
        tick - .2,
        4.1,
        #   medianL1[ind],
        mean_labels[ind],
        horizontalalignment='center',
        color='black',
        fontsize=3.8,
        weight='semibold')
    box_plot.text(
        tick + .2,
        3.75,
        #   medianL1[ind + 1],
        mean_labels[ind + 1],
        horizontalalignment='center',
        color='black',
        fontsize=3.8,
        weight='semibold')
    ind += 2
plt.xlabel("")
plt.ylabel("$\Delta$LST [K]", fontsize=fsize)
plt.legend(loc='best', prop={'size': 5})
ax.tick_params(labelsize=fsize)
ax.set_xticklabels(ECO_NAMES)
ax.axhline(0, ls='--', c="k", linewidth=0.5)
plt.tight_layout()
save(out_dir + "annual_gain_lost_extreme.png", dpi=600)

final_pd_gain = final_pd.loc[final_pd["gain_lost"] == "gain"]
plt.close()
fsize = 6
fig, ax = plt.subplots(figsize=(3.2, 2.4))
sns.set_style("white")
box_plot = sns.boxplot(ax=ax,
                       x="LC",
                       y="dlst",
                       hue_order=ECO_NAMES,
                       hue="changed_cover",
                       linewidth=1,
                       fliersize=1,
                       data=final_pd_gain,
                       palette="tab10")
plt.xlabel("")
plt.title("Gain")
plt.ylabel("$\Delta$LST [K]", fontsize=fsize)
plt.legend(loc='best', prop={'size': 5})
ax.axhline(0, ls='--', c="k", linewidth=0.5)
ax.tick_params(labelsize=fsize)
ax.set_xticklabels(ECO_NAMES)
plt.tight_layout()
save(out_dir + "annual_gain_extremes.png", dpi=600)

final_pd_lost = final_pd.loc[final_pd["gain_lost"] == "lost"]
plt.close()
fsize = 6
fig, ax = plt.subplots(figsize=(3.2, 2.4))
sns.set_style("white")
box_plot = sns.boxplot(ax=ax,
                       x="LC",
                       y="dlst",
                       hue_order=ECO_NAMES,
                       hue="changed_cover",
                       linewidth=1,
                       fliersize=1,
                       data=final_pd_lost,
                       palette="tab10")
plt.xlabel("")
plt.title("Lost")
plt.ylabel("$\Delta$LST [K]", fontsize=fsize)
plt.legend(loc='best', prop={'size': 5})
ax.axhline(0, ls='--', c="k", linewidth=0.5)
ax.tick_params(labelsize=fsize)
ax.set_xticklabels(ECO_NAMES)
plt.tight_layout()
save(out_dir + "annual_lost_extremes.png", dpi=600)
# ------------------------------------------------------------

cnt = ds["NORMALIZED_CONFUSION"]

dlst = ds['DLST_MEAN_LCC']
dlst.shape
conversions = ds['Conversion'].values
cnt1 = cnt.values
dlst1 = dlst.values
diagnoal_list = np.array([0, 8, 16, 24, 32, 40, 48])
I = np.delete(np.arange(49), diagnoal_list)
conversions = conversions[I]
cnt1 = cnt1[:, I]
lst = dlst1[~outliers_index(dlst, 2)]
lcc = cnt1[~outliers_index(dlst, 2), :]

EXT_THRESH = 0.50
ave = []
for i in range(len(conversions)):
    tmp = lcc[:, i]
    lst_conv = lst[np.where(tmp > EXT_THRESH)]
    if len(lst_conv) < 15:
        ave.append(np.nan)
        continue
    ave_tmp = np.round(np.mean(lst_conv), 2)
    ave.append(ave_tmp)
    print(conversions[i] + f"--> ({len(lst_conv)},{ave_tmp})")

ave_final = np.array(np.round(ave, 2)).reshape(7, 7)
df = pd.DataFrame(data=ave_final, index=ECO_NAMES, columns=ECO_NAMES)
df.to_string(out_dir + "test.txt")

conversions[3]
tmp = lcc[:, 3]
I = tmp > EXT_THRESH

cnt2 = lcc[I, :]
lst_conv1 = lst[np.where(I)]
a = np.mean(cnt2, axis=0)
a[np.argmax(a)]

conversions[28]
tmp = lcc[:, 28]
lst_conv2 = lst[np.where(tmp > EXT_THRESH)]
np.mean(lst_conv1)

plt.close()
sns.boxplot(data=[lst_conv1, lst_conv2])
save(out_dir + "test.png")
