#
"""----------------------------------------------------------------
This script is for ET, Albedo and LST of different landcover types
- pixels should be covered by a land cover more than 98% (consistent
with our 2% for calculating natural variability)
----------------------------------------------------------------"""
import xarray as xr
import matplotlib.pylab as plt
import numpy as np
from matplotlib.pylab import savefig as save
import pandas as pd


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


def make_mask(data):
    # Filter data using np.isnan
    mask = ~np.isnan(data)
    filtered_data = [d[m] for d, m in zip(data.T, mask.T)]
    return filtered_data


def prepare_data(lc, et, albedo, lst, lc_names):
    df_lst = pd.DataFrame(columns=lc_names)
    df_albedo = pd.DataFrame(columns=lc_names)
    df_et = pd.DataFrame(columns=lc_names)
    for i in range(len(lc_names)):
        print(i)
        lst_tmp = lst.where(lc.isel(band=i) >= 0.5)
        et_tmp = et.where(lc.isel(band=i) >= 0.5)
        albedo_tmp = albedo.where(lc.isel(band=i) >= 0.5)
        I_lst = outliers_index(lst_tmp)
        I_et = outliers_index(et_tmp)
        I_albedo = outliers_index(albedo_tmp)
        lst_tmp = lst_tmp.where((I_lst == False) & (I_et == False)
                                & (I_albedo == False))
        df_lst[lc_names[i]] = lst_tmp.values.ravel()
        et_tmp = et_tmp.where((I_lst == False) & (I_et == False)
                              & (I_albedo == False))
        df_et[lc_names[i]] = et_tmp.values.ravel()
        albedo_tmp = albedo_tmp.where((I_lst == False) & (I_et == False)
                                      & (I_albedo == False))
        df_albedo[lc_names[i]] = albedo_tmp.values.ravel()
    df_et = df_et.dropna(how="all")
    df_albedo = df_albedo.dropna(how="all")
    df_lst = df_lst.dropna(how="all")

    df_final_lst = {
        "name": "LST",
        "df": df_lst,
        "label": "LST [K]",
        "ylim": [230, 322],
    }

    df_final_et = {
        "name": "ET",
        "df": df_et,
        "label": "ET [mm]",
        "ylim": [0, 650],
    }

    df_final_albedo = {
        "name": "Albedo",
        "df": df_albedo,
        "label": "Albedo",
        "ylim": [0, 1],
    }
    return (df_final_lst, df_final_et, df_final_albedo)


def myboxplot_group(df1, df2, df3, title, columns, txt_pos, outname):
    """ boxplot of multiple pandas dataframe
    """
    plt.close()
    fig, ax1 = plt.subplots(figsize=(12, 6))
    widths = 0.3
    pltfont = {'fontname': 'Times New Roman'}
    df1_mean = np.round(df1["df"].mean().values, 2)
    df1_sd = np.round(df1["df"].std().values, 2)
    df2_mean = np.round(df2["df"].mean().values, 2)
    df2_sd = np.round(df2["df"].std().values, 2)
    df3_mean = np.round(df3["df"].mean().values, 2)
    df3_sd = np.round(df3["df"].std().values, 2)
    ax1.set_ylabel(df1["label"], color="tab:orange", fontsize=16)
    ax1.set_ylim(df1["ylim"])
    ax1.yaxis.set_tick_params(labelsize=13)
    n = df1["df"].notnull().sum()
    # Filtering nan valuse for matplotlib boxplot
    filtered_df1 = make_mask(df1["df"].values)
    res1 = ax1.boxplot(
        filtered_df1,
        widths=widths,
        positions=np.arange(len(columns)) - 0.31,
        patch_artist=True,
    )
    for element in ["boxes", "whiskers", "fliers", "means", "medians", "caps"]:
        plt.setp(res1[element], color="k")
    for patch in res1["boxes"]:
        patch.set_facecolor("tab:orange")

    # Here we add the mean and std information to the plot
    pos = range(len(df1_mean))

    # a = np.arange(0, len(columns)) % 2
    # txt_pos = np.where(a == 0, 7, 9)
    counter = 0
    for tick, label in zip(pos, ax1.get_xticklabels()):
        ax1.text(
            pos[tick],
            # q[tick] + (q[tick]-lst_mean[tick])/2,
            txt_pos[counter],
            df1["name"] + " = " + str(df1_mean[tick]) + "$\pm$" +
            str(df1_sd[tick]) + "\n" + df2["name"] + " = " +
            str(df2_mean[tick]) + "$\pm$" + str(df2_sd[tick]) + "\n" +
            df3["name"] + " = " + str(df3_mean[tick]) + "$\pm$" +
            str(df3_sd[tick]) + "\n" + "n" + " = " + str(n[tick]),
            horizontalalignment="center",
            fontsize=10.5,
            color="k",
            weight="semibold",
            **pltfont)
        counter = counter + 1
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylim(df2["ylim"])
    ax2.set_ylabel(df2["label"], color="tab:blue", fontsize=16, **pltfont)
    ax2.yaxis.set_tick_params(labelsize=13)
    filtered_df2 = make_mask(df2["df"].values)
    res2 = ax2.boxplot(
        filtered_df2,
        positions=np.arange(len(columns)) + 0,
        widths=widths,
        patch_artist=True,
    )
    ##from https://stackoverflow.com/a/41997865/2454357
    for element in ["boxes", "whiskers", "fliers", "means", "medians", "caps"]:
        plt.setp(res2[element], color="k")
    for patch in res2["boxes"]:
        patch.set_facecolor("tab:blue")
    # To make the border of the right-most axis visible, we need to turn the frame
    # on. This hides the other plots, however, so we need to turn its fill off.
    ax3 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax3.set_ylim(df3["ylim"])
    ax3.spines["right"].set_position(("axes", 1.1))
    ax3.set_frame_on(True)
    ax3.patch.set_visible(False)
    ax3.set_ylabel(df3["label"], color="tab:green", fontsize=16, **pltfont)
    ax3.yaxis.set_tick_params(labelsize=13)
    filtered_df3 = make_mask(df3["df"].values)
    res3 = ax3.boxplot(
        filtered_df3,
        positions=np.arange(len(columns)) + 0.31,
        widths=widths,
        patch_artist=True,
    )
    ##from https://stackoverflow.com/a/41997865/2454357
    for element in ["boxes", "whiskers", "fliers", "means", "medians", "caps"]:
        plt.setp(res3[element], color="k")
    for patch in res3["boxes"]:
        patch.set_facecolor("tab:green")
    ax1.set_xlim([-0.55, len(columns) - 0.25])
    ax1.set_xticks(np.arange(len(columns)))
    ax1.set_xticklabels(columns, fontsize=12, fontweight="semibold", **pltfont)
    ax1.yaxis.grid(False)
    ax1.axhline(color="k")
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title(title, fontsize=13, fontweight="semibold", **pltfont)
    plt.savefig(out_dir + outname)
    plt.close()


in_dir = "/data/ABOVE/Final_data/"
out_dir = ("/data/home/hamiddashti/nasa_above/outputs/data_analyses/Annual/"
           "Albers/Figures_MS1/")
# out_dir = "/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/test/"
lc_names = ["EF", "DF", "Shrub", "Herbaceous", "Sparse", "Barren", "Fen"]
luc = xr.open_dataarray(in_dir + "LUC/albers/LULC_10_2003_2014.nc")
lst_mean = xr.open_dataset(in_dir + (
    "LST_Final/LST/Annual_Mean/albers/albers_proj_lst_mean_Annual.nc"))
lst_mean = lst_mean["lst_mean_Annual"]
lst_mean = lst_mean.rename({"x": "lon", "y": "lat"})
albedo = xr.open_dataarray(
    in_dir + "ALBEDO_Final/Annual_Albedo/albers/final_annual_albedo.nc")
albedo = albedo.rename({"x": "lon", "y": "lat"})
et = xr.open_dataset(in_dir +
                     "ET_Final/Annual_ET/albers/albers_proj_ET_Annual.nc")
et = et["ET_Annual"]
et = et.rename({"x": "lon", "y": "lat"})

lc_2003 = luc.loc[2003]
lst_mean_2003 = lst_mean.loc[2003]
albedo_2003 = albedo.loc[2003]
et_2003 = et.loc[2003]

lc_2013 = luc.loc[2013]
lst_mean_2013 = lst_mean.loc[2013]
albedo_2013 = albedo.loc[2013]
et_2013 = et.loc[2013]

a = np.arange(0, len(lc_names)) % 2
txt_pos = np.where(a == 0, 300, 310)

df_final_lst, df_final_et, df_final_albedo = prepare_data(lc=lc_2003,
                                                          et=et_2003,
                                                          albedo=albedo_2003,
                                                          lst=lst_mean_2003,
                                                          lc_names=lc_names)
myboxplot_group(df_final_lst, df_final_et, df_final_albedo, "2003", lc_names,
                txt_pos, "Boxplot_groups_2003.png")

df_final_lst, df_final_et, df_final_albedo = prepare_data(lc=lc_2013,
                                                          et=et_2013,
                                                          albedo=albedo_2013,
                                                          lst=lst_mean_2013,
                                                          lc_names=lc_names)
myboxplot_group(df_final_lst, df_final_et, df_final_albedo, "2013", lc_names,
                txt_pos, "Boxplot_groups_2013.png")
