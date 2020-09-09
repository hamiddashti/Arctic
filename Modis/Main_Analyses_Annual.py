"""

This script is mainly for analyzing the results of the Max_Change_Analyses.py

"""
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

# import glob

# in_dir = '/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/LULC_Maximum_Change_Analyses/'
in_dir = "/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/"
out_dir = "/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/Test/"
fig_dir = out_dir + "Figures/Annual/"

# This is a netcdf file that shows all the extreme conversions calculated using the "Max_Change_Analyses.py"
conversions = xr.open_dataarray(
    in_dir + "LUC_Change_Extracted/LUC_max_conversions/conversions_Table.nc"
)

# Summing the conversion over the year dimension
print(conversions.sum("time"))

# Calling the results of the Max_Changed_Analyses.py
EF = xr.open_dataarray(
    in_dir + "LUC_Change_Extracted/LUC_max_conversions/EF_to_other.nc"
)
shrub = xr.open_dataarray(
    in_dir + "LUC_Change_Extracted/LUC_max_conversions/shrub_to_other.nc"
)
herb = xr.open_dataarray(
    in_dir + "LUC_Change_Extracted/LUC_max_conversions/herb_to_other.nc"
)
sparse = xr.open_dataarray(
    in_dir + "LUC_Change_Extracted/LUC_max_conversions/sparse_to_other.nc"
)
wetland = xr.open_dataarray(
    in_dir + "LUC_Change_Extracted/LUC_max_conversions/wetland_to_other.nc"
)
water = xr.open_dataarray(
    in_dir + "LUC_Change_Extracted/LUC_max_conversions/water_to_other.nc"
)
DF = xr.open_dataarray(
    in_dir + "LUC_Change_Extracted/LUC_max_conversions/DF_to_other.nc"
)

# The year DF_2007 is just two aribitraty pixels to show the changes between changed and unchaneged pixels (for presentation purposes)
DF_2007 = DF.sel(time=2007)
# DF_2007.to_netcdf('DF_2007.nc')

# importing LST (natural and LUC components)
lst_lulc = xr.open_dataarray(in_dir + "Natural_Variability_Annual_outputs/delta_lst_changed_lulc_component.nc")
lst_nv = xr.open_dataarray(in_dir + "Natural_Variability_Annual_outputs/delta_lst_changed_nv_component.nc")
lst_diff_total = xr.open_dataarray(in_dir + "Natural_Variability_Annual_outputs/delta_lst_total.nc")
lst = xr.open_dataarray(in_dir + "Data/Annual/Annual_LST/lst_mean_annual.nc") - 273.15
lst = lst.sel(year=slice("2003", "2014"))
lst = lst.rename({"lat": "y", "lon": "x"})

# Importing Albedo and taking the temporal difference
albedo = xr.open_dataarray(in_dir + "Data/Annual/Annual_Albedo/Albedo_annual.nc")
albedo = albedo.sel(year=slice("2003", "2014"))
albedo_diff = albedo.diff("year")

# Importing ET and taking the temporal difference
EC = xr.open_dataarray(in_dir + "Data/Annual/Annual_ET/EC.nc")
EI = xr.open_dataarray(in_dir + "Data/Annual/Annual_ET/EI.nc")
ES = xr.open_dataarray(in_dir + "Data/Annual/Annual_ET/ES.nc")
EW = xr.open_dataarray(in_dir + "Data/Annual/Annual_ET/EW.nc")
ET = xr.open_dataarray(in_dir + "Data/Annual/Annual_ET/ET.nc")
EC_diff = EC.diff("year")
EI_diff = EI.diff("year")
ES_diff = ES.diff("year")
EW_diff = EW.diff("year")
ET_diff = ET.diff("year")

# This is the map of water_energy limited areas
WL_EL = xr.open_dataarray(
    in_dir + "Data/Water_Energy_Limited/Tif/WE_5classes_reprojected.nc"
)


def find_coord(da, val, n):
    """
    Find a specific value in xarray object and returns its coordinate
    da --> xarray object
    val --> interested values to be find in da
    n --> index of a number if there are multiple of replicate of the val
    """
    tmp = da.where(da == val, drop=True)
    tmp_stack = tmp.stack(z=["x", "y"])
    a = tmp_stack[tmp_stack.notnull()][n]
    b = a.coords["z"].values
    x = b.tolist()[0]
    y = b.tolist()[1]
    return x, y


def plot_example(changed, not_changed, var, outname):
    changed.plot()
    not_changed.plot()
    plt.axvline(2003, color="k", linestyle="--")
    plt.axvline(2006, color="r", linestyle="--")
    plt.axvline(2007, color="r", linestyle="--")
    plt.xlabel("Time")
    plt.ylabel(var)
    plt.legend(["LUC changed", "LUC not changed"])
    plt.savefig(fig_dir + outname)
    plt.close()


def make_mask(data):
    # Filter data using np.isnan
    mask = ~np.isnan(data)
    filtered_data = [d[m] for d, m in zip(data.T, mask.T)]
    return filtered_data


""" ------------------------------------------------------------------
Analyzing an two example pixles where one the LUC (e.g. DF) is changed
and the other which is approximaltly close hasn't been changed. These pixles 
are arbitrary and just for show.  
-------------------------------------------------------------------"""
# Find the coordinates of a random pixel where DF changed to herbaceuous
x_changed, y_changed = find_coord(DF_2007, 4, 2)
# some arbitarary pixle nearby with no change in DF class (~94%)
x_not_changed = -121.6786
y_not_changed = 56.2286

lst_changed = lst.sel(x=x_changed, y=y_changed, method="nearest")
lst_not_changed = lst.sel(x=x_not_changed, y=y_not_changed, method="nearest")
plot_example(lst_changed, lst_not_changed, "LST [C]", outname="LST_DF_Herb_Example.png")

lst_lulc_changed = lst_lulc.sel(x=x_changed, y=y_changed, method="nearest")
lst_nv_changed = lst_nv.sel(x=x_changed, y=y_changed, method="nearest")

lst_diff_total_changed = lst_diff_total.sel(x=x_changed, y=y_changed, method="nearest")
lst_diff_total_not_changed = lst_diff_total.sel(
    x=x_not_changed, y=y_not_changed, method="nearest"
)
plot_example(
    lst_diff_total_changed,
    lst_diff_total_not_changed,
    r"$\Delta$ LST [C]",
    outname="LST_DF_Herb_trend_Example.png",
)

albedo_changed = albedo.sel(x=x_changed, y=y_changed, method="nearest")
albedo_not_changed = albedo.sel(x=x_not_changed, y=y_not_changed, method="nearest")
plot_example(
    albedo_changed, albedo_not_changed, "Albedo", outname="Albedo_DF_Herb_Example.png"
)

albedo_diff_changed = albedo_diff.sel(x=x_changed, y=y_changed, method="nearest")
albedo_diff_not_changed = albedo_diff.sel(
    x=x_not_changed, y=y_not_changed, method="nearest"
)
plot_example(
    albedo_diff_changed,
    albedo_diff_not_changed,
    r"$\Delta$ Albedo",
    outname="Albedo_DF_Herb_trend_Example.png",
)

ET_changed = ET.sel(x=x_changed, y=y_changed, method="nearest")
ET_not_changed = ET.sel(x=x_not_changed, y=y_not_changed, method="nearest")
plot_example(ET_changed, ET_not_changed, "ET [mm/year]", outname="ET.png")

EC_changed = EC.sel(x=x_changed, y=y_changed, method="nearest")
EC_not_changed = EC.sel(x=x_not_changed, y=y_not_changed, method="nearest")
plot_example(EC_changed, EC_not_changed, "EC [mm/year]", outname="EC.png")

ES_changed = ES.sel(x=x_changed, y=y_changed, method="nearest")
ES_not_changed = ES.sel(x=x_not_changed, y=y_not_changed, method="nearest")
plot_example(ES_changed, ES_not_changed, "ES [mm/year]", outname="ES.png")

EI_changed = EI.sel(x=x_changed, y=y_changed, method="nearest")
EI_not_changed = EI.sel(x=x_not_changed, y=y_not_changed, method="nearest")
plot_example(EI_changed, EI_not_changed, "EI [mm/year]", outname="EI.png")

""" ---------------------------------------------------------------------
Now lets do the same analyses for the entire region...
----------------------------------------------------------------------"""
def extract_vals(orig_class, val, var, conv_name):
    """
    var ---> Name of the varaibke we are interested to extract (lst/albedo/ET)
    val ---> Name of the class after convrsion of the original class (orig_class)
    orig_class --->  the xarray object of original class conversion from Max_Change_Anlyses.py
    conv_name ---> conversion name
    """
    import numpy as np

    if var == "lst":
        tmp_lst = lst_lulc.where(orig_class == val, drop=True).values
        tmp_lst = tmp_lst[~np.isnan(tmp_lst)]
        tmp_df = pd.DataFrame({conv_name: tmp_lst}, columns=[conv_name])
    elif var == "albedo":
        tmp_lst = lst_lulc.where(orig_class == val, drop=True).values
        tmp = albedo_diff.where(orig_class == val, drop=True).values
        tmp = tmp[~np.isnan(tmp_lst)]
        tmp_df = pd.DataFrame({conv_name: tmp}, columns=[conv_name])
    elif var == "ec":
        tmp_lst = lst_lulc.where(orig_class == val, drop=True).values
        tmp = EC_diff.where(orig_class == val, drop=True).values
        tmp = tmp[~np.isnan(tmp_lst)]
        tmp_df = pd.DataFrame({conv_name: tmp}, columns=[conv_name])
    elif var == "ei":
        tmp_lst = lst_lulc.where(orig_class == val, drop=True).values
        tmp = EI_diff.where(orig_class == val, drop=True).values
        tmp = tmp[~np.isnan(tmp_lst)]
        tmp_df = pd.DataFrame({conv_name: tmp}, columns=[conv_name])
    elif var == "es":
        tmp_lst = lst_lulc.where(orig_class == val, drop=True).values
        tmp = ES_diff.where(orig_class == val, drop=True).values
        tmp = tmp[~np.isnan(tmp_lst)]
        tmp_df = pd.DataFrame({conv_name: tmp}, columns=[conv_name])
    elif var == "ew":
        tmp_lst = lst_lulc.where(orig_class == val, drop=True).values
        tmp = EW_diff.where(orig_class == val, drop=True).values
        tmp = tmp[~np.isnan(tmp_lst)]
        tmp_df = pd.DataFrame({conv_name: tmp}, columns=[conv_name])
    elif var == "et":
        tmp_lst = lst_lulc.where(orig_class == val, drop=True).values
        tmp = ET_diff.where(orig_class == val, drop=True).values
        tmp = tmp[~np.isnan(tmp_lst)]
        tmp_df = pd.DataFrame({conv_name: tmp}, columns=[conv_name])
    return tmp_df


columns = [
    "EF_to_Shrub",
    #"EF_to_Herb",
    "EF_to_Sparse",
    # "DF_to_Shrub",
    "DF_to_Herb",
    "DF_to_Sparse",
    "Shrub_to_Sparse",
    #"Shrub_to_Wetland",
    "Herb_to_Shrub",
    "Herb_to_Sparse",
    #"Herb_to_Wetland",
    "Sparse_to_Shrub",
    "Sparse_to_Herb",
    "Wetland_to_Sparse",
    #"Water_to_Sparse",
    #"Water_to_Wetland",
]


tmp_lst = lst_lulc.where(EF == 3, drop=True).values
lst_EF_to_shrub = extract_vals(
    orig_class=EF, val=3, var="lst", conv_name="lst_EF_to_shrub"
)
# lst_EF_to_herb = extract_vals(
#     orig_class=EF, val=4, var="lst", conv_name="lst_EF_to_herb"
# )
lst_EF_to_sparse = extract_vals(
    orig_class=EF, val=5, var="lst", conv_name="lst_EF_to_sparse"
)

# lst_DF_to_shrub = extract_vals(DF, 3, "lst", "lst_DF_to_shrub")
lst_DF_to_herb = extract_vals(
    orig_class=DF, val=4, var="lst", conv_name="lst_DF_to_herb"
)
lst_DF_to_sparse = extract_vals(
    orig_class=DF, val=5, var="lst", conv_name="lst_DF_to_sparse"
)

lst_shrub_to_sparse = extract_vals(
    orig_class=shrub, val=5, var="lst", conv_name="lst_shrub_to_sparse"
)
# lst_shrub_to_wetland = extract_vals(
#     orig_class=shrub, val=6, var="lst", conv_name="lst_shrub_to_wetland"
# )

lst_herb_to_shrub = extract_vals(
    orig_class=herb, val=3, var="lst", conv_name="lst_herb_to_shrub"
)
lst_herb_to_sparse = extract_vals(
    orig_class=herb, val=5, var="lst", conv_name="lst_herb_to_sparse"
)
# lst_herb_to_wetland = extract_vals(
#     orig_class=herb, val=6, var="lst", conv_name="lst_herb_to_wetland"
# )

lst_sparse_to_shrub = extract_vals(
    orig_class=sparse, val=3, var="lst", conv_name="lst_sparse_to_shrub"
)
lst_sparse_to_herb = extract_vals(
    orig_class=sparse, val=4, var="lst", conv_name="lst_sparse_to_herb"
)
# lst_sparse_to_water = extract_vals(orig_class=sparse,val= 7,var= "lst",conv_name= "lst_sparse_to_water")

lst_wetland_to_sparse = extract_vals(
    orig_class=wetland, val=5, var="lst", conv_name="lst_wetland_to_sparse"
)

# lst_water_to_sparse = extract_vals(
#     orig_class=water, val=5, var="lst", conv_name="lst_water_to_sparse"
# )
# lst_water_to_wetland = extract_vals(
#     orig_class=water, val=6, var="lst", conv_name="lst_water_to_wetland"
# )

df_lst = pd.concat(
    [
        lst_EF_to_shrub,
        lst_EF_to_sparse,
        # lst_EF_to_herb,
        # lst_DF_to_shrub,
        lst_DF_to_herb,
        lst_DF_to_sparse,
        lst_shrub_to_sparse,
        # lst_shrub_to_wetland,
        # lst_herb_to_shrub,
        lst_herb_to_sparse,
        # lst_herb_to_wetland,
        lst_sparse_to_shrub,
        lst_sparse_to_herb,
        lst_wetland_to_sparse,
        # lst_water_to_sparse,
        # lst_water_to_wetland,
    ],
    ignore_index=True,
    axis=1,
)
df_lst.columns = columns

df_lst.boxplot(figsize=(16, 10))
plt.xticks(rotation=45, fontsize=16)
plt.title("LST", fontsize=20)
plt.ylabel("$\Delta$ LST [C]", fontsize=16)
plt.tight_layout()
plt.savefig(fig_dir + "LST_Boxplot.png")


albedo_EF_to_shrub = extract_vals(EF, 3, "albedo", "albedo_EF_to_shrub")
# albedo_EF_to_herb = extract_vals(EF, 4, "albedo", "albedo_EF_to_herb")
albedo_EF_to_sparse = extract_vals(EF, 5, "albedo", "albedo_EF_to_sparse")

# albedo_DF_to_shrub = extract_vals(DF, 3, "albedo", "albedo_DF_to_shrub")
albedo_DF_to_herb = extract_vals(DF, 4, "albedo", "albedo_DF_to_herb")
albedo_DF_to_sparse = extract_vals(DF, 5, "albedo", "albedo_DF_to_sparse")

albedo_shrub_to_sparse = extract_vals(shrub, 5, "albedo", "albedo_shrub_to_sparse")
# albedo_shrub_to_wetland = extract_vals(shrub, 6, "albedo", "albedo_shrub_to_wetland")

albedo_herb_to_shrub = extract_vals(herb, 3, "albedo", "albedo_herb_to_shrub")
albedo_herb_to_sparse = extract_vals(herb, 5, "albedo", "albedo_herb_to_sparse")
# albedo_herb_to_wetland = extract_vals(herb, 6, "albedo", "albedo_herb_to_wetland")

albedo_sparse_to_shrub = extract_vals(sparse, 3, "albedo", "albedo_sparse_to_shrub")
albedo_sparse_to_herb = extract_vals(sparse, 4, "albedo", "albedo_sparse_to_herb")

albedo_wetland_to_sparse = extract_vals(
    wetland, 5, "albedo", "albedo_wetland_to_sparse"
)

# albedo_water_to_sparse = extract_vals(water, 5, "albedo", "albedo_water_to_sparse")
# albedo_water_to_wetland = extract_vals(water, 6, "albedo", "albedo_water_to_wetland")

df_albedo = pd.concat(
    [
        albedo_EF_to_shrub,
        # albedo_EF_to_herb,
        albedo_EF_to_sparse,
        # albedo_DF_to_shrub,
        albedo_DF_to_herb,
        albedo_DF_to_sparse,
        albedo_shrub_to_sparse,
        # albedo_shrub_to_wetland,
        albedo_herb_to_shrub,
        albedo_herb_to_sparse,
        # albedo_herb_to_wetland,
        albedo_sparse_to_shrub,
        albedo_sparse_to_herb,
        albedo_wetland_to_sparse,
        # albedo_water_to_sparse,
        # albedo_water_to_wetland,
    ],
    ignore_index=True,
    axis=1,
)
df_albedo.columns = columns

df_albedo.boxplot(figsize=(16, 10))
plt.xticks(rotation=45, fontsize=16)
plt.title("Albedo", fontsize=20)
plt.ylabel("Albedo", fontsize=16)
plt.tight_layout()
plt.savefig(fig_dir + "Albedo_Boxplot.png")
plt.show()


ET_EF_to_shrub = extract_vals(EF, 3, "et", "ET_EF_to_shrub")
# ET_EF_to_herb = extract_vals(EF, 4, "et", "ET_EF_to_herb")
ET_EF_to_sparse = extract_vals(EF, 5, "et", "ET_EF_to_sparse")

# ET_DF_to_shrub = extract_vals(DF, 3, "et", "ET_DF_to_shrub")
ET_DF_to_herb = extract_vals(DF, 4, "et", "ET_DF_to_herb")
ET_DF_to_sparse = extract_vals(DF, 5, "et", "ET_DF_to_sparse")

ET_shrub_to_sparse = extract_vals(shrub, 5, "et", "ET_shrub_to_sparse")
# ET_shrub_to_wetland = extract_vals(shrub, 6, "et", "ET_shrub_to_wetland")

ET_herb_to_shrub = extract_vals(herb, 3, "et", "ET_herb_to_shrub")
ET_herb_to_sparse = extract_vals(herb, 5, "et", "ET_herb_to_sparse")
# ET_herb_to_wetland = extract_vals(herb, 6, "et", "ET_herb_to_wetland")

ET_sparse_to_shrub = extract_vals(sparse, 3, "et", "ET_sparse_to_shrub")
ET_sparse_to_herb = extract_vals(sparse, 4, "et", "ET_sparse_to_herb")

ET_wetland_to_sparse = extract_vals(wetland, 5, "et", "ET_wetland_to_sparse")

# ET_water_to_sparse = extract_vals(water, 5, "et", "ET_water_to_sparse")
# ET_water_to_wetland = extract_vals(water, 6, "et", "ET_water_to_wetland")

df_et = pd.concat(
    [
        ET_EF_to_shrub,
        # ET_EF_to_herb,
        ET_EF_to_sparse,
        # ET_DF_to_shrub,
        ET_DF_to_herb,
        ET_DF_to_sparse,
        ET_shrub_to_sparse,
        # ET_shrub_to_wetland,
        ET_herb_to_shrub,
        ET_herb_to_sparse,
        # ET_herb_to_wetland,
        ET_sparse_to_shrub,
        ET_sparse_to_herb,
        ET_wetland_to_sparse,
        # ET_water_to_sparse,
        # ET_water_to_wetland,
    ],
    ignore_index=True,
    axis=1,
)

df_et.columns = columns

df_et.boxplot(figsize=(16, 10))
plt.xticks(rotation=45, fontsize=16)
plt.title("ET", fontsize=20)
plt.ylabel("ET [mm $year^{-1}$]", fontsize=16)
plt.tight_layout()
plt.savefig(fig_dir + "ET_Boxplot.png")
plt.show()


fig, ax1 = plt.subplots(figsize=(12, 6))
widths = 0.3

# ax1.set_xlabel('Extrene conversions',fontsize=16)
ax1.set_ylabel("$\Delta$LST [C]", color="tab:orange", fontsize=16)
ax1.set_ylim(-15, 15)
ax1.yaxis.set_tick_params(labelsize=12)
# Filtering nan valuse for matplotlib boxplot
filtered_lst = make_mask(df_lst.values)
res1 = ax1.boxplot(
    filtered_lst,
    widths=widths,
    positions=np.arange(len(columns)) - 0.31,
    patch_artist=True,
)
for element in ["boxes", "whiskers", "fliers", "means", "medians", "caps"]:
    plt.setp(res1[element], color="k")
for patch in res1["boxes"]:
    patch.set_facecolor("tab:orange")

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel("$\Delta$Albedo", color="tab:blue", fontsize=16)
ax2.yaxis.set_tick_params(labelsize=12)
filtered_albedo = make_mask(df_albedo.values)
res2 = ax2.boxplot(
    filtered_albedo,
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
ax3.spines["right"].set_position(("axes", 1.1))
ax3.set_frame_on(True)
ax3.patch.set_visible(False)
ax3.set_ylabel("$\Delta$ET", color="tab:green", fontsize=16)
ax3.yaxis.set_tick_params(labelsize=12)
filtered_et = make_mask(df_et.values)
res3 = ax3.boxplot(
    filtered_et,
    positions=np.arange(len(columns)) + 0.31,
    widths=widths,
    patch_artist=True,
)
##from https://stackoverflow.com/a/41997865/2454357
for element in ["boxes", "whiskers", "fliers", "means", "medians", "caps"]:
    plt.setp(res3[element], color="k")
for patch in res3["boxes"]:
    patch.set_facecolor("tab:green")

ax1.set_xlim([-0.55, 14.55])
ax1.set_xticks(np.arange(len(columns)))
ax1.set_xticklabels(columns, rotation=50, fontsize=11)
ax1.yaxis.grid(True, which="major")

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(fig_dir + "Grouped_Boxplot.png")
plt.show()

"""---------------------------------------------------------------------------

					Analyzing energy vs. water limited 

# Here we restircted the analyses to the EF ---> Sparse since we don't have much data for other conversions
------------------------------------------------------------------------------"""

class_name = ["High_WL", "Moderate_WL", "Low_WL", "Low_EL", "High_EL"]


def ELvsWL(var, orig_class, val, WL_EL_class):
    if var == "lst":
        try:
            tmp_lst = lst_lulc.where(
                (orig_class == val) & (WL_EL == WL_EL_class), drop=True
            ).values
            tmp = tmp_lst[~np.isnan(tmp_lst)]
            tmp_df = pd.DataFrame({class_name[WL_EL_class - 1]: tmp})
        except ValueError:
            # tmp_df = pd.DataFrame({class_name[WL_EL_class-1]: np.nan})
            tmp_df = pd.DataFrame(
                np.nan, index=[0], columns=[class_name[WL_EL_class - 1]]
            )
    # 	return tmp_df
    if var == "albedo":
        try:
            tmp_albedo = albedo_diff.where(
                (orig_class == val) & (WL_EL == WL_EL_class), drop=True
            ).values
            tmp = tmp_albedo[~np.isnan(tmp_albedo)]
            tmp_df = pd.DataFrame({class_name[WL_EL_class - 1]: tmp})
        except ValueError:
            # tmp_df = pd.DataFrame({class_name[WL_EL_class-1]: np.nan})
            tmp_df = pd.DataFrame(
                np.nan, index=[0], columns=[class_name[WL_EL_class - 1]]
            )
    # 	return tmp_df

    if var == "et":
        try:
            tmp_et = ET_diff.where(
                (orig_class == val) & (WL_EL == WL_EL_class), drop=True
            ).values
            tmp = tmp_et[~np.isnan(tmp_et)]
            tmp_df = pd.DataFrame({class_name[WL_EL_class - 1]: tmp})
        except ValueError:
            # tmp_df = pd.DataFrame({class_name[WL_EL_class-1]: np.nan})
            tmp_df = pd.DataFrame(
                np.nan, index=[0], columns=[class_name[WL_EL_class - 1]]
            )
    # 	return tmp_df
    return tmp_df


lst_ef_EL_WL = []
albedo_ef_EL_WL = []
et_ef_EL_WL = []
for i in np.arange(1, 6):
    print(i)
    df_lst = ELvsWL(var="lst", orig_class=EF, val=5, WL_EL_class=i)
    df_albedo = ELvsWL(var="albedo", orig_class=EF, val=5, WL_EL_class=i)
    df_et = ELvsWL(var="et", orig_class=EF, val=5, WL_EL_class=i)
    lst_ef_EL_WL.append(pd.Series(df_lst[class_name[i - 1]], name=class_name[i - 1]))
    albedo_ef_EL_WL.append(
        pd.Series(df_albedo[class_name[i - 1]], name=class_name[i - 1])
    )
    et_ef_EL_WL.append(pd.Series(df_et[class_name[i - 1]], name=class_name[i - 1]))

lst_ef_EL_WL = pd.concat(lst_ef_EL_WL, axis=1)
albedo_ef_EL_WL = pd.concat(albedo_ef_EL_WL, axis=1)
et_ef_EL_WL = pd.concat(et_ef_EL_WL, axis=1)

fig, ax1 = plt.subplots(figsize=(16, 8))
widths = 0.3
# ax1.set_xlabel('Extrene conversions',fontsize=16)
ax1.set_ylabel("$\Delta$LST [C]", color="tab:orange", fontsize=16)
ax1.set_ylim(-15, 15)
ax1.yaxis.set_tick_params(labelsize=12)
# Filtering nan valuse for matplotlib boxplot
filtered_lst_ef_EL_WL = make_mask(lst_ef_EL_WL.values)
res1 = ax1.boxplot(
    filtered_lst_ef_EL_WL,
    widths=widths,
    positions=np.arange(len(class_name)) - 0.31,
    patch_artist=True,
)
for element in ["boxes", "whiskers", "fliers", "means", "medians", "caps"]:
    plt.setp(res1[element], color="k")
for patch in res1["boxes"]:
    patch.set_facecolor("tab:orange")

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel("$\Delta$Albedo", color="tab:blue", fontsize=16)
ax2.yaxis.set_tick_params(labelsize=12)
filtered_albedo_ef_EL_WL = make_mask(albedo_ef_EL_WL.values)
res2 = ax2.boxplot(
    filtered_albedo_ef_EL_WL,
    positions=np.arange(len(class_name)) + 0,
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
ax3.spines["right"].set_position(("axes", 1.08))
ax3.set_frame_on(True)
ax3.patch.set_visible(False)
ax3.set_ylabel("$\Delta$ET", color="tab:green", fontsize=16)
ax3.yaxis.set_tick_params(labelsize=12)
ax3.set_ylim(-850, 850)
filtered_et_ef_EL_WL = make_mask(et_ef_EL_WL.values)
res3 = ax3.boxplot(
    filtered_et_ef_EL_WL,
    positions=np.arange(len(class_name)) + 0.31,
    widths=widths,
    patch_artist=True,
)
##from https://stackoverflow.com/a/41997865/2454357
for element in ["boxes", "whiskers", "fliers", "means", "medians", "caps"]:
    plt.setp(res3[element], color="k")
for patch in res3["boxes"]:
    patch.set_facecolor("tab:green")

ax1.set_xlim([-0.55, 4.55])
ax1.set_xticks(np.arange(len(class_name)))
ax1.set_xticklabels(class_name, rotation=30, fontsize=11)
ax1.yaxis.grid(True, which="major")

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(fig_dir + "EF_Sparse_EL_WL.png")
plt.show()
