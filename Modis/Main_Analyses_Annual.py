"""

This script is mainly for analyzing the results of the Max_Change_Analyses.py
The main goal is to plot the changes in ET, albedo and LST following a major change in LUC

There are seven classes in general which in this scriot is numbered from 1 to 7 where numbers are: 
1: EF (Evergreen forest); 2:DF (Decisuous forest); 3:shrub; 
4: Herbacious; 5:sparse; 6:wetland; and 7:water 

"""
# Loading the libraries
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

'''---------------------------------------------------------
							Parameters
---------------------------------------------------------'''
analyses_mode = "Growing" 
LUC_Type = [
	"Evergreen Forest",
	"Deciduous Forest", 
	"Shrub",
	"Herbaceous",
	"Sparse",
	"Wetland",
	"Water"
	]

'''---------------------------------------------------------
							Prepare data
---------------------------------------------------------'''
print('----------------- Preparing data -----------------\n')
in_dir = "/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/Data/"
out_dir = "/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/outputs/"
fig_dir = out_dir + "Main_Max_Change/Figures/"+analyses_mode+"/"

if analyses_mode=="Growing":
	print("Analyses mode is set to Growing, meaning we are working on the growing season (Apr-Nov; seven months)\n")
elif analyses_mode=="Annual":
	print("Analyses mode is set to Annual, meaning we are working on the annual data (Jan-Dec)\n")
print(f"Input directory:{in_dir}")
print(f"Output directory:{out_dir}")
print(f"Figure directory:{fig_dir}\n")

print("Working on LST...")
# importing LST (natural and LUC components)
lst_lulc = xr.open_dataarray(
	out_dir
	+ "Natural_Variability/Natural_Variability_"+analyses_mode+"_outputs/delta_lst_changed_lulc_component.nc"
)

lst_nv = xr.open_dataarray(
	out_dir
	+ "Natural_Variability/Natural_Variability_"+analyses_mode+"_outputs/delta_lst_changed_nv_component.nc"
)
lst_diff_total = xr.open_dataarray(
	out_dir
	+ "Natural_Variability/Natural_Variability_"+analyses_mode+"_outputs/delta_lst_total.nc"
)
lst = (
	xr.open_dataarray(in_dir +analyses_mode+"/"+analyses_mode+"_LST/lst_mean_"+analyses_mode+".nc") - 273.15
)  # kelvin to C
lst = lst.sel(year=slice("2003", "2014"))
lst = lst.rename({"lat": "y", "lon": "x"})

print("Working on albedo...")
albedo = xr.open_dataarray(in_dir + analyses_mode+"/"+analyses_mode+"_Albedo/Albedo_"+analyses_mode+".nc")
albedo = albedo.sel(year=slice("2003", "2014"))
albedo_diff = albedo.diff("year")

print("Working on ET and its components...\n")
EC = xr.open_dataarray(in_dir + analyses_mode+"/"+analyses_mode+"_ET/EC_"+analyses_mode+".nc") # Vegetation transpiration
EI = xr.open_dataarray(in_dir + analyses_mode+"/"+analyses_mode+"_ET/EI_"+analyses_mode+".nc") # Vegetation transpiration
ES = xr.open_dataarray(in_dir + analyses_mode+"/"+analyses_mode+"_ET/ES_"+analyses_mode+".nc") # Vegetation transpiration
EW = xr.open_dataarray(in_dir + analyses_mode+"/"+analyses_mode+"_ET/EW_"+analyses_mode+".nc") # Vegetation transpiration
ET = xr.open_dataarray(in_dir + analyses_mode+"/"+analyses_mode+"_ET/ET_"+analyses_mode+".nc") # Vegetation transpiration
EC = EC.fillna(0)
EI = EI.fillna(0)
ES = ES.fillna(0)
EW = EW.fillna(0)

ECI = EC + EI  # canopy evapotranspiration
ESW = ES + EW  # soil/water/ice/snow evaporation

EC = EC.where(EC != 0)  # Convert zeros to nan
EI = EI.where(EI != 0)
ES = ES.where(ES != 0)
EW = EW.where(EW != 0)
ET = ET.where(ET != 0)
ECI = ECI.where(ECI != 0)
ESW = ESW.where(ESW != 0)
#Take the difference in ET 
EC_diff = EC.diff("year")
EI_diff = EI.diff("year")
ES_diff = ES.diff("year")
EW_diff = EW.diff("year")
ET_diff = ET.diff("year")
ECI_diff = ECI.diff("year")
ESW_diff = ESW.diff("year")

# EW.isel(year=1).to_netcdf(out_dir+"test2_ew.nc")
# This is a netcdf file that shows all the extreme conversions calculated using the "Max_Change_Analyses.py"
conversions = xr.open_dataarray(
	out_dir + "LUC_Change_Extracted/LUC_max_conversions/conversions_Table.nc"
)

conversions_sum = conversions.sum("year")
conversions_sum = pd.DataFrame(data = conversions_sum.values,index = LUC_Type,columns=LUC_Type)

with open(out_dir +analyses_mode+"_report.txt", "w") as text_file:
    print("Number of pixles undergone extreme conversion. Rows/columns are\
 classes before/after conversion (Table XXXX in the paper):",
        file=text_file,
    )
    print(conversions_sum, file=text_file)
text_file.close()
print("Number of pixles undergone extreme conversion. Rows/columns are\
 classes before/after conversion (Table XXXX in the paper):")
print(conversions_sum)

# Calling the results of the Max_Changed_Analyses.py
EF = xr.open_dataarray(
	out_dir + "LUC_Change_Extracted/LUC_max_conversions/EF_to_other.nc"
)
shrub = xr.open_dataarray(
	out_dir + "LUC_Change_Extracted/LUC_max_conversions/shrub_to_other.nc"
)
herb = xr.open_dataarray(
	out_dir + "LUC_Change_Extracted/LUC_max_conversions/herb_to_other.nc"
)
sparse = xr.open_dataarray(
	out_dir + "LUC_Change_Extracted/LUC_max_conversions/sparse_to_other.nc"
)
wetland = xr.open_dataarray(
	out_dir + "LUC_Change_Extracted/LUC_max_conversions/wetland_to_other.nc"
)
water = xr.open_dataarray(
	out_dir + "LUC_Change_Extracted/LUC_max_conversions/water_to_other.nc"
)
DF = xr.open_dataarray(
	out_dir + "LUC_Change_Extracted/LUC_max_conversions/DF_to_other.nc"
)

# This is the map of water_energy limited areas with two classes
WL_EL = xr.open_dataarray(
	in_dir + "Water_Energy_Limited/Tif/WL_EL_Reclassified_reproject.nc"
)
WL_EL = WL_EL.squeeze() 

'''----------------------------------------------------------------------------
					Functions used in various part of teh script
----------------------------------------------------------------------------'''
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
	# plot the example pixel (for presentations only)
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
	elif var == "eci":
		tmp_lst = lst_lulc.where(orig_class == val, drop=True).values
		tmp = ECI_diff.where(orig_class == val, drop=True).values
		tmp = tmp[~np.isnan(tmp_lst)]
		tmp_df = pd.DataFrame({conv_name: tmp}, columns=[conv_name])
	elif var == "esw":
		tmp_lst = lst_lulc.where(orig_class == val, drop=True).values
		tmp = ESW_diff.where(orig_class == val, drop=True).values
		tmp = tmp[~np.isnan(tmp_lst)]
		tmp_df = pd.DataFrame({conv_name: tmp}, columns=[conv_name])
	return tmp_df
def myboxplot(df, title, ylabel, margin, outname):
	df_mean = np.round(df.mean().values, 2)
	df_sd = np.round(df.std().values, 2)
	q = df.quantile(q=0.75).values
	ax = df.boxplot(figsize=(16, 10))
	pos = range(len(df_mean))
	for tick, label in zip(pos, ax.get_xticklabels()):
		ax.text(
			pos[tick] + 1,
			q[tick] + margin,
			str(df_mean[tick]) + "$\pm$" + str(df_sd[tick]),
			horizontalalignment="right",
			fontsize=12,
			color="k",
			weight="bold",
		)

	plt.xticks(rotation=45, fontsize=16)
	plt.title(title, fontsize=20)
	plt.ylabel(ylabel, fontsize=16)
	plt.tight_layout()
	plt.savefig(fig_dir + outname)
	plt.close()

def myboxplot_group(df1, df2, df3, columns, txt_pos, outname):

	fig, ax1 = plt.subplots(figsize=(16, 8))
	widths = 0.3
	df1_mean = np.round(df1["df"].mean().values, 2)
	df1_sd = np.round(df1["df"].std().values, 2)
	df2_mean = np.round(df2["df"].mean().values, 2)
	df2_sd = np.round(df2["df"].std().values, 2)
	df3_mean = np.round(df3["df"].mean().values, 2)
	df3_sd = np.round(df3["df"].std().values, 2)

	ax1.set_ylabel(df1["label"], color="tab:orange", fontsize=16)
	ax1.set_ylim(df1["ylim"])
	ax1.yaxis.set_tick_params(labelsize=12)
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
	for tick, label in zip(pos, ax1.get_xticklabels()):
		ax1.text(
			pos[tick],
			# q[tick] + (q[tick]-lst_mean[tick])/2,
			txt_pos,
			df1["name"]
			+ " = "
			+ str(df1_mean[tick])
			+ "$\pm$"
			+ str(df1_sd[tick])
			+ "\n"
			+ df2["name"]
			+ " = "
			+ str(df2_mean[tick])
			+ "$\pm$"
			+ str(df2_sd[tick])
			+ "\n"
			+ df3["name"]
			+ " = "
			+ str(df3_mean[tick])
			+ "$\pm$"
			+ str(df3_sd[tick])
			+ "\n",
			horizontalalignment="center",
			fontsize=8,
			color="k",
			weight="semibold",
		)
	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
	ax2.set_ylim(df2["ylim"])
	ax2.set_ylabel(df2["label"], color="tab:blue", fontsize=16)
	ax2.yaxis.set_tick_params(labelsize=12)
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
	ax3.set_ylabel(df3["label"], color="tab:green", fontsize=16)
	ax3.yaxis.set_tick_params(labelsize=12)
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
	ax1.set_xticklabels(columns, rotation=50, fontsize=11)
	ax1.yaxis.grid(False)
	ax1.axhline(color="k")
	fig.tight_layout()  # otherwise the right y-label is slightly clipped
	plt.savefig(fig_dir + outname)
	plt.close()

def ELvsWL(var, orig_class, val, WL_EL_class):
	'''
	var --------->	Name of variable to be extracted (lst, albedo)
	orig_class -->	Name of the conversion (EF, DF, shrub etc)
	val --------->	Class number after conversion of the orig_class
	WL_EL_Class -> 1:Water limited; 2:water limited 

	'''
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
def extract_wl_el(orig_class, val):
	lst_EL_WL = []
	albedo_EL_WL = []
	et_EL_WL = []
	for i in np.arange(1, 3):
		print(i)

		df_lst = ELvsWL(var="lst", orig_class=orig_class, val=val, WL_EL_class=i)

		# if number of pixles are less than 25 (25% of the 100 minimum)
		# we ignore the EL_WL analysis in the LUC conversion
		if len(df_lst) < 25:
			print(
				f"There is not enough pixels in the {class_name[i - 1]} area to make robust conclusions"
			)
			return

		else:
			df_albedo = ELvsWL(
				var="albedo", orig_class=orig_class, val=val, WL_EL_class=i
			)
			df_et = ELvsWL(var="et", orig_class=orig_class, val=val, WL_EL_class=i)
			lst_EL_WL.append(
				pd.Series(df_lst[class_name[i - 1]], name=class_name[i - 1])
			)
			albedo_EL_WL.append(
				pd.Series(df_albedo[class_name[i - 1]], name=class_name[i - 1])
			)
			et_EL_WL.append(pd.Series(df_et[class_name[i - 1]], name=class_name[i - 1]))

	lst_EL_WL = pd.concat(lst_EL_WL, axis=1)
	albedo_EL_WL = pd.concat(albedo_EL_WL, axis=1)
	et_EL_WL = pd.concat(et_EL_WL, axis=1)
	return lst_EL_WL, albedo_EL_WL, et_EL_WL

""" ---------------------------------------------------------------------
Regional analyses: The main criteria is that a LUC should include more 
than 50 pixels that we can make robust conclusions 
----------------------------------------------------------------------"""

print('\n----------------- Working on the entire region (takes several minutes/hours) -----------------\n')
columns = [
	"EF_to_Shrub",
	# "EF_to_Herb", 		# We ignore this conversions since the number of pixels including this conversion is less than 50
	"EF_to_Sparse",
	# "DF_to_Shrub", 		# We ignore this conversions since the number of pixels including this conversion is less than 50
	"DF_to_Herb",
	"DF_to_Sparse",
	"Shrub_to_Sparse",
	# "Shrub_to_Wetland",	# We ignore this conversions since the number of pixels including this conversion is less than 50
	"Herb_to_Shrub",
	"Herb_to_Sparse",
	# "Herb_to_Wetland",	# We ignore this conversions since the number of pixels including this conversion is less than 50
	"Sparse_to_Shrub",
	"Sparse_to_Herb",
	"Wetland_to_Sparse",
	# "Water_to_Sparse",	# We ignore this conversions since the number of pixels including this conversion is less than 50
	# "Water_to_Wetland",	# We ignore this conversions since the number of pixels including this conversion is less than 50
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
		lst_herb_to_shrub,
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
print(f"saving LST_Boxplot.png in {fig_dir}")
myboxplot(
	df=df_lst,
	title="LST",
	ylabel="$\Delta$ LST [C]",
	margin=0.3,
	outname="LST_Boxplot.png",
)

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
print(f"saving Albedo_Boxplot.png in {fig_dir}")
myboxplot(
	df=df_albedo,
	title="Albedo",
	ylabel="$\Delta$ Albedo",
	margin=0.005,
	outname="Albedo_boxplot.png",
)

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
print(f"saving ET_Boxplot.png in {fig_dir}")
myboxplot(
	
	df=df_et,
	title="ET",
	ylabel="$\Delta$ ET [mm/year]",
	margin=5,
	outname="ET_boxplot.png",
)

df1 = {
	"name": "LST",
	"df": df_lst,
	"label": "$\Delta$LST [C]",
	"ylim": [-12, 12]
	}
df2 = {
	"name": "Albedo",
	"df": df_albedo,
	"label": "$\Delta$Albedo",
	"ylim": [-0.5, 0.5],
	}
df3 = {
	"name": "ET",
	"df": df_et, 
	"label": "$\Delta$ET [mm/year]", 
	"ylim": [-850, 850]
	}
print(f"saving Boxplot_groups.png in {fig_dir}")
myboxplot_group(df1, df2, df3, columns=columns, txt_pos=9, outname="Boxplot_groups.png")

df_lst_mean = df_lst.mean()
df_albedo_mean = df_albedo.mean()
df_et_mean = df_et.mean()
df_lst_std = df_lst.std()
df_albedo_std = df_albedo.std()
df_et_std = df_et.std()

frames = pd.concat([df_lst_mean,df_albedo_mean,df_et_mean,df_lst_std,df_albedo_std,df_et_std],axis=1)
frames.columns = ["LST Mean","Albedo Mean","ET Mean","LST STD","Albedo STD","ET STD"]

with open(out_dir +analyses_mode+"_report.txt", "a") as text_file:
    print("\n Mean and STD of the LST, albedo and ET aftr LUC conversion:",
        file=text_file,
    )
    print(frames, file=text_file)
text_file.close()
print("Mean and STD of the LST, albedo and ET after LUC conversion:")
print(frames)

"""---------------------------------------------------------------------------
					Analyzing energy vs. water limited 
------------------------------------------------------------------------------"""

print('----------------- Working on the energy vs water limited -----------------\n')
# class_name = ["High_WL", "Moderate_WL", "Low_WL", "Low_EL", "High_EL"]
class_name = ["Water_Limited", "Energy_limited"]

print("Extracting EF to Sparse energy vs. water limited LST, albedo and ET\n")
# EF to sparse conversion which we have enough data for EL_EL analyses
lst_EL_WL, albedo_EL_WL, et_EL_WL = extract_wl_el(
	orig_class=EF, val=5
)  # There are enough data
df1 = {"name": "LST", "df": lst_EL_WL, "label": "$\Delta$LST [C]", "ylim": [-12, 12]}
df2 = {
	"name": "Albedo",
	"df": albedo_EL_WL,
	"label": "$\Delta$Albedo",
	"ylim": [-0.5, 0.5],
}
df3 = {
	"name": "ET",
	"df": et_EL_WL,
	"label": "$\Delta$ET [mm/year]",
	"ylim": [-850, 850],
}
print(f"saving EF_sparse_EL_WL.png in {fig_dir}")
myboxplot_group(
	df1, df2, df3, columns=class_name, txt_pos=9, outname="EF_sparse_EL_WL.png"
)
lst_mean=lst_EL_WL.mean()
albedo_mean=albedo_EL_WL.mean()
et_mean=et_EL_WL.mean()
lst_std=lst_EL_WL.std()
albedo_std=albedo_EL_WL.std()
et_std=et_EL_WL.std()

ef_sparse_el_wl_df = pd.concat([lst_mean,albedo_mean,et_mean,lst_std,albedo_std,et_std],axis=1)
ef_sparse_el_wl_df.columns = ['LST Mean','Albedo Mean','ET Mean','LST STD','Albedo STD','ET STD']

with open(out_dir +analyses_mode+"_report.txt", "a") as text_file:
    print("\n Mean and STD of water vs. energy limited LST, albedo and ET for EF to sparse conversion:",
        file=text_file,
    )
    print(ef_sparse_el_wl_df, file=text_file)
text_file.close()

# extract_wl_el(orig_class=EF, val=3, outname="EF_shrub_EL_WL.png")  # Not enough data
# extract_wl_el(orig_class=DF, val=5, outname="DF_sparse_EL_WL.png")  # Not enough data
# extract_wl_el(orig_class=DF, val=4, outname="DF_herb_EL_WL.png")  # Not enough data
# extract_wl_el(
# 	orig_class=shrub, val=5, outname="Shrub_sparse_EL_WL.png"
# )  # Not enough data

print("Extracting Herbaceous to Sparse energy vs. water limited LST, albedo and ET\n")
# Herb to sparse conversion which we have enought data to analyze
lst_EL_WL, albedo_EL_WL, et_EL_WL = extract_wl_el(
	orig_class=herb, val=5
)  # There are enough data

df1 = {"name": "LST", "df": lst_EL_WL, "label": "$\Delta$LST [C]", "ylim": [-12, 12]}
df2 = {
	"name": "Albedo",
	"df": albedo_EL_WL,
	"label": "$\Delta$Albedo",
	"ylim": [-0.5, 0.5],
}
df3 = {
	"name": "ET",
	"df": et_EL_WL,
	"label": "$\Delta$ET [mm/year]",
	"ylim": [-850, 850],
}
print(f"saving Herb_sparse_EL_WL.png in {fig_dir}")
myboxplot_group(
	df1, df2, df3, columns=class_name, txt_pos=9, outname="Herb_sparse_EL_WL.png"
)

lst_mean=lst_EL_WL.mean()
albedo_mean=albedo_EL_WL.mean()
et_mean=et_EL_WL.mean()
lst_std=lst_EL_WL.std()
albedo_std=albedo_EL_WL.std()
et_std=et_EL_WL.std()

herb_sparse_el_wl_df = pd.concat([lst_mean,albedo_mean,et_mean,lst_std,albedo_std,et_std],axis=1)
herb_sparse_el_wl_df.columns = ['LST Mean','Albedo Mean','ET Mean','LST STD','Albedo STD','ET STD']

with open(out_dir +analyses_mode+"_report.txt", "a") as text_file:
    print("\n Mean and STD of water vs. energy limited LST, albedo and ET for herb to sparse conversion:",
        file=text_file,
    )
    print(herb_sparse_el_wl_df, file=text_file)
text_file.close()

# extract_wl_el(
# 	orig_class=sparse, val=3, outname="Sparse_shrub_EL_WL.png"
# )  # Not enough data
# extract_wl_el(
# 	orig_class=sparse, val=4, outname="Sparse_herb_EL_WL.png"
# )  # Not enough data
# extract_wl_el(
# 	orig_class=wetland, val=5, outname="Wetland_sparse_EL_WL.png"
# )  # Not enough data


"""---------------------------------------------------------------------------

					Now we focus on different components of the ET 

------------------------------------------------------------------------------"""
print('----------------- Plotting ET component -----------------\n')
EC_EF_to_shrub = extract_vals(EF, 3, "ec", "EC_EF_to_shrub")
# ET_EF_to_herb = extract_vals(EF, 4, "et", "ET_EF_to_herb")
EC_EF_to_sparse = extract_vals(EF, 5, "ec", "EC_EF_to_sparse")

# ET_DF_to_shrub = extract_vals(DF, 3, "et", "ET_DF_to_shrub")
EC_DF_to_herb = extract_vals(DF, 4, "ec", "EC_DF_to_herb")
EC_DF_to_sparse = extract_vals(DF, 5, "ec", "EC_DF_to_sparse")

EC_shrub_to_sparse = extract_vals(shrub, 5, "ec", "EC_shrub_to_sparse")
# ET_shrub_to_wetland = extract_vals(shrub, 6, "et", "ET_shrub_to_wetland")

EC_herb_to_shrub = extract_vals(herb, 3, "ec", "EC_herb_to_shrub")
EC_herb_to_sparse = extract_vals(herb, 5, "ec", "EC_herb_to_sparse")
# ET_herb_to_wetland = extract_vals(herb, 6, "et", "ET_herb_to_wetland")

EC_sparse_to_shrub = extract_vals(sparse, 3, "ec", "EC_sparse_to_shrub")
EC_sparse_to_herb = extract_vals(sparse, 4, "ec", "EC_sparse_to_herb")

EC_wetland_to_sparse = extract_vals(wetland, 5, "ec", "EC_wetland_to_sparse")

# ET_water_to_sparse = extract_vals(water, 5, "et", "ET_water_to_sparse")
# ET_water_to_wetland = extract_vals(water, 6, "et", "ET_water_to_wetland")

df_ec = pd.concat(
	[
		EC_EF_to_shrub,
		# ET_EF_to_herb,
		EC_EF_to_sparse,
		# ET_DF_to_shrub,
		EC_DF_to_herb,
		EC_DF_to_sparse,
		EC_shrub_to_sparse,
		# ET_shrub_to_wetland,
		EC_herb_to_shrub,
		EC_herb_to_sparse,
		# ET_herb_to_wetland,
		EC_sparse_to_shrub,
		EC_sparse_to_herb,
		EC_wetland_to_sparse,
		# ET_water_to_sparse,
		# ET_water_to_wetland,
	],
	ignore_index=True,
	axis=1,
)

df_ec.columns = columns
print(f"saving EC_Boxplot.png in {fig_dir}")
myboxplot(
	df=df_ec,
	title="EC",
	ylabel="EC [mm $year^{-1}$]",
	margin=5,
	outname="EC_Boxplot.png",
)


ES_EF_to_shrub = extract_vals(EF, 3, "es", "ES_EF_to_shrub")
# ET_EF_to_herb = extract_vals(EF, 4, "et", "ET_EF_to_herb")
ES_EF_to_sparse = extract_vals(EF, 5, "es", "ES_EF_to_sparse")

# ET_DF_to_shrub = extract_vals(DF, 3, "et", "ET_DF_to_shrub")
ES_DF_to_herb = extract_vals(DF, 4, "es", "ES_DF_to_herb")
ES_DF_to_sparse = extract_vals(DF, 5, "es", "ES_DF_to_sparse")

ES_shrub_to_sparse = extract_vals(shrub, 5, "es", "ES_shrub_to_sparse")
# ET_shrub_to_wetland = extract_vals(shrub, 6, "et", "ET_shrub_to_wetland")

ES_herb_to_shrub = extract_vals(herb, 3, "es", "ES_herb_to_shrub")
ES_herb_to_sparse = extract_vals(herb, 5, "es", "ES_herb_to_sparse")
# ET_herb_to_wetland = extract_vals(herb, 6, "et", "ET_herb_to_wetland")

ES_sparse_to_shrub = extract_vals(sparse, 3, "es", "ES_sparse_to_shrub")
ES_sparse_to_herb = extract_vals(sparse, 4, "es", "ES_sparse_to_herb")

ES_wetland_to_sparse = extract_vals(wetland, 5, "es", "ES_wetland_to_sparse")

# ET_water_to_sparse = extract_vals(water, 5, "et", "ET_water_to_sparse")
# ET_water_to_wetland = extract_vals(water, 6, "et", "ET_water_to_wetland")

df_es = pd.concat(
	[
		ES_EF_to_shrub,
		# ET_EF_to_herb,
		ES_EF_to_sparse,
		# ET_DF_to_shrub,
		ES_DF_to_herb,
		ES_DF_to_sparse,
		ES_shrub_to_sparse,
		# ET_shrub_to_wetland,
		ES_herb_to_shrub,
		ES_herb_to_sparse,
		# ET_herb_to_wetland,
		ES_sparse_to_shrub,
		ES_sparse_to_herb,
		ES_wetland_to_sparse,
		# ET_water_to_sparse,
		# ET_water_to_wetland,
	],
	ignore_index=True,
	axis=1,
)

df_es.columns = columns
print(f"saving ES_Boxplot.png in {fig_dir}")
myboxplot(
	df=df_es,
	title="ES",
	ylabel="ES [mm $year^{-1}$]",
	margin=5,
	outname="ES_Boxplot.png",
)


EW_EF_to_shrub = extract_vals(EF, 3, "ew", "EW_EF_to_shrub")
# ET_EF_to_herb = extract_vals(EF, 4, "et", "ET_EF_to_herb")
EW_EF_to_sparse = extract_vals(EF, 5, "ew", "EW_EF_to_sparse")

# ET_DF_to_shrub = extract_vals(DF, 3, "et", "ET_DF_to_shrub")
EW_DF_to_herb = extract_vals(DF, 4, "ew", "EW_DF_to_herb")
EW_DF_to_sparse = extract_vals(DF, 5, "ew", "EW_DF_to_sparse")

EW_shrub_to_sparse = extract_vals(shrub, 5, "ew", "EW_shrub_to_sparse")
# ET_shrub_to_wetland = extract_vals(shrub, 6, "et", "ET_shrub_to_wetland")

EW_herb_to_shrub = extract_vals(herb, 3, "ew", "EW_herb_to_shrub")
EW_herb_to_sparse = extract_vals(herb, 5, "ew", "EW_herb_to_sparse")
# ET_herb_to_wetland = extract_vals(herb, 6, "et", "ET_herb_to_wetland")

EW_sparse_to_shrub = extract_vals(sparse, 3, "ew", "EW_sparse_to_shrub")
EW_sparse_to_herb = extract_vals(sparse, 4, "ew", "EW_sparse_to_herb")

EW_wetland_to_sparse = extract_vals(wetland, 5, "ew", "EW_wetland_to_sparse")

# ET_water_to_sparse = extract_vals(water, 5, "et", "ET_water_to_sparse")
# ET_water_to_wetland = extract_vals(water, 6, "et", "ET_water_to_wetland")

df_ew = pd.concat(
	[
		EW_EF_to_shrub,
		# ET_EF_to_herb,
		EW_EF_to_sparse,
		# ET_DF_to_shrub,
		EW_DF_to_herb,
		EW_DF_to_sparse,
		EW_shrub_to_sparse,
		# ET_shrub_to_wetland,
		EW_herb_to_shrub,
		EW_herb_to_sparse,
		# ET_herb_to_wetland,
		EW_sparse_to_shrub,
		EW_sparse_to_herb,
		EW_wetland_to_sparse,
		# ET_water_to_sparse,
		# ET_water_to_wetland,
	],
	ignore_index=True,
	axis=1,
)

df_ew.columns = columns
print(f"saving EW_Boxplot.png in {fig_dir}")
myboxplot(
	df=df_ew,
	title="EW",
	ylabel="EW [mm $year^{-1}$]",
	margin=5,
	outname="EW_Boxplot.png",
)

ECI_EF_to_shrub = extract_vals(EF, 3, "eci", "ECI_EF_to_shrub")
# ET_EF_to_herb = extract_vals(EF, 4, "et", "ET_EF_to_herb")
ECI_EF_to_sparse = extract_vals(EF, 5, "eci", "ECI_EF_to_sparse")

# ET_DF_to_shrub = extract_vals(DF, 3, "et", "ET_DF_to_shrub")
ECI_DF_to_herb = extract_vals(DF, 4, "eci", "ECI_DF_to_herb")
ECI_DF_to_sparse = extract_vals(DF, 5, "eci", "ECI_DF_to_sparse")

ECI_shrub_to_sparse = extract_vals(shrub, 5, "eci", "ECI_shrub_to_sparse")
# ET_shrub_to_wetland = extract_vals(shrub, 6, "et", "ET_shrub_to_wetland")

ECI_herb_to_shrub = extract_vals(herb, 3, "eci", "ECI_herb_to_shrub")
ECI_herb_to_sparse = extract_vals(herb, 5, "eci", "ECI_herb_to_sparse")
# ET_herb_to_wetland = extract_vals(herb, 6, "et", "ET_herb_to_wetland")

ECI_sparse_to_shrub = extract_vals(sparse, 3, "eci", "ECI_sparse_to_shrub")
ECI_sparse_to_herb = extract_vals(sparse, 4, "eci", "ECI_sparse_to_herb")

ECI_wetland_to_sparse = extract_vals(wetland, 5, "eci", "ECI_wetland_to_sparse")

# ET_water_to_sparse = extract_vals(water, 5, "et", "ET_water_to_sparse")
# ET_water_to_wetland = extract_vals(water, 6, "et", "ET_water_to_wetland")

df_eci = pd.concat(
	[
		ECI_EF_to_shrub,
		# ET_EF_to_herb,
		ECI_EF_to_sparse,
		# ET_DF_to_shrub,
		ECI_DF_to_herb,
		ECI_DF_to_sparse,
		ECI_shrub_to_sparse,
		# ET_shrub_to_wetland,
		ECI_herb_to_shrub,
		ECI_herb_to_sparse,
		# ET_herb_to_wetland,
		ECI_sparse_to_shrub,
		ECI_sparse_to_herb,
		ECI_wetland_to_sparse,
		# ET_water_to_sparse,
		# ET_water_to_wetland,
	],
	ignore_index=True,
	axis=1,
)

df_eci.columns = columns
print(f"saving ECI_Boxplot.png in {fig_dir}")
myboxplot(
	df=df_eci,
	title="ECI",
	ylabel="ECI [mm $year^{-1}$]",
	margin=5,
	outname="ECI_Boxplot.png",
)

ESW_EF_to_shrub = extract_vals(EF, 3, "esw", "ESW_EF_to_shrub")
# ET_EF_to_herb = extract_vals(EF, 4, "et", "ET_EF_to_herb")
ESW_EF_to_sparse = extract_vals(EF, 5, "esw", "ESW_EF_to_sparse")

# ET_DF_to_shrub = extract_vals(DF, 3, "et", "ET_DF_to_shrub")
ESW_DF_to_herb = extract_vals(DF, 4, "esw", "ESW_DF_to_herb")
ESW_DF_to_sparse = extract_vals(DF, 5, "esw", "ESW_DF_to_sparse")

ESW_shrub_to_sparse = extract_vals(shrub, 5, "esw", "ESW_shrub_to_sparse")
# ET_shrub_to_wetland = extract_vals(shrub, 6, "et", "ET_shrub_to_wetland")

ESW_herb_to_shrub = extract_vals(herb, 3, "esw", "ESW_herb_to_shrub")
ESW_herb_to_sparse = extract_vals(herb, 5, "esw", "ESW_herb_to_sparse")
# ET_herb_to_wetland = extract_vals(herb, 6, "et", "ET_herb_to_wetland")

ESW_sparse_to_shrub = extract_vals(sparse, 3, "esw", "ESW_sparse_to_shrub")
ESW_sparse_to_herb = extract_vals(sparse, 4, "esw", "ESW_sparse_to_herb")

ESW_wetland_to_sparse = extract_vals(wetland, 5, "esw", "ESW_wetland_to_sparse")

# ET_water_to_sparse = extract_vals(water, 5, "et", "ET_water_to_sparse")
# ET_water_to_wetland = extract_vals(water, 6, "et", "ET_water_to_wetland")

df_esw = pd.concat(
	[
		ESW_EF_to_shrub,
		# ET_EF_to_herb,
		ESW_EF_to_sparse,
		# ET_DF_to_shrub,
		ESW_DF_to_herb,
		ESW_DF_to_sparse,
		ESW_shrub_to_sparse,
		# ET_shrub_to_wetland,
		ESW_herb_to_shrub,
		ESW_herb_to_sparse,
		# ET_herb_to_wetland,
		ESW_sparse_to_shrub,
		ESW_sparse_to_herb,
		ESW_wetland_to_sparse,
		# ET_water_to_sparse,
		# ET_water_to_wetland,
	],
	ignore_index=True,
	axis=1,
)

df_esw.columns = columns
print(f"saving ESW_Boxplot.png in {fig_dir}")
myboxplot(
	df=df_esw,
	title="ESW",
	ylabel="ESW [mm $year^{-1}$]",
	margin=5,
	outname="ESW_Boxplot.png",
)

df1 = {
	"name": "CI",
	"df": df_eci,
	"label": "$\Delta$CI [mm/year]]",
	"ylim": [-300, 300],
}
df2 = {
	"name": "SW",
	"df": df_esw,
	"label": "$\Delta$SW [mm/year]]",
	"ylim": [-300, 300],
}

df3 = {"name": "ET", "df": df_et, "label": "$\Delta$ET [mm/year]]", "ylim": [-850, 850],}
myboxplot_group(
	df1, df2, df3, columns=columns, txt_pos=200, outname="ET_Components_integrated.png"
)

df_eci_mean = df_eci.mean()
df_esw_mean = df_esw.mean()
df_et_mean = df_et.mean()
df_eci_std = df_eci.std()
df_esw_std = df_esw.std()
df_et_std = df_et.std()

frames = pd.concat([df_eci_mean,df_esw_mean,df_et_mean,df_eci_std,df_esw_std,df_et_std],axis=1)
frames.columns = ["ECI Mean","ESW Mean","ET Mean","ECI STD","ESW STD","ET STD"]

with open(out_dir +analyses_mode+"_report.txt", "a") as text_file:
    print("\n Mean and STD of ET components for different LUC conversions:",
        file=text_file,
    )
    print(frames, file=text_file)
text_file.close()
print("Mean and STD of ET components for different LUC conversions:")
print(frames)

""" ------------------------------------------------------------------
Analyzing an two example pixles where one the LUC (e.g. DF) is changed
and the other which is approximaltly close hasn't been changed. These pixles 
are arbitrary and just for show.  
-------------------------------------------------------------------"""
print('----------------- Working on the example (note this is just for the presentation and everything is arbitarary) -----------------\n')
# The year DF_2007 is just two aribitraty pixels to show the changes between changed and unchaneged pixels (for presentation purposes)
DF_2007 = DF.sel(year=2007)
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

print('----------------- All done! -----------------\n')

# ------------------------------ End of the script --------------------------------------------