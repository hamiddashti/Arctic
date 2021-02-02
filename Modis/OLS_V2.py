# PRocessing the results of the extract_conversion.py
import xarray as xr
import matplotlib.pylab as plt
import numpy as np
from scipy import stats
import statsmodels.api as sm
import pandas as pd
from sklearn.preprocessing import PowerTransformer
import dask

dask.config.set(scheduller="processes")
from dask.diagnostics import ProgressBar
from sklearn import preprocessing

# in_dir = "F:\\working\\LUC\\test\\"
# out_dir = "F:\\working\\LUC\\test\\outputs\\"
in_dir = "/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/outputs/Sensitivity/EndPoints/"
out_dir = "/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/outputs/Sensitivity/EndPoints/"
# data_dir = '/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/Data/'
# min_thresh = 1 # Minimum percent cover change for a conversion to be included in the OLS analyses
# var_type = "LST"
print("Estimating slopes using OLS method")


def lsg(predictors, target):
	# least square solution
	a = np.linalg.inv(np.matmul(predictors.T, predictors))
	b = np.matmul(predictors.T, target)
	coefficients = np.matmul(a, b)
	return coefficients


def reject_outliers(data, m):
	# m is number of std
	import numpy as np

	data = data.astype(float)
	data[abs(data - np.nanmean(data)) > m * np.nanstd(data)] = np.nan
	return data


def my_count(d):
	return len(d[np.where(d > 1)])


def my_cal(n, x, y):
	# for n in range(10):
	p = np.random.choice(np.arange(0, len(y)), 100000)
	yy = y[p]
	xx = x[p, :]
	tmp = lsg(predictors=xx, target=yy)
	cof.append(tmp)


def create_empty_dataframe(size):
	empty_matrix = np.empty((size, size))
	empty_matrix[:] = np.nan
	empty_df = pd.DataFrame(data=empty_matrix, index=classes, columns=classes)
	return empty_df


def save_results(tri):
	if tri == "upper":
		summary_upper = []
		for i in range(0, len(par_upper[:-1])):
			tmp = (
				"("
				+ par_upper.astype(str)[i]
				+ u"\u00B1"
				+ stand_err_upper.astype(str)[i]
				+ ";"
				+ pvalue_upper.astype(str)[i]
				+ ")"
			)
			summary_upper.append(tmp)
		upper_df = create_empty_dataframe(7)
		I = np.array(np.triu_indices(n=7, k=1))
		for i in range(0, I.shape[1]):
			upper_df.iloc[I[:, i][0], I[:, i][1]] = summary_upper[i]
		output_file = open(out_dir + "upper_results.txt", "a")
		output_file.write(upper_df.to_string())
		output_file.close()
	elif tri == "lower":
		summary_lower = []
		for i in range(0, len(par_lower[:-1])):
			tmp = (
				"("
				+ par_lower.astype(str)[i]
				+ u"\u00B1"
				+ stand_err_lower.astype(str)[i]
				+ ";"
				+ pvalue_lower.astype(str)[i]
				+ ")"
			)
			summary_lower.append(tmp)
		lower_df = create_empty_dataframe(7)
		I = np.array(np.tril_indices(n=7, k=-1))
		for i in range(0, I.shape[1]):
			lower_df.iloc[I[:, i][0], I[:, i][1]] = summary_lower[i]
		output_file = open(out_dir + "lower_results.txt", "a")
		output_file.write(lower_df.to_string())
		output_file.close()
	elif tri == "all":
		summary_all = []
		for i in range(0, len(par_all[:-1])):
			tmp = (
				"("
				+ par_all.astype(str)[i]
				+ u"\u00B1"
				+ stand_err_all.astype(str)[i]
				+ ";"
				+ pvalue_all.astype(str)[i]
				+ ")"
			)
			summary_all.append(tmp)
		all_df = create_empty_dataframe(7)
		counter = 0
		for i in range(0, 7):
			for j in range(0, 7):
				if i == j:
					# counter=counter+1
					continue
				all_df.iloc[i, j] = summary_all[counter]
				counter = counter + 1
		output_file = open(out_dir + "all_results.txt", "a")
		output_file.write(all_df.to_string())
		output_file.close()


def myboxplot(df, title, ylabel, margin, outname):
	plt.close()
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
			horizontalalignment="center",
			fontsize=12,
			color="k",
			weight="bold",
		)

	plt.xticks(rotation=45, fontsize=16)
	plt.title(title, fontsize=20)
	plt.ylabel(ylabel, fontsize=16)
	plt.tight_layout()
	plt.savefig(out_dir + outname)
	plt.close()


def make_mask(data):
	# Filter data using np.isnan
	mask = ~np.isnan(data)
	filtered_data = [d[m] for d, m in zip(data.T, mask.T)]
	return filtered_data


def myboxplot_group(df1, df2, df3, columns,txt_pos, outname):
	plt.close()
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

	# a = np.arange(0, len(columns)) % 2
	# txt_pos = np.where(a == 0, 7, 9)
	counter = 0
	for tick, label in zip(pos, ax1.get_xticklabels()):
		ax1.text(
			pos[tick],
			# q[tick] + (q[tick]-lst_mean[tick])/2,
			txt_pos[counter],
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
			fontsize=7.2,
			color="k",
			weight="semibold",
		)
		counter = counter + 1
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
	plt.savefig(out_dir + outname)
	plt.close()


# This is the list of 49 conversions
lower_list = np.array(
	[7, 14, 15, 21, 22, 23, 28, 29, 30, 31, 35, 36, 37, 38, 39, 40, 43, 44, 45, 46, 47]
)
diagnoal_list = np.array([0, 8, 16, 24, 32, 40, 48])
upper_list = np.array(
	[1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 17, 18, 19, 20, 25, 26, 27, 33, 34, 41]
)
water_list = np.array([6, 13, 20, 27, 34, 41, 42, 43, 44, 45, 46, 47])

# ds = xr.open_dataset('ds_2013_Endpoints.nc')
classes = ["EF", "DF", "shrub", "herb", "sparse", "wetland", "water"]
# ds = xr.open_dataset(in_dir+"ds_2013_EndPoints.nc")
ds = xr.open_dataset(in_dir + "Confusion_Table_v1.nc")
# ds_shape = [ds.dims['time'],ds.dims['ID'],ds.dims['Conversions']]
Conversions = ds["Conversion"].values
Conversion_all = np.delete(Conversions, diagnoal_list)
Conversion_upper = Conversions[upper_list]

# Anaylzing the EndPoints (changes at 2003 compared with 2013)
cnt = ds["NORMALIZED_CONFUSION"].values.squeeze() * 100

lst_2003 = ds["LST_2003"].values.squeeze()
lst_2013 = ds["LST_2013"].values.squeeze()
lst = ds["DELTA_LST_LULC"].values.squeeze()

albedo_2003 = ds["ALBEDO_2003"].values.squeeze()
albedo_2013 = ds["ALBEDO_2013"].values.squeeze()
albedo = ds["DELTA_ALBEDO"].values.squeeze()

et_2003 = ds["ET_2003"].values.squeeze()
et_2013 = ds["ET_2013"].values.squeeze()
et = ds["DELTA_ET"].values.squeeze()

ec_2003 = ds["EC_2003"].values.squeeze()
ec_2013 = ds["EC_2013"].values.squeeze()
ec = ds["DELTA_EC"].values.squeeze()

eci_2003 = ds["ECI_2003"].values.squeeze()
eci_2013 = ds["ECI_2013"].values.squeeze()
eci = ds["DELTA_ECI"].values.squeeze()

esw_2003 = ds["ESW_2003"].values.squeeze()
esw_2013 = ds["ESW_2013"].values.squeeze()
esw = ds["DELTA_ESW"].values.squeeze()

# Devide data based on upper and lower part of the confusion table
cnt_upper = cnt[:, upper_list]
cnt_lower = cnt[:, lower_list]
cnt_all = np.delete(cnt, diagnoal_list, axis=1)

# Removing outliers in the LST
lst_outliers = ~np.isnan(reject_outliers(lst, 2))
cnt_upper_clean_tmp = cnt_upper[lst_outliers].squeeze()
cnt_lower_clean_tmp = cnt_lower[lst_outliers].squeeze()
cnt_all_clean_tmp = cnt_all[lst_outliers].squeeze()

lst_2003_clean = lst_2003[lst_outliers].squeeze()
lst_2013_clean = lst_2013[lst_outliers].squeeze()
lst_clean = lst[lst_outliers].squeeze()

albedo_2003_clean = albedo_2003[lst_outliers].squeeze()
albedo_2013_clean = albedo_2013[lst_outliers].squeeze()
albedo_clean = albedo[lst_outliers].squeeze()

ec_2003_clean = ec_2003[lst_outliers].squeeze()
ec_2013_clean = ec_2013[lst_outliers].squeeze()
ec_clean = ec[lst_outliers].squeeze()

et_2003_clean = et_2003[lst_outliers].squeeze()
et_2013_clean = et_2013[lst_outliers].squeeze()
et_clean = et[lst_outliers].squeeze()

eci_2003_clean = eci_2003[lst_outliers].squeeze()
eci_2013_clean = eci_2013[lst_outliers].squeeze()
eci_clean = eci[lst_outliers].squeeze()

esw_2003_clean = esw_2003[lst_outliers].squeeze()
esw_2013_clean = esw_2013[lst_outliers].squeeze()
esw_clean = esw[lst_outliers].squeeze()

# Get rid rows wehere all elements are zero
all_upper_zeros = np.all((cnt_upper_clean_tmp == 0), axis=1)
all_lower_zeros = np.all((cnt_lower_clean_tmp == 0), axis=1)
all_all_zeros = np.all((cnt_all_clean_tmp == 0), axis=1)

cnt_upper_clean = cnt_upper_clean_tmp[~all_upper_zeros]
# cnt_upper_clean = preprocessing.scale(cnt_upper_clean)
lst_2003_upper_clean = lst_2003_clean[~all_upper_zeros]
lst_2013_upper_clean = lst_2013_clean[~all_upper_zeros]
lst_upper_clean = lst_clean[~all_upper_zeros]

albedo_2003_upper_clean = albedo_2003_clean[~all_upper_zeros]
albedo_2013_upper_clean = albedo_2013_clean[~all_upper_zeros]
albedo_upper_clean = albedo_clean[~all_upper_zeros]

et_2003_upper_clean = et_2003_clean[~all_upper_zeros]
et_2013_upper_clean = et_2013_clean[~all_upper_zeros]
et_upper_clean = et_clean[~all_upper_zeros]

ec_2003_upper_clean = ec_2003_clean[~all_upper_zeros]
ec_2013_upper_clean = ec_2013_clean[~all_upper_zeros]
ec_upper_clean = ec_clean[~all_upper_zeros]

eci_2003_upper_clean = eci_2003_clean[~all_upper_zeros]
eci_2013_upper_clean = eci_2013_clean[~all_upper_zeros]
eci_upper_clean = eci_clean[~all_upper_zeros]

esw_2003_upper_clean = esw_2003_clean[~all_upper_zeros]
esw_2013_upper_clean = esw_2013_clean[~all_upper_zeros]
esw_upper_clean = esw_clean[~all_upper_zeros]

# cnt_scaled = preprocessing.scale(cnt_clean)
cnt_lower_clean = cnt_lower_clean_tmp[~all_lower_zeros]
# cnt_lower_clean = preprocessing.scale(cnt_lower_clean)
lst_2003_lower_clean = lst_2003_clean[~all_lower_zeros]
lst_2013_lower_clean = lst_2013_clean[~all_lower_zeros]
lst_lower_clean = lst_clean[~all_lower_zeros]

albedo_2003_lower_clean = albedo_2003_clean[~all_lower_zeros]
albedo_2013_lower_clean = albedo_2013_clean[~all_lower_zeros]
albedo_lower_clean = albedo_clean[~all_lower_zeros]

et_2003_lower_clean = et_2003_clean[~all_lower_zeros]
et_2013_lower_clean = et_2013_clean[~all_lower_zeros]
et_lower_clean = et_clean[~all_lower_zeros]

ec_2003_lower_clean = ec_2003_clean[~all_lower_zeros]
ec_2013_lower_clean = ec_2013_clean[~all_lower_zeros]
ec_lower_clean = ec_clean[~all_lower_zeros]

eci_2003_lower_clean = eci_2003_clean[~all_lower_zeros]
eci_2013_lower_clean = eci_2013_clean[~all_lower_zeros]
eci_lower_clean = eci_clean[~all_lower_zeros]

esw_2003_lower_clean = esw_2003_clean[~all_lower_zeros]
esw_2013_lower_clean = esw_2013_clean[~all_lower_zeros]
esw_lower_clean = esw_clean[~all_lower_zeros]

cnt_all_clean = cnt_all_clean_tmp[~all_all_zeros]
# cnt_all_clean = preprocessing.scale(cnt_all_clean)
lst_2003_all_clean = lst_2003_clean[[~all_all_zeros]]
lst_2013_all_clean = lst_2013_clean[[~all_all_zeros]]
lst_all_clean = lst_clean[~all_all_zeros]

albedo_2003_all_clean = albedo_2003_clean[[~all_all_zeros]]
albedo_2013_all_clean = albedo_2013_clean[[~all_all_zeros]]
albedo_all_clean = albedo_clean[~all_all_zeros]

et_2003_all_clean = et_2003_clean[[~all_all_zeros]]
et_2013_all_clean = et_2013_clean[[~all_all_zeros]]
et_all_clean = et_clean[~all_all_zeros]

ec_2003_all_clean = ec_2003_clean[[~all_all_zeros]]
ec_2013_all_clean = ec_2013_clean[[~all_all_zeros]]
ec_all_clean = ec_clean[~all_all_zeros]

eci_2003_all_clean = eci_2003_clean[[~all_all_zeros]]
eci_2013_all_clean = eci_2013_clean[[~all_all_zeros]]
eci_all_clean = eci_clean[~all_all_zeros]

esw_2003_all_clean = esw_2003_clean[[~all_all_zeros]]
esw_2013_all_clean = esw_2013_clean[[~all_all_zeros]]
esw_all_clean = esw_clean[~all_all_zeros]

# Adding a column vector of ones (intercept)
one_upper_col = np.ones(len(lst_upper_clean))
one_lower_col = np.ones(len(lst_lower_clean))
one_all_col = np.ones(len(lst_all_clean))

cnt_upper_final = np.c_[cnt_upper_clean, one_upper_col]
cnt_lower_final = np.c_[cnt_lower_clean, one_lower_col]
cnt_all_final = np.c_[cnt_all_clean, one_all_col]
upper_non_zero = np.apply_along_axis(my_count, 0, cnt_upper_final)
lower_non_zero = np.apply_along_axis(my_count, 0, cnt_lower_final)
all_non_zero = np.apply_along_axis(my_count, 0, cnt_all_final)

results_upper = sm.OLS(lst_upper_clean, cnt_upper_final).fit()
results_lower = sm.OLS(lst_lower_clean, cnt_lower_final).fit()
results_all = sm.OLS(lst_all_clean, cnt_all_final).fit()

par_upper = np.round(results_upper.params, 5)
pvalue_upper = np.round(results_upper.pvalues, 5)
stand_err_upper = np.round(results_upper.bse, 5)
save_results("upper")
par_lower = np.round(results_lower.params, 5)
pvalue_lower = np.round(results_lower.pvalues, 5)
stand_err_lower = np.round(results_lower.bse, 5)
save_results("lower")
par_all = np.round(results_all.params, 5)
pvalue_all = np.round(results_all.pvalues, 5)
stand_err_all = np.round(results_all.bse, 5)
save_results("all")
print("OLS results saved in:")
print(out_dir)
# ----------------------------------------------------------------
# 				Boxplot of extreme changes
# -----------------------------------------------------------------
lst_2003_tmp = []
lst_2013_tmp = []
lst_tmp = []
albedo_2003_tmp = []
albedo_2013_tmp = []
albedo_tmp = []
et_2003_tmp = []
et_2013_tmp = []
et_tmp = []
ec_2003_tmp = []
ec_2013_tmp = []
ec_tmp = []
eci_2003_tmp = []
eci_2013_tmp = []
eci_tmp = []
esw_2003_tmp = []
esw_2013_tmp = []
esw_tmp = []

conversions_tmp = []
for i in range(0, cnt_upper_final.shape[1] - 1):
	I = np.where(cnt_upper_final[:, i] > 50)
	# Remove coversions where we don't have any extreme observatios for them
	if np.array(I).shape[1] < 15:
		continue
	lst_2003_tmp.append(lst_2003_upper_clean[I].squeeze())
	lst_2013_tmp.append(lst_2013_upper_clean[I].squeeze())
	lst_tmp.append(lst_upper_clean[I].squeeze())
	albedo_2003_tmp.append(albedo_2003_upper_clean[I].squeeze())
	albedo_2013_tmp.append(albedo_2013_upper_clean[I].squeeze())
	albedo_tmp.append(albedo_upper_clean[I].squeeze())
	et_2003_tmp.append(et_2003_upper_clean[I].squeeze())
	et_2013_tmp.append(et_2013_upper_clean[I].squeeze())
	et_tmp.append(et_upper_clean[I].squeeze())
	ec_2003_tmp.append(ec_2003_upper_clean[I].squeeze())
	ec_2013_tmp.append(ec_2013_upper_clean[I].squeeze())
	ec_tmp.append(ec_upper_clean[I].squeeze())
	eci_2003_tmp.append(eci_2003_upper_clean[I].squeeze())
	eci_2013_tmp.append(eci_2013_upper_clean[I].squeeze())
	eci_tmp.append(eci_upper_clean[I].squeeze())
	esw_2003_tmp.append(esw_2003_upper_clean[I].squeeze())
	esw_2013_tmp.append(esw_2013_upper_clean[I].squeeze())
	esw_tmp.append(esw_upper_clean[I].squeeze())
	conversions_tmp.append(Conversion_upper[i])

lst_2003_array = np.array(lst_2003_tmp)
lst_2013_array = np.array(lst_2013_tmp)
lst_array = np.array(lst_tmp)
lst_2003_array_df = pd.DataFrame(lst_2003_array[0])
lst_2013_array_df = pd.DataFrame(lst_2013_array[0])
lst_array_df = pd.DataFrame(lst_array[0])

albedo_2003_array = np.array(albedo_2003_tmp)
albedo_2013_array = np.array(albedo_2013_tmp)
albedo_array = np.array(albedo_tmp)
albedo_2003_array_df = pd.DataFrame(albedo_2003_array[0])
albedo_2013_array_df = pd.DataFrame(albedo_2013_array[0])
albedo_array_df = pd.DataFrame(albedo_array[0])

et_2003_array = np.array(et_2003_tmp)
et_2013_array = np.array(et_2013_tmp)
et_array = np.array(et_tmp)
et_2003_array_df = pd.DataFrame(et_2003_array[0])
et_2013_array_df = pd.DataFrame(et_2013_array[0])
et_array_df = pd.DataFrame(et_array[0])

ec_2003_array = np.array(ec_2003_tmp)
ec_2013_array = np.array(ec_2013_tmp)
ec_array = np.array(ec_tmp)
ec_2003_array_df = pd.DataFrame(ec_2003_array[0])
ec_2013_array_df = pd.DataFrame(ec_2013_array[0])
ec_array_df = pd.DataFrame(ec_array[0])

eci_2003_array = np.array(eci_2003_tmp)
eci_2013_array = np.array(eci_2013_tmp)
eci_array = np.array(eci_tmp)
eci_2003_array_df = pd.DataFrame(eci_2003_array[0])
eci_2013_array_df = pd.DataFrame(eci_2013_array[0])
eci_array_df = pd.DataFrame(eci_array[0])

esw_2003_array = np.array(esw_2003_tmp)
esw_2013_array = np.array(esw_2013_tmp)
esw_array = np.array(esw_tmp)
esw_2003_array_df = pd.DataFrame(esw_2003_array[0])
esw_2013_array_df = pd.DataFrame(esw_2013_array[0])
esw_array_df = pd.DataFrame(esw_array[0])

for k in range(1, len(lst_array)):
	lst_2003_array_df = pd.concat([lst_2003_array_df, pd.DataFrame(lst_2003_array[k])], ignore_index=True, axis=1)
	lst_2013_array_df = pd.concat([lst_2013_array_df, pd.DataFrame(lst_2013_array[k])], ignore_index=True, axis=1)
	lst_array_df = pd.concat([lst_array_df, pd.DataFrame(lst_array[k])], ignore_index=True, axis=1)
	albedo_2003_array_df = pd.concat([albedo_2003_array_df, pd.DataFrame(albedo_2003_array[k])], ignore_index=True, axis=1)
	albedo_2013_array_df = pd.concat([albedo_2013_array_df, pd.DataFrame(albedo_2013_array[k])], ignore_index=True, axis=1)
	albedo_array_df = pd.concat([albedo_array_df, pd.DataFrame(albedo_array[k])], ignore_index=True, axis=1)
	et_2003_array_df = pd.concat([et_2003_array_df, pd.DataFrame(et_2003_array[k])], ignore_index=True, axis=1)
	et_2013_array_df = pd.concat([et_2013_array_df, pd.DataFrame(et_2013_array[k])], ignore_index=True, axis=1)
	et_array_df = pd.concat([et_array_df, pd.DataFrame(et_array[k])], ignore_index=True, axis=1)
	ec_2003_array_df = pd.concat([ec_2003_array_df, pd.DataFrame(ec_2003_array[k])], ignore_index=True, axis=1)
	ec_2013_array_df = pd.concat([ec_2013_array_df, pd.DataFrame(ec_2013_array[k])], ignore_index=True, axis=1)
	ec_array_df = pd.concat([ec_array_df, pd.DataFrame(ec_array[k])], ignore_index=True, axis=1)
	eci_2003_array_df = pd.concat([eci_2003_array_df, pd.DataFrame(eci_2003_array[k])], ignore_index=True, axis=1)
	eci_2013_array_df = pd.concat([eci_2013_array_df, pd.DataFrame(eci_2013_array[k])], ignore_index=True, axis=1)
	eci_array_df = pd.concat([eci_array_df, pd.DataFrame(eci_array[k])], ignore_index=True, axis=1)
	esw_2003_array_df = pd.concat([esw_2003_array_df, pd.DataFrame(esw_2003_array[k])], ignore_index=True, axis=1)
	esw_2013_array_df = pd.concat([esw_2013_array_df, pd.DataFrame(esw_2013_array[k])], ignore_index=True, axis=1)
	esw_array_df = pd.concat([esw_array_df, pd.DataFrame(esw_array[k])], ignore_index=True, axis=1)

lst_array_df.columns = conversions_tmp
myboxplot(
	df=lst_array_df,
	title="LST",
	ylabel="$\Delta$ ET [mm/year]",
	margin=0.3,
	outname="LST_Boxplot.png",
)

albedo_array_df.columns = conversions_tmp
myboxplot(
	df=albedo_array_df,
	title="Albedo",
	ylabel="$\Delta$ Albedo",
	margin=0.005,
	outname="Albedo_boxplot.png",
)

et_array_df.columns = conversions_tmp
myboxplot(
	df=et_array_df,
	title="ET",
	ylabel="$\Delta$ ET [mm/year]",
	margin=5,
	outname="ET_boxplot.png",
)

df1 = {
	"name": "LST",
	"df": lst_array_df,
	"label": "$\Delta$LST [C]",
	"ylim": [-12, 12],
}
df2 = {
	"name": "Albedo",
	"df": albedo_array_df,
	"label": "$\Delta$Albedo",
	"ylim": [-0.5, 0.5],
}
df3 = {
	"name": "ET",
	"df": et_array_df,
	"label": "$\Delta$ET [mm/year]",
	"ylim": [-850, 850],
}

columns = lst_array_df.columns.values
a = np.arange(0, len(columns)) % 2
txt_pos = np.where(a == 0, 7, 9)
myboxplot_group(df1, df2, df3, columns=columns,txt_pos=txt_pos, outname="Boxplot_groups.png")

df1 = {
	"name": "CI",
	"df": eci_array_df,
	"label": "$\Delta$CI [mm/year]]",
	"ylim": [-300, 300],
}
df2 = {
	"name": "SW",
	"df": esw_array_df,
	"label": "$\Delta$SW [mm/year]]",
	"ylim": [-300, 300],
}

a = np.arange(0, len(columns)) % 2
txt_pos = np.where(a == 0, 250, 230)
myboxplot_group(df1, df2, df3, columns=columns,txt_pos=txt_pos, outname="ET_Components.png")

"""
################## Monte Carlo approach ############################
# import time 
# t1 = time.time()

cof = []
IDs = np.arange(0,100000)
x = cnt_upper_final
y = lst_upper_clean
lazy_results=[]
for n in IDs:
	lazy_result = dask.delayed(my_cal)(n,x,y)
	lazy_results.append(lazy_result)
with ProgressBar():
	futures = dask.persist(*lazy_results)
	results = dask.compute(*futures)

final_cof = np.array(cof)
final_cof_mean = np.mean(final_cof,axis=0)
final_cof_std = np.std(final_cof,axis=0)
data_stacked = np.stack((final_cof_mean,final_cof_std),axis=0)
pd_upper_mc = pd.DataFrame(data = data_stacked,index=["cof_mean","cof_std"],columns=upper_name)
pd_upper_mc.to_csv(out_dir+"upper_OLS_results_mc.csv")


cof = []
x = cnt_lower_final
y = lst_lower_clean
lazy_results=[]
for n in IDs:
	lazy_result = dask.delayed(my_cal)(n,x,y)
	lazy_results.append(lazy_result)
with ProgressBar():
	futures = dask.persist(*lazy_results)
	results = dask.compute(*futures)
final_cof = np.array(cof)
final_cof_mean = np.mean(final_cof,axis=0)
final_cof_std = np.std(final_cof,axis=0)
data_stacked = np.stack((final_cof_mean,final_cof_std),axis=0)
pd_lower_mc = pd.DataFrame(data = data_stacked,index=["cof_mean","cof_std"],columns=lower_name)
pd_lower_mc.to_csv(out_dir+"lower_OLS_results_mc.csv")


cof = []
IDs = np.arange(0,100000)
x = cnt_all_final
y = lst_all_clean
lazy_results=[]
for n in IDs:
	lazy_result = dask.delayed(my_cal)(n,x,y)
	lazy_results.append(lazy_result)
with ProgressBar():
	futures = dask.persist(*lazy_results)
	results = dask.compute(*futures)

final_cof = np.array(cof)
final_cof_mean = np.mean(final_cof,axis=0)
final_cof_std = np.std(final_cof,axis=0)
data_stacked = np.stack((final_cof_mean,final_cof_std),axis=0)
pd_all_mc = pd.DataFrame(data = data_stacked,index=["cof_mean","cof_std"],columns=all_name)
pd_all_mc.to_csv(out_dir+"all_OLS_results_mc.csv")


################################### Removing water and sparse #############################

lower_list = np.array([7,14,15,21,22,23,28,29,30,31,35,36,37,38,39,40,43,44,45,46,47])
diagnoal_list = np.array([0,8,16,24,32,40,48])
upper_list = np.array([1,2,3,4,5,6,9,10,11,12,13,17,18,19,20,25,26,27,33,34,41])
ignore_list = np.array([0,6,8,13,16,20,24,27,32,34,40,41,42,43,44,45,46,47,48])


# in_dir = '/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/outputs/Sensitivity/EndPoints/'
# in_dir = 'F\\working\\LUC\\test\\'
ds = xr.open_dataset(in_dir+'Confusion_Table.nc')
Conversion = ds['Conversion'].values
Conversion_all = np.delete(Conversion,diagnoal_list)
lst_lulc = ds['LST_LULC'].values

lst_outliers = ~np.isnan(reject_outliers(lst_lulc,2))
lst_lulc_clean = lst_lulc[lst_outliers].squeeze()
normalized_confusion = ds['NORMALIZED_CONFUSION'].values*100
normalized_confusion_clean = normalized_confusion[lst_outliers].squeeze()

no_water_luc=normalized_confusion_clean
no_water_lst = lst_lulc_clean
for i in water_list:
	print(i)
	I = np.where(no_water_luc[:,i]==0)
	no_water_luc = no_water_luc[I]
	no_water_lst = no_water_lst[I]

no_water_luc = np.delete(no_water_luc,ignore_list,axis=1)
all_all_zeros = np.all((no_water_luc==0),axis=1)

cnt_no_water_clean = no_water_luc[~all_all_zeros]
lst_no_water_clean = no_water_lst[~all_all_zeros]

# from sklearn.preprocessing import PowerTransformer
# pt = PowerTransformer()
# pt.fit(cnt_no_water_clean)
cnt_no_water_clean_transform = preprocessing.scale(cnt_no_water_clean)

# Add constant term
one_all_col = np.ones(len(lst_no_water_clean))
cnt_no_water_final = np.c_[cnt_no_water_clean_transform,one_all_col]

results_no_water = sm.OLS(lst_no_water_clean, cnt_no_water_final).fit()
no_water_non_zero = np.apply_along_axis(my_count,0,cnt_no_water_final)

par_no_water = results_no_water.params
pvalue_no_water = results_no_water.pvalues
stand_err_no_water = results_no_water.bse
data_no_water = np.stack((par_no_water,pvalue_no_water,stand_err_no_water,no_water_non_zero),axis=0)

no_water_name= list(np.delete(Conversions,ignore_list,axis=0)) 
no_water_name.append("intercept")

pd_no_water = pd.DataFrame(data = data_no_water,index=["par_no_water","pvalue_no_water","stand_err_no_water","no_water_non_zero (>0)"],columns=no_water_name)
pd_no_water.to_csv(out_dir+"no_water_sparse_OLS_results_v3.csv")


a = np.delete(normalized_confusion_clean,diagnoal_list,axis=1)

Conversions[np.where(np.max(a,axis=0)<50)]
Conversions[np.where(np.max(a,axis=0)>50)]

"""