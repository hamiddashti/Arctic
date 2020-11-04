import xarray as xr 
import matplotlib.pylab as plt
import numpy as np
from scipy import stats 
import statsmodels.api as sm
import pandas
from sklearn.preprocessing import PowerTransformer
import dask


# in_dir = "F:\\working\\LUC\\test\\"
# out_dir = "F:\\working\\LUC\\test\\outputs\\"
in_dir = '/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/outputs/Sensitivity/EndPoints/'
out_dir = '/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/outputs/Sensitivity/EndPoints/'
# min_thresh = 1 # Minimum percent cover change for a conversion to be included in the OLS analyses
# var_type = "LST"

def lsg(predictors,target):
	# least square solution
	a = np.linalg.inv(np.matmul(predictors.T, predictors))
	b = np.matmul(predictors.T,target)
	coefficients = np.matmul(a,b)
	return coefficients

def reject_outliers(data, m):
	# m is number of std
	import numpy as np
	data = data.astype(float)
	data[abs(data - np.nanmean(data)) > m * np.nanstd(data)] = np.nan
	return data

def my_count(d):
	return len(d[np.where(d>10)])

ignore_list = np.array([0,7,8,14,15,16,21,22,23,24,28,29,30,31,32,35,36,37,38,39,40,42,43,44,45,46,47,48])
include_list = np.array([1,2,3,4,5,6,9,10,11,12,13,17,18,19,20,25,26,27,33,34,41])
# ds = xr.open_dataset('ds_2013_Endpoints.nc')
ds = xr.open_dataset(in_dir+"ds_2013_EndPoints.nc")
ds_shape = [ds.dims['time'],ds.dims['ID'],ds.dims['Conversions']]
Conversions = ds['Conversions'].values

# Anaylzing the EndPoints (changes at 2003 compared with 2013)
cnt = ds['confused_normalized_Total'].values.squeeze()*100
lst = ds['delta_var_total'].values.squeeze()

# Remove the unwanted columns
cnt = cnt[:,include_list]

# Removing outliers in the LST
lst_outliers = ~np.isnan(reject_outliers(lst,2))
cnt_clean = cnt[lst_outliers].squeeze()
lst_clean = lst[lst_outliers].squeeze()

print("Power Transform")
pt = PowerTransformer(method='yeo-johnson', standardize=True, copy=True)
pt.fit(cnt_clean)
cnt_transform = pt.transform(cnt_clean)

cof = []
def my_cal(n):
# for n in range(10):
	p = np.random.choice(np.arange(0,cnt_transform.shape[0]),10000)
	y = lst_clean[p]
	x = cnt_transform[p,:]
	tmp = lsg(predictors=x,target=y)
	cof.append(tmp)

import time 
t1 = time.time()

dask.config.set(scheduller = 'processes')
IDs = np.arange(0,10000)
lazy_results=[]
for n in IDs:
	lazy_result = dask.delayed(my_cal)(n)
	lazy_results.append(lazy_result)
from dask.diagnostics import ProgressBar
with ProgressBar():
	futures = dask.persist(*lazy_results)
	results = dask.compute(*futures)
final_cof = np.array(cof)
da = xr.DataArray(final_cof)
da.to_netcdf(out_dir + 'Final_Slopes.nc')

t2 = time.time()
print(t2-t1) 
print("All Done!")

"""
plt.close()
plt.figure(figsize=(8,6))
plt.hist(lst_clean,color = 'gray')
plt.xlabel("$\Delta$LST [C]")
plt.ylabel("Number of pixels")
plt.title("Histogram of $\Delta LST$")
plt.tight_layout()
plt.show()

# data = np.c_[lst_clean,cnt_clean]

results = sm.OLS(lst_clean, cnt_clean).fit()
print(results.summary())


pt = PowerTransformer(method='yeo-johnson', standardize=False, copy=True)
pt.fit(cnt_clean)
cnt_false = pt.transform(cnt_clean)

for i in range(5):
	plt.hist(cnt_true[:,i][np.where(cnt_true[:,i]>0)],bins=100)
	plt.show()

for i in range(5):
	plt.hist(cnt_false[:,i],bins=100)
	plt.show()


a1 = lsg(predictors=cnt_clean,target=lst_clean)
a2 = lsg(predictors=cnt_true,target=lst_clean)
a3 = lsg(predictors=cnt_false,target=lst_clean)




plt.scatter(a1,a2)
plt.show()



np.transpose(lst_clean)
.shape
col =np.hstack(("LST",Conversions[include_list]))

df = pandas.DataFrame(data,columns=np.concatenate("x",Conversions[include_list])


plt.bar(range(len(cof)),cof)
plt.xticks(range(len(cof)),["a",Conversions[include_list]],rotation = 'vertical')
plt.show()

df = pandas.DataFrame(data,columns=col)
df.head()


housing = pandas.read_csv(
    ## http://jse.amstat.org/v19n3/decock.pdf
    'https://raw.githubusercontent.com/nickkunz/smogn/master/data/housing.csv'
)

len(housing)
housing_smogn = smogn.smoter(
    data = housing,  ## pandas dataframe
    y = 'SalePrice'  ## string ('header name')
)
housing_smogn.shape


a = smogn.smoter(data= df,y="LST")
df.shape 
a.shape 







housing
########################### Analyses for all years ##########################
cnt_tmp = []
ct_tmp = []
lst_tmp = []
for i in range(ds.dims['time']):
	i
	tmp1 = ds.isel(time=i)["confused_normalized_Total"].squeeze().values
	tmp2 = ds.isel(time=i)["confused_total"].squeeze().values
	tmp3 = ds.isel(time=i)["delta_var_total"].squeeze().values
	cnt_tmp.append(tmp1)
	ct_tmp.append(tmp2)
	lst_tmp.append(tmp3)
cnt = np.array(cnt_tmp).reshape(ds_shape[0]*ds_shape[1],49)
ct = np.array(ct_tmp).reshape(ds_shape[0]*ds_shape[1],49)
lst = np.array(lst_tmp).reshape(ds_shape[0]*ds_shape[1])

# Removing outliers in the LST
lst_outliers = ~np.isnan(reject_outliers(lst,2))
cnt_clean = cnt[lst_outliers].squeeze()
ct_clean = ct[lst_outliers].squeeze()
lst_clean = lst[lst_outliers].squeeze()

#plot the histogram of var (LST)
plt.close()
plt.figure(figsize=(8,6))
plt.hist(lst_clean,color = 'gray')
plt.xlabel("$\Delta$LST [C]")
plt.ylabel("Number of pixels")
plt.title("Histogram of $\Delta LST$")
plt.tight_layout()
plt.show()
# plt.savefig(out_dir+"Histogram_delta_lst.png")

non_zeros = np.apply_along_axis(my_count,0,cnt_clean)
zero_counts = cnt_clean.shape[0]-non_zeros
zero_percent = np.round(((cnt_clean.shape[0]-non_zeros)/cnt_clean.shape[0])*100,2)

plt.close()
plt.figure(figsize=(12,8))
plt.bar(range(1,50),non_zeros)
plt.xticks(range(1,50),Conversions,rotation = 'vertical')
plt.ylabel("Number of non-zero datapoints")
plt.tight_layout()
for i, v in enumerate(non_zeros):
    plt.text(range(1,50)[i] - 0.25, v + 0.01, str(v),rotation=60)
# plt.show()
plt.savefig(out_dir+'Unbalanced_data.png')

plt.close()
plt.figure(figsize=(12,8))
plt.bar(range(1,50),zero_counts)
plt.xticks(range(1,50),Conversions,rotation = 'vertical')
plt.ylabel("Number of non-zero datapoints")
plt.tight_layout()
for i, v in enumerate(zero_percent):
    plt.text(range(1,50)[i] - 0.25, v + 10e6, str(v),rotation=60)
# plt.show()
plt.savefig(out_dir+'zero_inflated.png')

plt.close()
plt.hist(cnt_clean[:,4],bins=100)
plt.ylabel("Frequency (zeros included)")
plt.xlabel("Fractional change in land cover")
plt.savefig(out_dir+"hist_ef_shrub.png")
# plt.show()


#The upper/lower triangle and diagonal of the confusion matrix
include = [1,2,3,4,5,6,9,10,11,12,13,17,18,19,20,25,26,27,33,34,41] 			
not_include = [7,14,15,21,22,23,28,29,30,31,35,36,37,38,39,42,43,44,45,46,47] 	#The lower triangle of confusion matrix
no_change = [0,8,16,24,32,40,48]   	# The diagonal of the confudion matrix

x =cnt_clean 
y = lst_clean
for i in not_include:
	# Focusing on just one way of transition
	i
	I = np.where(x[:,i]==0)
	x = x[I,:].squeeze()
	y = y[I].squeeze()

xt = x[:,include]
xt = xt+1
xt_log = np.arcsinh(xt)


results = sm.OLS(y, xt).fit()
print(results.summary())
lsg(predictors=xt_log,target=y)

x0 = xt[np.where(xt[:,0]>0.1)]
y0 = y[np.where(xt[:,0]>0.1)]



x0 = x0[:,~np.all(x0 == 0, axis = 0)]
y0.shape




import seaborn as sns 
sns.regplot(x=xt[:,1], y=y) 
plt.show()


b, m = polyfit(xt[:,1],y, 1)
plt.scatter(xt[:,1],y)
plt.plot(xt[:,1], b + m * x, '-')
plt.show()
plt.hist(xxx,bins=100)
plt.show()


plt.hist(stats.boxcox(b,lmbda=-0.36992465608450925),bins=1000)
plt.show()
a,b = stats.boxcox(b)
b
xt = x[:,include]

xt = np.sqrt(xt)

lsg(predictors=xt,target=y)
results = sm.OLS(y, xt).fit()
print(results.summary())

###################################
# non_zeros = np.count_nonzero(filtered_x,axis=0)
non_zeros = np.apply_along_axis(my_count,0,filtered_x) # Pixels with percent cover change more than 1%
non_zeros_min_idx = np.argmin(non_zeros) # The index of column in X with minimum number of pixels


# Bar plots of number of non_zero pixels
plt.close()
plt.figure(figsize=(12,8))
plt.bar(range(1,50),non_zeros)
plt.xticks(range(1,50),Conversions,rotation = 'vertical')
plt.tight_layout()
for i, v in enumerate(non_zeros):
    plt.text(range(1,50)[i] - 0.25, v + 0.01, str(v),rotation=60)
# plt.show()
plt.savefig(out_dir+'bar_plot_2004.png')

cx = filtered_x[:,np.argmin(non_zeros)] # Column with minimum number of conversions
print(f"The '{Conversions[non_zeros_min_idx]}' class has the minimum number of pixels in the confusion table")
idx = filtered_x[np.where(cx>min_thresh)]
idy = filtered_y[np.where(cx>min_thresh)]

min_len = len(idy) # Minimum lenght of the X matrix for under-sampling 



X = []
Y = []
for i in range(0,49):
	if i==43:
		X.append(idx)
		Y.append(idy)
		continue
	tmp_x = filtered_x[:,i]
	I = np.where(tmp_x > min_thresh)
	tmp_idx = filtered_x[I,:].squeeze()
	tmp_idy = filtered_y[I]
	rand_num = np.random.choice(np.arange(0,tmp_idx.shape[0]),size=min_len)
	tmp_idx_rand = tmp_idx[rand_num,:]
	tmp_idy_rand = tmp_idy[rand_num]
	X.append(tmp_idx_rand)
	Y.append(tmp_idy_rand)

a = np.asarray(X).squeeze()
b = np.asarray(Y)
xx = a.reshape(a.shape[0]*a.shape[1],len(Conversions))
yy = b.reshape(b.shape[0]*b.shape[1]) 
x_final = np.c_[xx,np.ones(xx.shape[0])]






plt.close()
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
prob = stats.probplot(x,dist=stats.norm,plot=ax1)
prob_t = stats.probplot(xt,dist=stats.norm,plot=ax2)
plt.show()


min(x)

lsg(predictors=xx,target=yy)
results = sm.OLS(yy, xx).fit()
print(results.summary())

np.sum(xx,axis=1)






plt.close()
fig, axs = plt.subplots(7,7, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.001)
axs = axs.ravel()
for i in range(len(Conversions)):
    axs[i].hist(xx[:,i])
    # axs[i].set_title(str(250+i))
plt.show()

x_final = np.c_[xx,np.ones(xx.shape[0])]  # add bias term


y =np.asarray([1,2,3,4,3,4,5,4,5,5,4,5,4,5,4,5,6,5,4,5,4,3,4])
x = np.transpose(np.asarray([
     [4,2,3,4,5,4,5,6,7,4,8,9,8,8,6,6,5,5,5,5,5,5,5],
     [4,1,2,3,4,5,6,7,5,8,7,8,7,8,7,8,7,7,7,7,7,6,5],
     [4,1,2,5,6,7,8,9,7,8,7,8,7,7,7,7,7,7,6,6,4,4,4]
     ]))

x_final = np.c_[x,np.ones(x.shape[0])]

lsg(predictors=x_final,target=y)

results = sm.OLS(y, x_final).fit()
print(results.summary())

a=xx[:,0][np.where(xx[:,0]!=0)]
plt.close()
plt.hist(np.log(a))
# plt.hist(xx[:,0])
plt.show()


################ Goog Stuff Below #######################

"""