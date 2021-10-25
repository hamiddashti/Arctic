from matplotlib.pyplot import axis
import xarray as xr
import numpy as np
import matplotlib.pylab as plt
from skbio.stats.composition import multiplicative_replacement, closure
import comp_funcs
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import statsmodels.api as sm
import pandas as pd 
from scipy.stats import pearsonr



in_dir = "/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/test/"
out_dir = "/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/test/"
X_pd = pd.read_csv(in_dir + "features.csv")
y_pd = pd.read_csv(in_dir + "target.csv")
X = X_pd.values
y = y_pd.values

model = comp_funcs.comp_reg(X, y, method="robust")
coef = model["coef"]

tmp = np.zeros((X.shape[0],X.shape[1]))
tmp[:] = np.nan
for i in range(0,10):
    Xj = np.column_stack((X[:, i], np.delete(X, i, axis=1))) 
    z = comp_funcs.pivotCoord(Xj)
    a = z[:,0]
    tmp[:,i] = coef[i+1]*a

y
y_pred = coef[0]+tmp.sum(axis=1)

corr,_ = pearsonr(y_pred,np.squeeze(y))
corr**2
corr 

plt.close()
plt.scatter(y,y_pred)
plt.savefig(out_dir+"test.png")
r2_score(np.squeeze(y),y_pred)



x_val = sm.add_constant(X)
pred = np.multiply(coef.values, x_val).sum(axis=1)




#####################################################
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


in_dir = "/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/test/"
out_dir = "/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/test/"

data = xr.open_dataset(in_dir + "data.nc")
x_raw = data["lc_2003_2013"].sel(year=2003).values
y_raw = data["lst_2003_2013"].sel(year=2003).values

I_good = ~outliers_index(y_raw, 2)
y_clean = y_raw[I_good]
x_clean = x_raw[I_good, ]

np.random.seed(1)
idx = np.random.permutation(len(y_clean))
x_shuffled, y_shuffled = x_clean[idx, ], y_clean[idx]

X = multiplicative_replacement(x_shuffled, 0.005)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y_shuffled,
                                                    test_size=0.2,
                                                    random_state=2)

model = comp_funcs.comp_reg(X_train, y_train, method="classic")

X_val = sm.add_constant(X_train)
coef = model["coef"]
pred = np.multiply(coef.values, X_val).sum(axis=1)

plt.close()
plt.hist(pred)
plt.savefig(out_dir + "test.png")
y_train
pred

nsample = 50
sig = 0.25
x1 = np.linspace(0, 20, nsample)
X = np.column_stack((x1, np.sin(x1), (x1 - 5)**2))
X = sm.add_constant(X)
beta = [5.0, 0.5, 0.5, -0.02]
y_true = np.dot(X, beta)
y = y_true + sig * np.random.normal(size=nsample)

olsmod = sm.OLS(y, X)
olsres = olsmod.fit()
print(olsres.summary())

ypred = olsres.predict(X)

olsres.params
np.multiply(olsres.params, X).sum(axis=1)
ypred

import pandas as pd

