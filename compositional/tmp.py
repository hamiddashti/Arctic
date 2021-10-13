import numpy as np
from numpy.core.fromnumeric import shape
import skbio.stats.composition as sk
import pandas as pd 
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm



in_dir = "/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/test/"
X_pd = pd.read_csv(in_dir+"features.csv")
y_pd = pd.read_csv(in_dir+"target.csv")
X = X_pd.values
y = y_pd.values



def pivotCoord(x):
    """Isometric logratio transformation simillar to RobComposition in R"""
    if x.ndim==1:
        D = x.shape[0]
        x_ilr = np.zeros((x.shape[0]-1))
        x_ilr[:]=np.nan
        for i in range(1,D):
            x_ilr[i-1] = np.sqrt((D-i)/(D-i+1)) * np.log(x[i-1]/(np.prod(x[i:])**(1/(D-i))))
        return x_ilr
    num_rows, num_cols = x.shape
    x_ilr = np.zeros((num_rows,num_cols-1))
    x_ilr[:]=np.nan
    D = num_cols
    for i in range(1,D):
        x_ilr[:,i-1] = np.sqrt((D-i)/(D-i+1)) * np.log(x[:,i-1]/(np.prod(x[:,i:],axis=1)**(1/(D-i))))
    return x_ilr

def results_summary_to_dataframe(results):
    '''take the result of an statsmodel results table and transforms it into a dataframe'''
    pvals = results.pvalues
    coeff = results.params
    conf_lower = results.conf_int()[0]
    conf_higher = results.conf_int()[1]

    results_df = pd.DataFrame({"pvals":pvals,
                               "coeff":coeff,
                               "conf_lower":conf_lower,
                               "conf_higher":conf_higher
                                })

    #Reordering...
    results_df = results_df[["coeff","pvals","conf_lower","conf_higher"]]
    return results_df


def comp_reg(X,y,method="classic"):
    if method=="classic":
        xx = sm.add_constant(X)
        results = sm.OLS(y,xx).fit()
        results_summary = results.summary()
        results_as_html = results_summary.tables[1].as_html()
        results_df = pd.read_html(results_as_html, header=0, index_col=0)[0]
        for j in range(0,X.shape[1]):
            Zj = np.column_stack((X[:,j],np.delete(X,j,axis=1)))
            dj = sm.add_constant(pivotCoord(Zj))
            res = sm.OLS(y,dj).fit()
            res_as_html = res.summary().tables[1].as_html()
            res_df = pd.read_html(res_as_html, header=0, index_col=0)[0]
            if j==0:
                results_df.iloc[0:2]=res_df.iloc[0:2]
            results_df.iloc[j+1] = res_df.iloc[1]   
    if method=="robust":
        xx = sm.add_constant(X)
        results = sm.RLM(y,xx).fit()
        results_summary = results.summary()
        results_as_html = results_summary.tables[1].as_html()
        results_df = pd.read_html(results_as_html, header=0, index_col=0)[0]
        for j in range(0,X.shape[1]):
            Zj = np.column_stack((X[:,j],np.delete(X,j,axis=1)))
            dj = sm.add_constant(pivotCoord(Zj))
            res = sm.RLM(y,dj).fit()
            res_as_html = res.summary().tables[1].as_html()
            res_df = pd.read_html(res_as_html, header=0, index_col=0)[0]
            if j==0:
                results_df.iloc[0:2]=res_df.iloc[0:2]
            results_df.iloc[j+1] = res_df.iloc[1]   
    return results_df

comp_reg(X,y,method="classic")
comp_reg(X,y,method="robust")


xx = sm.add_constant(X)
results = sm.RLM(y,xx).fit()
results_summary = results.summary()
results_as_html = results_summary.tables[1].as_html()
results_df = pd.read_html(results_as_html, header=0, index_col=0)[0]
results_df
for j in range(0,X.shape[1]):
    Zj = np.column_stack((X[:,j],np.delete(X,j,axis=1)))
    dj = sm.add_constant(pivotCoord(Zj))
    res = sm.RLM(y,dj).fit()
    res_as_html = res.summary().tables[1].as_html()
    res_df = pd.read_html(res_as_html, header=0, index_col=0)[0]
    if j==0:
        results_df.iloc[0:2]=res_df.iloc[0:2]
    else:
        results_df.iloc[j+1] = res_df.iloc[1]   

results_df