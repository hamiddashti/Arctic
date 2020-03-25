# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 11:24:46 2020

@author: hamiddashti
"""
import xarray as xr
import matplotlib.pyplot as plt
import dask
from timeit import default_timer as timer
from matplotlib.ticker import MaxNLocator
import pandas as pd

def open_dat (path,fname,nrow,ncol):
   
   #This function just opens the file
   path_to_file = path+fname
   da = xr.open_rasterio(path_to_file,chunks={'x': ncol, 'y': nrow,'band': 1})
   return da
   
def my_plot_point(df,ylabel,out_fig):
    fsize = 5
    dpi=300
    plt.figure(num=None, figsize=(3.2, 2.4),dpi=dpi, facecolor="w")
    plt.rc('font', family='serif')
    ax = plt.subplot()
    ax.plot(df, marker=".", linestyle="None", markersize=3)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel(ylabel,fontsize=fsize)
    plt.xlabel("Time",fontsize=fsize)
    plt.xticks(fontsize=fsize, rotation=90)
    plt.yticks(fontsize=fsize)
    plt.tight_layout()
    plt.savefig(out_fig)
    plt.close('all')

def my_plot_line(df,ylabel,out_fig):
    fsize = 5
    dpi=300
    plt.figure(num=None, figsize=(3.2, 2.4), dpi=dpi, facecolor="w")
    plt.rc('font', family='serif')
    ax = plt.subplot()
    ax.plot(df, linestyle="solid")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel(ylabel, fontsize=fsize)
    plt.xlabel("Time", fontsize=fsize)
    plt.xticks(fontsize=fsize, rotation=90)
    plt.yticks(fontsize=fsize)
    plt.tight_layout()
    plt.savefig(out_fig)
    plt.close('all')


def month_stat(xfile,var_name,out_path):
    
    count = xfile.resample(time="1MS").count()  # Maximum number of observation per month for the entire time series
    count = count.where(count != 0)
    val = count.max(dim=["x", "y"]).values
    date = count.time.values
    df = pd.DataFrame(val, index=date)
    ylabel = 'Maximum number of '+var_name+ ' in each month'
    out_fig = out_path+var_name+'_monthly_count.png'
    my_plot_point(df,ylabel,out_fig)

    var_mean = xfile.resample(time="1MS").mean()
    val = var_mean.mean(dim=["x", "y"]).values
    date = var_mean.time.values
    df = pd.DataFrame(val, index=date)
    ylabel = 'mean of ' + var_name + ' in each month'
    out_fig = out_path+var_name+'_monthly_mean.png'
    my_plot_point(df,ylabel,out_fig)

    group_count = xfile.groupby("time.month").count()
    group_count = group_count.where(group_count != 0)
    val = group_count.max(dim=["x", "y"]).values
    date_tmp = group_count.month.values
    date = pd.to_datetime(date_tmp, format="%m").strftime("%B")
    df = pd.DataFrame(val, index=date)
    ylabel = 'Cumulative number of ' + var_name + ' (entire time series)'
    out_fig = out_path+var_name+'_monthly_group_count.png'
    my_plot_point(df,ylabel,out_fig)

    group_mean = xfile.groupby("time.month").mean()
    val = group_mean.mean(dim=["x", "y"]).values
    date_tmp = group_mean.month.values
    date = pd.to_datetime(date_tmp, format="%m").strftime("%B")
    df = pd.DataFrame(val, index=date)
    ylabel = 'Mean of ' + var_name + ' (entire time series)'
    out_fig = out_path+var_name+'_monthly_group_mean.png'
    my_plot_line(df,ylabel,out_fig)


def season_stat(xfile,var_name,out_path):
    
    count = xfile.resample(time='QS-DEC').count()
    count = count.where(count != 0)
    val = count.max(dim=["x", "y"]).values
    date = count.time.values
    df = pd.DataFrame(val, index=date)
    SEASONS = {
        'winter': [12, 1, 2],
        'spring': [3, 4, 5],
        'summer': [6, 7, 8],
        'fall': [9, 10, 11]
    }
    MONTHS = {month: season for season in SEASONS.keys()
                            for month in SEASONS[season]}

    df.index = (pd.Series(df.index.month).map(MONTHS) +
                    ' ' + df.index.year.astype(str))
    ylabel = 'Maximum number of '+var_name+ ' in each season'
    out_fig = out_path+var_name+'_season_count.png'
    my_plot_point(df,ylabel,out_fig)
    
    var_mean = xfile.resample(time='QS-DEC').mean()
    val = var_mean.max(dim=["x", "y"]).values
    date = var_mean.time.values
    df = pd.DataFrame(val, index=date)
    SEASONS = {
        'winter': [12, 1, 2],
        'spring': [3, 4, 5],
        'summer': [6, 7, 8],
        'fall': [9, 10, 11]
    }
    MONTHS = {month: season for season in SEASONS.keys()
                            for month in SEASONS[season]}

    df.index = (pd.Series(df.index.month).map(MONTHS) +
                    ' ' + df.index.year.astype(str))
    ylabel = 'mean of ' + var_name + ' in each season'
    out_fig = out_path+var_name+'_season_mean.png'
    my_plot_point(df,ylabel,out_fig)


    group_count = xfile.groupby("time.month").count()
    group_count = group_count.where(group_count != 0)
    val = group_count.max(dim=["x", "y"]).values
    date_tmp = group_count.month.values
    date = pd.to_datetime(date_tmp, format="%m").strftime("%B")
    df = pd.DataFrame(val, index=date)
    ylabel = 'Cumulative number of ' + var_name + ' observations '+' (entire time series)'
    out_fig = out_path+var_name+'_season_group_count.png'
    my_plot_point(df,ylabel,out_fig)

    group_mean = xfile.groupby("time.month").mean()
    val = group_mean.mean(dim=["x", "y"]).values
    date_tmp = group_mean.month.values
    date = pd.to_datetime(date_tmp, format="%m").strftime("%B")
    df = pd.DataFrame(val, index=date)
    ylabel = 'Mean of ' + var_name + ' (entire time series)'
    out_fig = out_path+var_name+'_season_group_mean.png'
    my_plot_line(df,ylabel,out_fig)

