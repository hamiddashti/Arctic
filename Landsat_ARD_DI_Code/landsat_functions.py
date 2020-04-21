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
import rasterio
import rioxarray
import numpy as np
from rasterio.enums import Resampling
import os
import glob


def change_LC(xrfile, number_of_pixels):
    # Produces the change in LC over time and produce a table
    print("calculating the change in LC over time")
    date = pd.date_range(start="1984", periods=31, freq="Y")
    col_names = date.strftime("%Y")
    number_of_classes = np.arange(1, 11)
    # create empty dataframe
    df = pd.DataFrame(index=number_of_classes, columns=col_names)
    for i in np.arange(0, len(date)):
        # count the number of each class in each year and save them in a dataframe.
        b = xrfile.to_series().groupby("band").value_counts().loc[i + 1]
        df.iloc[:, i] = b

    # get the percent cover of each class
    df = (df / number_of_pixels) * 100
    # substitute the class numbers with class names ---> https://daac.ornl.gov/ABOVE/guides/Annual_Landcover_ABoVE.html
    class_nemes = [
        "Evergreen Forest",
        "Deciduous Forest",
        "Shrubland",
        "Herbaceous",
        "Sparsely Vegetated",
        "Barren",
        "Fen",
        "Bog",
        "Shallows/Littoral",
        "Water",
    ]

    # Plot the time series of changes in the percent cover ----------
    df.index = class_nemes
    df_T = df.T
    return df_T


# ---------- The mode function ----------------- #
# this function returns the mode of xarray object
# along a dimension
def _mode(*args, **kwargs):
    from scipy import stats

    vals = stats.mode(*args, **kwargs)
    # only return the mode (discard the count)
    return vals[0].squeeze()


def mode(obj, dim):
    # note: apply always moves core dimensions to the end
    # usually axis is simply -1 but scipy's mode function doesn't seem to like that
    # this means that this version will only work for DataArray's (not Datasets)
    # assert isinstance(obj, xr.DataArray)
    axis = obj.ndim - 1
    return xr.apply_ufunc(_mode, obj, input_core_dims=[[dim]], kwargs={"axis": axis})


####################################################


def open_dat(path, fname, nrow, ncol):

    # This function just opens the file
    path_to_file = path + fname
    da = xr.open_rasterio(path_to_file, chunks={"x": ncol, "y": nrow, "band": 1})
    return da


def my_plot_lulc_percent(df, ylabel, out_fig):
    # This function produce the bar chart of LCC over time from the output of change_LC function. 
    # Note that the format of the df should be like class percents are
    # in clolumns and the years are in rows
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    fsize = 14
    plt.rc("font", family="serif")
    # ------plot-------------------
    df.plot.bar(ax=ax1, stacked=True)
    # ------------------------------
    plt.ylabel(ylabel, fontsize=fsize)
    plt.xlabel("Time", fontsize=fsize)
    plt.xticks(fontsize=fsize, rotation=90)
    plt.yticks(fontsize=fsize)
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.subplots_adjust(right=0.85)
    plt.tight_layout()
    plt.savefig(out_fig)
    plt.close("all")


def my_plot_point(df, ylabel, out_fig):
    fsize = 5
    dpi = 300
    plt.figure(num=None, figsize=(3.2, 2.4), dpi=dpi, facecolor="w")
    plt.rc("font", family="serif")
    ax = plt.subplot()
    ax.plot(df, marker=".", linestyle="None", markersize=3)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel(ylabel, fontsize=fsize)
    plt.xlabel("Time", fontsize=fsize)
    plt.xticks(fontsize=fsize, rotation=90)
    plt.yticks(fontsize=fsize)
    plt.tight_layout()
    plt.savefig(out_fig)
    plt.close("all")


def my_plot_line(df, ylabel, out_fig):
    fsize = 5
    dpi = 300
    plt.figure(num=None, figsize=(3.2, 2.4), dpi=dpi, facecolor="w")
    plt.rc("font", family="serif")
    ax = plt.subplot()
    ax.plot(df, linestyle="solid")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel(ylabel, fontsize=fsize)
    plt.xlabel("Time", fontsize=fsize)
    plt.xticks(fontsize=fsize, rotation=90)
    plt.yticks(fontsize=fsize)
    plt.tight_layout()
    plt.savefig(out_fig)
    plt.close("all")


def month_stat(xfile, var_name, out_path):

    count = xfile.resample(
        time="1MS"
    ).count()  # Maximum number of observation per month for the entire time series
    count = count.where(count != 0)
    val = count.max(dim=["x", "y"]).values
    date = count.time.values
    df = pd.DataFrame(val, index=date)
    ylabel = "Maximum number of " + var_name + " in each month"
    out_fig = out_path + var_name + "_monthly_count.png"
    my_plot_point(df, ylabel, out_fig)

    var_mean = xfile.resample(time="1MS").mean()
    val = var_mean.mean(dim=["x", "y"]).values
    date = var_mean.time.values
    df = pd.DataFrame(val, index=date)
    ylabel = "mean of " + var_name + " in each month"
    out_fig = out_path + var_name + "_monthly_mean.png"
    my_plot_point(df, ylabel, out_fig)

    group_count = xfile.groupby("time.month").count()
    group_count = group_count.where(group_count != 0)
    val = group_count.max(dim=["x", "y"]).values
    date_tmp = group_count.month.values
    date = pd.to_datetime(date_tmp, format="%m").strftime("%B")
    df = pd.DataFrame(val, index=date)
    ylabel = "Number of " + var_name + " (entire time series)"
    out_fig = out_path + var_name + "_monthly_group_count.png"
    my_plot_point(df, ylabel, out_fig)

    group_mean = xfile.groupby("time.month").mean()
    val = group_mean.mean(dim=["x", "y"]).values
    date_tmp = group_mean.month.values
    date = pd.to_datetime(date_tmp, format="%m").strftime("%B")
    df = pd.DataFrame(val, index=date)
    ylabel = "Mean of " + var_name + " (entire time series)"
    out_fig = out_path + var_name + "_monthly_group_mean.png"
    my_plot_line(df, ylabel, out_fig)


def season_stat(xfile, var_name, out_path):

    # SEASONAL COUNT
    count = xfile.resample(time="QS-DEC").count()
    count = count.where(count != 0)
    val = count.max(dim=["x", "y"]).values
    date = count.time.values
    df = pd.DataFrame(val, index=date)
    SEASONS = {
        "winter": [12, 1, 2],
        "spring": [3, 4, 5],
        "summer": [6, 7, 8],
        "fall": [9, 10, 11],
    }
    MONTHS = {month: season for season in SEASONS.keys() for month in SEASONS[season]}
    df.index = pd.Series(df.index.month).map(MONTHS) + " " + df.index.year.astype(str)
    ylabel = "Maximum number of " + var_name + " in each season"
    out_fig = out_path + var_name + "_season_count.png"
    my_plot_point(df, ylabel, out_fig)

    # SEASONAL MEAN
    var_mean = xfile.resample(time="QS-DEC").mean()
    val = var_mean.mean(dim=["x", "y"]).values
    date = var_mean.time.values
    df = pd.DataFrame(val, index=date)
    SEASONS = {
        "winter": [12, 1, 2],
        "spring": [3, 4, 5],
        "summer": [6, 7, 8],
        "fall": [9, 10, 11],
    }
    MONTHS = {month: season for season in SEASONS.keys() for month in SEASONS[season]}
    df.index = pd.Series(df.index.month).map(MONTHS) + " " + df.index.year.astype(str)
    ylabel = "mean of " + var_name + " in each season"
    out_fig = out_path + var_name + "_season_mean.png"
    my_plot_point(df, ylabel, out_fig)

    # SEASONAL GROUP COUNT
    group_count = xfile.groupby("time.season").count()
    val = group_count.max(dim=["x", "y"]).values
    date = group_count.season
    df = pd.DataFrame(val, index=date)
    df = df.reindex(["DJF", "MAM", "JJA", "SON"])
    ylabel = "Number of " + var_name + " observations " + " (entire time series)"
    out_fig = out_path + var_name + "_season_group_count.png"
    my_plot_point(df, ylabel, out_fig)

    # SEASONAL GROUP MEAN
    group_mean = xfile.groupby("time.season").mean()
    val = group_mean.mean(dim=["x", "y"]).values
    date = group_mean.season
    df = pd.DataFrame(val, index=date)
    df = df.fillna(0)
    df = df.reindex(["DJF", "MAM", "JJA", "SON"])
    ylabel = "Mean of " + var_name + " (entire time series)"
    out_fig = out_path + var_name + "_season_group_mean.png"
    my_plot_point(df, ylabel, out_fig)


def reproject(file, target_crs, out_dir):
    # file: the absolute (full) path to the file (tif) that we want to reproject
    # target crs: the crs that we want to convert to. -----> df.rio.crs
    # out_dir: full path to the output directory

    f_name = os.path.basename(file)
    # print(f_name)
    df = xr.open_rasterio(file)
    tmp = df.rio.reproject(target_crs, resolution=30, resampling=Resampling.nearest)
    tmp.rio.to_raster(out_dir + f_name)
    # print(tmp.rio.crs)

