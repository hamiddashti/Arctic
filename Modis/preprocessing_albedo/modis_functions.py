# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 14:49:29 2019

This function generates the wget bash file file for downloading modis in bulk
Inputs are: 
	prodcut: name of the modis product (e.g. product = 'MCD15A3H.006')
	folder: Name of the folder on NASA ftp site which contain the products (e.g.
	folder = 'MOTA')
	tiles = list of the tile number (e.g. 
	
	tiles=['h12v01','h13v01','h14v01','h15v01','h16v01',
	
	'h10v02','h11v02','h12v02','h13v02','h14v02','h15v02',
	
	'h09v03','h10v03','h11v03','h12v03','h13v03','h14v03',
	
	'h08v04','h09v04','h10v04','h11v04','h12v04','h13v04','h14v04']
	)
	dates: list of start and end date in the following format:dates = ["2002.07.23", "2002.08.05"]
	freq: Frequency of the data which is basically the temporal resolution of modis
	(e.g. freq = '4D' which means this LAI dataset collects data every 4 days)
	
EXAMPLE: 

	
folder = 'MOTA'
product = 'MCD15A3H.006'
out_dir = '/run/user/1008/gvfs/smb-share:server=gaea,share=projects,user=hamiddashti/nasa_above/working/modis_analyses/my_data/'
tiles=['h12v01','h13v01','h14v01','h15v01','h16v01',
	
	'h10v02','h11v02','h12v02','h13v02','h14v02','h15v02',
	
	'h09v03','h10v03','h11v03','h12v03','h13v03','h14v03',
	
	'h08v04','h09v04','h10v04','h11v04','h12v04','h13v04','h14v04']

start_date='12/15/2002'
end_date='1/10/2003'
freq = '4D'

modis_wget_generator(product,folder,tiles,start_date,end_date,out_dir)
The output is a bash file (test.sh) which can be run as ./test.sh 

@author: hamiddashti
"""
# -----------------------------------------------------------------------
# This section is for calculating the weighted seasonal mean
# The following funtions are for calculating the number of days of each month
# -----------------------------------------------------------------------
dpm = {
    "noleap": [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    "365_day": [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    "standard": [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    "gregorian": [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    "proleptic_gregorian": [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    "all_leap": [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    "366_day": [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    "360_day": [0, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
}


def leap_year(year, calendar="standard"):
    """Determine if year is a leap year"""
    leap = False
    if (calendar in ["standard", "gregorian", "proleptic_gregorian", "julian"
                     ]) and (year % 4 == 0):
        leap = True
        if ((calendar == "proleptic_gregorian") and (year % 100 == 0)
                and (year % 400 != 0)):
            leap = False
        elif ((calendar in ["standard", "gregorian"]) and (year % 100 == 0)
              and (year % 400 != 0) and (year < 1583)):
            leap = False
    return leap


def get_dpm(time, calendar="standard"):
    """
	return a array of days per month corresponding to the months provided in `months`
	"""
    import numpy as np

    month_length = np.zeros(len(time), dtype=np.int)

    cal_days = dpm[calendar]

    for i, (month, year) in enumerate(zip(time.month, time.year)):
        month_length[i] = cal_days[month]
        if leap_year(year, calendar=calendar):
            month_length[i] += 1
    return month_length


def weighted_season_group(ds):
    # Make a DataArray with the number of days in each month, size = len(time)
    import xarray as xr

    month_length = xr.DataArray(
        get_dpm(ds.time.to_index(), calendar="noleap"),
        coords=[ds.time],
        name="month_length",
    )
    # Calculate the weights by grouping by 'time.season'
    weights = (month_length.groupby("time.season") /
               month_length.groupby("time.season").sum())
    # Calculate the weighted average
    season_grouped = (ds * weights).groupby("time.season").sum(dim="time",
                                                               skipna=False)
    return season_grouped


def weighted_season_resmaple(ds):
    # Make a DataArray with the number of days in each month, size = len(time)
    import xarray as xr

    month_length = xr.DataArray(
        get_dpm(ds.time.to_index(), calendar="noleap"),
        coords=[ds.time],
        name="month_length",
    )
    season_resample = (ds * month_length).resample(time="QS-DEC").sum() / (
        month_length.where(ds.notnull()).resample(time="QS-DEC").sum())

    return season_resample


# -----------------------------------------------------------------------


# -----------------------------------------------------------------------------------
# This part is for plotting the percent cover chage vs. changes in LST, albedo, etc
# -----------------------------------------------------------------------------------
def reject_outliers(data, m):
    # m is number of std
    import numpy as np
    data = data.astype(float)
    data[abs(data - np.nanmean(data)) > m * np.nanstd(data)] = np.nan
    return data


def growing_season(da):
    da_grouped = da.where(da.time.dt.month.isin(
        [3, 4, 5, 6, 7, 8, 9, 10,
         11]))  # This line set other months than numbered to nan
    da_growing = da_grouped.groupby("time.year").mean()
    return da_growing


def modis_wget_generator(product, folder, tiles, dates, out_dir):

    import pandas as pd
    import numpy as np
    from datetime import datetime

    # import os

    def return_url(url):
        # import time
        try:
            import urllib.request as urllib2
        except ImportError:
            import urllib2
        # import logging

        # LOG = logging.getLogger( __name__ )
        # the_day_today = time.asctime().split()[0]
        # the_hour_now = int(time.asctime().split()[3].split(":")[0])
        # if the_day_today == "Wed" and 14 <= the_hour_now <= 17:
        #    LOG.info("Sleeping for %d hours... Yawn!" % (18 - the_hour_now))
        #    time.sleep(60 * 60 * (18 - the_hour_now))

        req = urllib2.Request("%s" % (url), None)
        html = urllib2.urlopen(req).readlines()
        return html

    # wget --user=hamiddashti --password=Iran1140 -p /tmp -r -nd --no-parent -A "*h11v06.006*.hdf" http://e4ftl01.cr.usgs.gov/MOTA/MCD15A3H.006/2002.07.04/
    url = "https://e4ftl01.cr.usgs.gov/"
    """
	dates=pd.date_range(start=start_date, end=end_date,freq = '4D')
	df = pd.DataFrame(dict(date_given=dates))
	df['day_of_year'] = df['date_given'].dt.dayofyear
	df['date_given'] = df['date_given'].dt.strftime('%Y.%m.%d')
	df['year'] = pd.to_datetime(df['date_given'])
	df['year'] = df['year'].dt.year
	
	year1 = df['year'][0]
	year2 = df['year'][len(df)-1]
	years = np.arange(year1,year2+1,1)
	"""
    url_tmp = url + folder + "/" + product + "/"
    html = return_url(url_tmp)

    # dates = ["2002.07.23", "2002.08.05"]
    start = datetime.strptime(dates[0], "%Y.%m.%d")
    end = datetime.strptime(dates[1], "%Y.%m.%d")
    modis_date = []
    for line in html:
        if line.decode().find("href") >= 0 and line.decode().find(
                "[DIR]") >= 0:

            the_date = line.decode().split('href="')[1].split('"')[0].strip(
                "/")
            tmp_date = datetime.strptime(the_date, "%Y.%m.%d")
            if start <= tmp_date <= end:
                modis_date.append(tmp_date)

    df = pd.DataFrame(dict(dates_available=modis_date))
    df["day_of_year"] = df["dates_available"].dt.dayofyear
    df["dates_available"] = df["dates_available"].dt.strftime("%Y.%m.%d")
    df["year"] = pd.to_datetime(df["dates_available"])
    df["year"] = df["year"].dt.year

    year1 = df["year"][0]
    year2 = df["year"][len(df) - 1]
    years = np.arange(year1, year2 + 1, 1)

    # for k in np.arange(len(years)):
    #    tmp_dir = str(years[k])
    #    new_dir = out_dir+tmp_dir
    #    os.mkdir(new_dir)

    f_path = []
    for n in np.arange(len(df)):
        f_tmp = out_dir + str(df["year"][n])
        f_path.append(f_tmp)

    name = []
    for i in np.arange(len(tiles)):
        for j in np.arange(len(modis_date)):
            tmp1 = ("wget --user=hamiddashti --password=Iran1140 -P " +
                    f_path[j] + " -r -nd --no-parent -A")
            tmp2 = (' "*' + tiles[i] + '*.hdf" ' + url + folder + "/" +
                    product + "/" + str(df["dates_available"][j]) + "/ -q")
            name_tmp = tmp1 + tmp2
            name.append(name_tmp)

    total_line = len(name)
    line10th = np.ceil(len(name) / 100)
    progress = np.arange(0, total_line, line10th)

    acc = 0
    x = 1
    for k in np.arange(len(progress)):
        if acc == 0:
            name.insert(
                0,
                'start_time="$(date -u +%s)"' + "; echo Downloading the " +
                product + " started\n\n",
            )
            acc += 1
        else:
            ins_char = "echo " + str(
                x) + " percent of requested data is downloaded"
            some_info = (
                ins_char + '; end_time="$(date -u +%s)"' +
                '; elapsed="$(($end_time-$start_time))"' +
                '; echo "Total of $elapsed seconds elapsed for downloading"')
            name.insert(int(progress[k] + acc), some_info)
            acc += 1
            x += 1

    name.insert(
        len(name) + 1,
        "echo " + "Downloading finished.Data stored in: " + out_dir)

    outname = out_dir + "/Download_script.sh"

    with open(outname, "w") as filehandle:
        for listitem in name:
            filehandle.write("%s\n" % listitem)


"""
This script includes some functions that I developed for processing MODIS
images including: 
	
	(1) lai_filter(filepath,out,std): 
		inputs: 
			filepat: Complete path to the file (including the name of the file)
			out: True/False flag for saving file on the disk (same path as input)
			std: True/False flag for returning the STD of the LAI
		
		outputs: 
			LAI and LAI standard deviation if requested as DataSet or DataArray
			
		Example:
			
			from modis_functions import lai_filter
			data_lai = lai_filter(filepath='P:\\nasa_above\\working\\modis_analyses\\MyTest.nc',out=True, std=True)
		   
		** The nc file can be read using xr.open_dataarray
		
	
				
			
"""
##################


def lai_filter(filepath, out, std):

    import os
    import xarray as xr

    # import collections
    # filepath = 'P:\\nasa_above\\working\\modis_analyses\\MyTest.nc'
    fname = os.path.basename(filepath)
    outname_lai = filepath.replace(
        fname,
        fname.replace(".nc", "") + "_lai_filtered.nc")

    # Read the nc file as xarray DataSet
    ds = xr.open_dataset(filepath)

    # Read the layers from the dataset (DataArray)
    FparLai_QC = ds["FparLai_QC"]
    lai = ds["Lai_500m"]

    # ------------------ Filtering -------------------------------------------
    print("\n\nStarted filtering LAI: takes some time")

    # These values come from the LAI manual for quality control
    # returns the quality value (0,2,32 and 34) and nan for every other quality
    # https://lpdaac.usgs.gov/documents/2/mod15_user_guide.pdf
    lai_flag = FparLai_QC.where((FparLai_QC.values == 0)
                                | (FparLai_QC.values == 2)
                                | (FparLai_QC.values == 32)
                                | (FparLai_QC.values == 34))

    # Convert it Boolean
    lai_flag = lai_flag >= 0
    # Convert Boolean to zero (bad quality) and one (high quality)
    lai_flag = lai_flag.astype(int)
    lai_tmp = lai_flag * lai
    # replace all zeros with nan values
    lai_final = lai_tmp.where(lai_tmp.values != 0)
    lai_final = lai_final.rename("LAI")

    # Did use asked for standard deviation of lai too?
    if std == True:
        lai_std = ds["LaiStdDev_500m"]
        print("\n\nStarted filtering LAI Standard Deviation: takes some time")
        lai_std_tmp = lai_flag * lai_std
        lai_std_final = lai_std_tmp.where(lai_std_tmp.values != 0)
        lai_std_final = lai_std_final.rename("LAI_STD")
        lai_dataset = xr.merge([lai_final, lai_std_final])
        outname_std_lai = filepath.replace(
            fname,
            fname.replace(".nc", "") + "_lai_dataset.nc")

    # -------------------- OUTPUTS ------------------------------------------
    print("\n\n Figuring out the outputs:")
    if std == True and out == True:
        print('\n   ---writing the lai and lai_std as a "dataset" on the disk')
        lai_dataset.to_netcdf(outname_std_lai)
        return lai_dataset
    elif std == False and out == True:
        print("\n   ---wirintg just lai to the disk (no STD) and return lai")
        lai_final.to_netcdf(outname_lai)
        return lai_final
    elif std == True and out == False:
        print('\n   ---return lai and lai_std as a "dataset"')
        return lai_dataset
    elif std == False and out == False:
        print(
            "\n   ---return lai (not standard deviation) and no writing on the disk"
        )
        return lai_final


#######################End of lai_filter function ###################################

####################### Clipping tif files based on the shapefile ###################
"""
The following function is to clip the tif files (produced by pymodis) based on the shape file

Here are some point:
- shape files are the intersection between the Above domain and modis tiles
- the name of the shapefiles are the based on modis tiles (e.g. h11v02.shp)
- Before running the code its necessary to create year folder in the out_dir_tmp 
The following is an example of how to run the code: 

		
import sys
sys.path.append('/data/home/hamiddashti/mnt/nasa_above/Arctic/Modis')
import modis_functions
import numpy as np
import glob
import os

tif_dir =  '/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/tmp/Tifs/'
out_dir_tmp = '/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/tmp/Tifs/clipped/'
shp_dir = '/data/home/hamiddashti/mnt/nasa_above/Study_area/Clean_study_area/'
year1 = 2002
year2 = 2004
years_list = np.arange(year1,year2)

tiles_list=['h12v01','h13v01','h14v01','h15v01','h16v01',

	'h09v02','h10v02','h11v02','h12v02','h13v02','h14v02','h15v02',

	'h09v03','h10v03','h11v03','h12v03','h13v03',

	'h11v04']

#tile='h12v01'
#year=2002

for year in years_list:

   in_dir = tif_dir+str(year)+'/'
   out_dir = out_dir_tmp+str(year)+'/'



   for tile in tiles_list:
	   tmp_tif = in_dir+'*'+tile+'*.tif'
	   file_list = glob.glob(tmp_tif)
	   shp_file = shp_dir+tile+'.shp'

	   for file in file_list:
		   base_name = os.path.basename(file)
		   print(file)
		   modis_functions.tif_clip(base_name,shp_file,in_dir,out_dir)
"""


def tif_clip(tif_file, shp_file, outname):

    import rasterio
    from rasterio.mask import mask
    import geopandas as gpd
    import pycrs

    # rastin = tif_file  # The input raster
    # data = rasterio.open(rastin)
    # geo = gpd.read_file(shp_file)  # the shp file

    def getFeatures(shp_file):
        """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
        import json
        return [json.loads(shp_file.to_json())["features"][0]["geometry"]]

    coords = getFeatures(shp_file)
    out_img, out_transform = mask(tif_file, shapes=coords, crop=True)
    out_meta = tif_file.meta.copy()
    epsg_code = 102001
    # int(tif_file.crs.data["init"][5:])
    out_meta.update({
        "driver": "GTiff",
        "height": out_img.shape[1],
        "width": out_img.shape[2],
        "transform": out_transform,
        # "crs": pycrs.parse.from_epsg_code(epsg_code).to_proj4(),
        "crs": tif_file.crs
    })
    # out_file = outname
    with rasterio.open(outname, "w", **out_meta) as dest:
        dest.write(out_img)


""" -------------------------------------------------------------------------
Mosaicing tif files:
mosaicing(in_dir, out_dir, fnames, out_name)

in_dir --->     directory where tif files are
out_fire --->   output directory
fnames --->     list of the name of the tif files 
out_name --->   Name for the output mosaic file
--------------------------------------------------------------------------"""


def mosaicing(out_dir, fnames, out_name, nodata):

    from rasterio.merge import merge
    import rasterio
    import numpy as np

    src_files_to_mosaic = []
    for f in fnames:
        src = rasterio.open(f)
        src_files_to_mosaic.append(src)
    mosaic, out_trans = merge(src_files_to_mosaic, nodata=nodata)
    out_meta = src.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "crs": 102001,
    })
    with rasterio.open(out_dir + out_name, "w", **out_meta) as dest:
        dest.write(mosaic)


# ----------------------------------------------------------------------------
# All plots used for modis analysis
# ---------------------------------------------------------------------------
def meshplot(ds, outname, mode, label, title):
    import matplotlib.pyplot as plt

    if mode == "paper":
        import matplotlib as mpl

        mpl.rcParams.update(mpl.rcParamsDefault)
        fsize = 14
        plt.figure(num=None, figsize=(12, 8), facecolor="w")
        plt.rc("font", family="serif")
        im = ds.plot.pcolormesh(add_colorbar=False)
        cb = plt.colorbar(im, orientation="vertical", pad=0.05)
        cb.set_label(label=label, size=fsize, weight="bold")
        plt.title(title, fontsize=fsize)
        plt.xlabel("Longitude", fontsize=fsize)
        plt.ylabel("Latitude", fontsize=fsize)
        plt.tight_layout()
    if mode == "presentation":
        plt.style.use("dark_background")
        fsize = 12
        plt.figure(num=None, figsize=(12, 8), facecolor="w")
        plt.rc("font", family="serif")
        # im = ds.plot.pcolormesh(add_colorbar=False,vmin=5, vmax=5.35)
        im = ds.plot.pcolormesh(add_colorbar=False)
        cb = plt.colorbar(im, orientation="vertical", pad=0.05)
        cb.set_label(label=label, size=fsize)
        plt.title(title, fontsize=14)
        plt.xlabel("", fontsize=fsize)
        plt.ylabel("", fontsize=fsize)
        plt.tight_layout()

    plt.savefig(outname)
    plt.close()


def lst_luc_plot(dluc, dvar, class_names, outname, mode, label, title, fsize,
                 multiband, legend):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    import pandas as pd
    import xarray as xr

    # dluc numpy array is delta LUC fraction changes between two years.
    # dvar is a numpy array and change in the y variable (lst, albedo, etc)
    if mode == "paper":
        import matplotlib as mpl

        mpl.rcParams.update(mpl.rcParamsDefault)
        jet = plt.get_cmap("tab10")
        colors = iter(jet(np.linspace(0, 1, 6)))
        fsize = fsize
        plt.figure(num=None, figsize=(12, 8), facecolor="w")
        plt.rc("font", family="serif")
        est_slope = []
        est_intercept = []
        est_pvalue = []
        est_rvalue = []
        est_std_error = []
        col_names = ["Slope", "Intercept", "P_value", "r_value", "std_err"]
        df = pd.DataFrame(index=class_names, columns=col_names)
        for i in np.arange(0, 6):
            print(f"{class_names[i]}")
            if multiband == True:
                luc = dluc.isel(band=i).values.ravel()
            else:
                luc = dluc.values.ravel()
            luc = reject_outliers(luc)
            lst = dvar.values.ravel()
            lst = reject_outliers(lst)
            idx = np.isfinite(luc) & np.isfinite(lst)
            z = np.polyfit(luc[idx], lst[idx], 1)
            p = np.poly1d(z)
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                luc[idx], lst[idx])
            plt.scatter(luc, lst, color="lightgray")
            if multiband == True:
                plt.plot(luc,
                         p(luc),
                         color=next(colors),
                         label=class_names[i],
                         linewidth=2)
            else:
                plt.plot(luc,
                         p(luc),
                         color="r",
                         label=class_names[i],
                         linewidth=4)
            est_slope.append(slope)
            est_intercept.append(intercept)
            est_pvalue.append(p_value)
            est_rvalue.append(r_value)
            est_std_error.append(std_err)
        plt.xlabel("Fractional change in land cover (2003-2013)",
                   fontsize=fsize)
        plt.ylabel(label, fontsize=fsize)
        plt.title(title)
        # plt.xlim(0, 100)
        plt.tight_layout()
        if legend == True:
            plt.legend(loc="center left",
                       bbox_to_anchor=(1.0, 0.5),
                       fontsize=fsize)
        plt.subplots_adjust(right=0.82)
        plt.savefig(outname)
        plt.close()

    if mode == "presentation":
        plt.style.use("dark_background")
        jet = plt.get_cmap("tab10")
        colors = iter(jet(np.linspace(0, 1, 6)))
        fsize = fsize
        plt.figure(num=None, figsize=(24, 20), facecolor="w")
        plt.rc("font", family="serif")
        est_slope = []
        est_intercept = []
        est_pvalue = []
        est_rvalue = []
        est_std_error = []
        col_names = ["Slope", "Intercept", "P_value", "r_value", "std_err"]
        df = pd.DataFrame(index=class_names, columns=col_names)
        for i in np.arange(0, 6):
            print(f"{class_names[i]}")
            if multiband == True:
                dvar = dvar.where(xr.ufuncs.isfinite(dluc.isel(band=i)))
                luc = dluc.isel(band=i).values.ravel()

            else:
                dvar = dvar.where(xr.ufuncs.isfinite(dluc))
                luc = dluc.values.ravel()

            lst = dvar.values.ravel()

            luc = reject_outliers(luc)
            lst = reject_outliers(lst)

            idx = np.isfinite(luc) & np.isfinite(lst)
            z = np.polyfit(luc[idx], lst[idx], 1)
            p = np.poly1d(z)
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                luc[idx], lst[idx])
            plt.scatter(luc, lst, color="lightgray")
            if multiband == True:
                plt.plot(luc,
                         p(luc),
                         color=next(colors),
                         label=class_names[i],
                         linewidth=2)
            else:
                plt.plot(luc,
                         p(luc),
                         color="r",
                         label=class_names[i],
                         linewidth=4)
            est_slope.append(slope)
            est_intercept.append(intercept)
            est_pvalue.append(p_value)
            est_rvalue.append(r_value)
            est_std_error.append(std_err)
        plt.rc("xtick", labelsize=16)
        plt.rc("ytick", labelsize=16)
        plt.xlabel("Fractional change in land cover (2003-2013)",
                   fontsize=fsize)
        plt.ylabel(label, fontsize=fsize)
        plt.title(title, fontsize=fsize)
        # plt.xlim(0, 100)
        if legend == True:
            plt.legend(loc="center left",
                       bbox_to_anchor=(1.0, 0.5),
                       fontsize=fsize)

        # plt.subplots_adjust(right=0.5)

        plt.tight_layout()
        plt.savefig(outname)
        plt.close()

    df["Slope"] = est_slope
    df["Intercept"] = est_intercept
    df["P_value"] = est_pvalue
    df["r_value"] = est_rvalue
    df["std_err"] = est_std_error
    return df


def argmax(x):
    import numpy as np
    if np.isnan(x).all():
        return np.nan
    else:
        return max(range(len(x)), key=lambda y: x[y])


def class_to_what(x, CType, idx):
    import numpy as np
    if CType == 'EF':
        n = []
        exceptions = 0  #Excluding EF from the conversion type
        if (x[exceptions] < -50):
            max1 = np.nanmax(np.delete(x, exceptions))
            #max1 = x[a1[a2]]
            if (abs(x[idx - 1]) - abs(max1)) < 10:
                I = np.where(x == max1)
                if I[0] == 1:
                    n = 2  # EF to DF
                elif I[0] == 2:
                    n = 3  # EF to shrub
                elif I[0] == 3:
                    n = 4  # EF to Herbacuous
                elif I[0] == 4:
                    n = 5  # EF to Sparse/Barren
                elif I[0] == 5:
                    n = 6  # EF to wetlands
                elif I[0] == 6:
                    n = 7  # EF to water
            else:
                n = np.nan
        else:
            n = np.nan
        return n

    elif CType == 'DF':
        n = []
        exceptions = 1  #Index of the class to be studied for change
        if (x[exceptions] < -50):
            max1 = np.nanmax(np.delete(x, exceptions))
            #max1 = x[a1[a2]]
            if (abs(x[idx - 1]) - abs(max1)) < 10:
                I = np.where(x == max1)
                if I[0] == 0:
                    n = 1  # DF to EF
                elif I[0] == 2:
                    n = 3  # DF to shrub
                elif I[0] == 3:
                    n = 4  # DF to Herbasuous
                elif I[0] == 4:
                    n = 5  # DF to Sparse/Barren
                elif I[0] == 5:
                    n = 6  # DF to wetlands
                elif I[0] == 6:
                    n = 7  # EF to water
            else:
                n = np.nan
        else:
            n = np.nan
        return n
    elif CType == 'shrub':
        n = []
        exceptions = 2  #Index of the class to be studied for change
        if (x[exceptions] < -50):
            max1 = np.nanmax(np.delete(x, exceptions))
            #max1 = x[a1[a2]]
            if (abs(x[idx - 1]) - abs(max1)) < 10:
                I = np.where(x == max1)
                if I[0] == 0:
                    n = 1  # DF to EF
                elif I[0] == 1:
                    n = 2  # DF to shrub
                elif I[0] == 3:
                    n = 4  # DF to Herbasuous
                elif I[0] == 4:
                    n = 5  # DF to Sparse/Barren
                elif I[0] == 5:
                    n = 6  # DF to wetlands
                elif I[0] == 6:
                    n = 7  # EF to water
            else:
                n = np.nan
        else:
            n = np.nan
        return n
    elif CType == 'herb':
        n = []
        exceptions = 3  #Index of the class to be studied for change
        if (x[exceptions] < -50):
            max1 = np.nanmax(np.delete(x, exceptions))
            #max1 = x[a1[a2]]
            if (abs(x[idx - 1]) - abs(max1)) < 10:
                I = np.where(x == max1)
                if I[0] == 0:
                    n = 1  # DF to EF
                elif I[0] == 1:
                    n = 2  # DF to shrub
                elif I[0] == 2:
                    n = 3  # DF to Herbasuous
                elif I[0] == 4:
                    n = 5  # DF to Sparse/Barren
                elif I[0] == 5:
                    n = 6  # DF to wetlands
                elif I[0] == 6:
                    n = 7  # EF to water
            else:
                n = np.nan
        else:
            n = np.nan
        return n
    elif CType == 'sparse':
        n = []
        exceptions = 4  #Index of the class to be studied for change
        if (x[exceptions] < -50):
            max1 = np.nanmax(np.delete(x, exceptions))
            #max1 = x[a1[a2]]
            if (abs(x[idx - 1]) - abs(max1)) < 10:
                I = np.where(x == max1)
                if I[0] == 0:
                    n = 1  # DF to EF
                elif I[0] == 1:
                    n = 2  # DF to shrub
                elif I[0] == 2:
                    n = 3  # DF to Herbasuous
                elif I[0] == 3:
                    n = 4  # DF to Sparse/Barren
                elif I[0] == 5:
                    n = 6  # DF to wetlands
                elif I[0] == 6:
                    n = 7  # EF to water
            else:
                n = np.nan
        else:
            n = np.nan
        return n
    elif CType == 'wetland':
        n = []
        exceptions = 5  #Index of the class to be studied for change
        if (x[exceptions] < -50):
            max1 = np.nanmax(np.delete(x, exceptions))
            #max1 = x[a1[a2]]
            if (abs(x[idx - 1]) - abs(max1)) < 10:
                I = np.where(x == max1)
                if I[0] == 0:
                    n = 1  # DF to EF
                elif I[0] == 1:
                    n = 2  # DF to shrub
                elif I[0] == 2:
                    n = 3  # DF to Herbasuous
                elif I[0] == 3:
                    n = 4  # DF to Sparse/Barren
                elif I[0] == 4:
                    n = 5  # DF to wetlands
                elif I[0] == 6:
                    n = 7  # EF to water
            else:
                n = np.nan
        else:
            n = np.nan
        return n
    elif CType == 'water':
        n = []
        exceptions = 6  #Index of the class to be studied for change
        if (x[exceptions] < -50):
            max1 = np.nanmax(np.delete(x, exceptions))
            #max1 = x[a1[a2]]
            if (abs(x[idx - 1]) - abs(max1)) < 10:
                I = np.where(x == max1)
                if I[0] == 0:
                    n = 1  # DF to EF
                elif I[0] == 1:
                    n = 2  # DF to shrub
                elif I[0] == 2:
                    n = 3  # DF to Herbasuous
                elif I[0] == 3:
                    n = 4  # DF to Sparse/Barren
                elif I[0] == 4:
                    n = 5  # DF to wetlands
                elif I[0] == 5:
                    n = 6  # DF to wetlands
            else:
                n = np.nan
        else:
            n = np.nan
        return n
    else:
        print('No class provided')


def get_raster_domain(in_ras, out_name):
    # Get the bounding box of a tif and create a shapefile polygon.
    # In_ ras = input raster (str)
    # out_name = desired path and name of the output file (str)
    import rasterio
    import geopandas as gpd
    from shapely.geometry import box
    ds = rasterio.open(in_ras)
    bounds = ds.bounds
    crs = ds.crs
    gdf = gpd.GeoDataFrame(columns=['geometry'])
    gdf = gdf.append(
        {'geometry': box(bounds[0], bounds[1], bounds[2], bounds[3])},
        ignore_index=True)
    gdf.crs = crs
    gdf.to_file(out_name)
