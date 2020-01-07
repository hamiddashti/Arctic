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



def viirs_wget_generator(product,folder,tiles,dates,out_dir):
    
    import pandas as pd
    import numpy as np
    from datetime import datetime
    import os
   
    def return_url(url):
        import time
        try:
            import urllib.request as urllib2
        except ImportError:
            import urllib2
        import logging
    
        LOG = logging.getLogger( __name__ )
        the_day_today = time.asctime().split()[0]
        the_hour_now = int(time.asctime().split()[3].split(":")[0])
        if the_day_today == "Wed" and 14 <= the_hour_now <= 17:
            LOG.info("Sleeping for %d hours... Yawn!" % (18 - the_hour_now))
            time.sleep(60 * 60 * (18 - the_hour_now))

        req = urllib2.Request("%s" % (url), None)
        html = urllib2.urlopen(req).readlines()
        return html
    
    #wget --user=hamiddashti --password=Iran1140 -p /tmp -r -nd --no-parent -A "*h11v06.006*.h5" http://e4ftl01.cr.usgs.gov/MOTA/MCD15A3H.006/2002.07.04/
    url = 'https://e4ftl01.cr.usgs.gov/'
    
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
    url_tmp = url+folder+"/"+product+"/"
    html = return_url(url_tmp)
        
    #dates = ["2002.07.23", "2002.08.05"]
    start = datetime.strptime(dates[0],'%Y.%m.%d')
    end = datetime.strptime(dates[1],'%Y.%m.%d')
    modis_date = []
    for line in html:
        if line.decode().find("href") >=0 and \
                        line.decode().find("[DIR]") >= 0:
                            
                            the_date = line.decode().split('href="')[1].split('"')[0].strip("/") 
                            tmp_date = datetime.strptime(the_date,'%Y.%m.%d')
                            if (start <= tmp_date <= end):
                                modis_date.append(tmp_date)
           
    df = pd.DataFrame(dict(dates_available=modis_date))
    df['day_of_year'] = df['dates_available'].dt.dayofyear
    df['dates_available'] = df['dates_available'].dt.strftime('%Y.%m.%d')
    df['year'] = pd.to_datetime(df['dates_available'])
    df['year'] = df['year'].dt.year
    
    year1 = df['year'][0]
    year2 = df['year'][len(df)-1]
    years = np.arange(year1,year2+1,1)
    
    
    for k in np.arange(len(years)):
        tmp_dir = str(years[k])
        new_dir = out_dir+tmp_dir
        os.mkdir(new_dir)
    
    f_path = []
    for n in np.arange(len(df)):
        f_tmp = out_dir + str(df['year'][n])
        f_path.append(f_tmp)  
    
    
    name = []
    for i in np.arange(len(tiles)):
        for j in np.arange(len(modis_date)):
            tmp1 = 'wget --user=hamiddashti --password=Iran1140 -P ' +f_path[j]+ ' -r -nd --no-parent -A'
            tmp2 = ' "*' + tiles[i]+'*.h5" ' + url+folder+'/'+product+'/'+str(df['dates_available'][j])+'/ -q'    
            name_tmp = tmp1+tmp2
            name.append(name_tmp)
            
    
    total_line=len(name)
    line10th = np.ceil(len(name)/100)
    progress = np.arange(0,total_line,line10th)
    
    
    
    acc = 0
    x=1
    for k in np.arange(len(progress)):
        if (acc==0):
            name.insert(0,'start_time="$(date -u +%s)"'+ '; echo Downloading the ' + product +" started\n\n")
            acc +=1
        else:
            ins_char = 'echo ' + str(x)+' percent of requested data is downloaded'
            some_info = ins_char+'; end_time="$(date -u +%s)"'+'; elapsed="$(($end_time-$start_time))"'+'; echo "Total of $elapsed seconds elapsed for downloading"'
            name.insert(int(progress[k]+acc),some_info)
            acc +=1
            x +=1
        
    name.insert(len(name)+1 , 'echo ' + 'Downloading finished.Data stored in: ' + out_dir)
    
    outname = out_dir+'/Download_script.sh'
    
    with open(outname, 'w') as filehandle:
        for listitem in name:
            filehandle.write('%s\n' % listitem)




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

def lai_filter(filepath,out,std):

    import os
    import xarray as xr
    #import collections
    #filepath = 'P:\\nasa_above\\working\\modis_analyses\\MyTest.nc'
    fname = os.path.basename(filepath)
    outname_lai = filepath.replace(fname,fname.replace('.nc','') + '_lai_filtered.nc')
    
    # Read the nc file as xarray DataSet
    ds = xr.open_dataset(filepath)
    
    # Read the layers from the dataset (DataArray)
    FparLai_QC = ds['FparLai_QC']
    lai = ds['Lai_500m']
    
    
    # ------------------ Filtering -------------------------------------------
    print('\n\nStarted filtering LAI: takes some time')
    
    # These values come from the LAI manual for quality control
    # returns the quality value (0,2,32 and 34) and nan for every other quality
    # https://lpdaac.usgs.gov/documents/2/mod15_user_guide.pdf
    lai_flag= FparLai_QC.where((FparLai_QC.values==0) | (FparLai_QC.values==2)
    | (FparLai_QC.values==32) | (FparLai_QC.values==34) )
    
    #Convert it Boolean 
    lai_flag = (lai_flag >= 0) 
    #Convert Boolean to zero (bad quality) and one (high quality)
    lai_flag = lai_flag.astype(int)
    lai_tmp = lai_flag*lai
    #replace all zeros with nan values
    lai_final = lai_tmp.where(lai_tmp.values != 0)
    lai_final = lai_final.rename('LAI')
    
    # Did use asked for standard deviation of lai too?
    if std == True:
        lai_std = ds['LaiStdDev_500m']
        print('\n\nStarted filtering LAI Standard Deviation: takes some time')
        lai_std_tmp = lai_flag*lai_std
        lai_std_final = lai_std_tmp.where(lai_std_tmp.values != 0)
        lai_std_final = lai_std_final.rename('LAI_STD')
        lai_dataset = xr.merge([lai_final,lai_std_final])
        outname_std_lai = filepath.replace(fname,fname.replace('.nc','') + '_lai_dataset.nc')
    
    # -------------------- OUTPUTS ------------------------------------------
    print('\n\n Figuring out the outputs:')
    if std == True and out == True:
        print('\n   ---writing the lai and lai_std as a "dataset" on the disk')
        lai_dataset.to_netcdf(outname_std_lai)
        return lai_dataset
    elif std == False and out == True:
        print('\n   ---wirintg just lai to the disk (no STD) and return lai' )
        lai_final.to_netcdf(outname_lai)
        return lai_final
    elif std == True and out == False:
        print('\n   ---return lai and lai_std as a "dataset"')
        return lai_dataset
    elif std == False and out == False:
        print('\n   ---return lai (not standard deviation) and no writing on the disk')
        return lai_final


#######################End of lai_filter function ###################################
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



