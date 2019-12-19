# -*- coding: utf-8 -*-
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



