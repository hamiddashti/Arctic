
import numpy as np
import datetime as dt
#import os
import gdal
import netCDF4

import glob 



#products_list = ['Fpar_500m','Lai_500m']
products_list = ['Lai_500m']
tiles_list = ['h10v02','h14v01']
in_dir = 'P:\\nasa_above\\working\\modis_analyses\\tmp\\Tifs\\'

year1 = 2002
year2 = 2004
years_list = np.arange(year1,year2)
    
basedate = dt.datetime(1858,11,17,0,0,0)



product=products_list[0]
tile='h11v02'
year = 2002


for product in products_list:

    
    for tile in tiles_list:
      

        init_path=in_dir+str(year1)+'\\*'+product+'*'
        files_init=[]
        files_init.append([file_init for file_init in glob.glob(init_path) if tile in file_init])
        if len(files_init[0])==0:
             break
        
        ds = gdal.Open(files_init[0][0])
        a = ds.ReadAsArray()
        nlat,nlon = np.shape(a)
        
        b = ds.GetGeoTransform() #bbox, interval
        lon = np.arange(nlon)*b[1]+b[0]
        lat = np.arange(nlat)*b[5]+b[3]
              
        # create NetCDF file
        out_nc = in_dir+'Netcdf\\'+product+'_'+tile+'.nc'
        nco = netCDF4.Dataset(out_nc,'w',clobber=True)
        
        # chunking is optional, but can improve access a lot: 
        # (see: http://www.unidata.ucar.edu/blogs/developer/entry/chunking_data_choosing_shapes)
        chunk_lat=ds.RasterYSize
        chunk_lon=ds.RasterXSize
        chunk_time=1
        
        # create dimensions, variables and attributes:
        nco.createDimension('lon',nlon)
        nco.createDimension('lat',nlat)
        nco.createDimension('time',None)
        timeo = nco.createVariable('time','f4',('time'))
        timeo.units = 'days since 1858-11-17 00:00:00'
        timeo.standard_name = 'time'
        
        lono = nco.createVariable('lon','f4',('lon'))
        lono.units = 'degrees_east'
        lono.standard_name = 'longitude'
        
        lato = nco.createVariable('lat','f4',('lat'))
        lato.units = 'degrees_north'
        lato.standard_name = 'latitude'
        
        # create container variable for CRS: lon/lat WGS84 datum
        crso = nco.createVariable('crs','i4')
        crso.long_name = 'Lon/Lat Coords in WGS84'
        crso.grid_mapping_name='latitude_longitude'
        crso.longitude_of_prime_meridian = 0.0
        crso.semi_major_axis = 6378137.0
        crso.inverse_flattening = 298.257223563
        
        # create short integer variable for temperature data, with chunking
        tmno = nco.createVariable(product, 'i2',  ('time', 'lat', 'lon'), 
           zlib=True,chunksizes=[chunk_time,chunk_lat,chunk_lon],fill_value=255)
        tmno.units = 'degC'
        tmno.scale_factor = 1
        tmno.add_offset = 0.00
        tmno.long_name = 'minimum monthly temperature'
        tmno.standard_name = 'air_temperature'
        tmno.grid_mapping = 'crs'
        tmno.set_auto_maskandscale(False)
        
        nco.Conventions='CF-1.6'
        
        #write lon,lat
        lono[:]=lon
        lato[:]=lat
        
        
        itime=0
                   
        
        for year in years_list:
            
            mypath=in_dir+str(year)+'\\*'+product+'*'
            files=[]
            files.append([files for files in glob.glob(mypath) if tile in files])
            files=[item for elem in files for item in elem]
            
            for f in files:
                #year=int(f[61:65])
                doy=int(f[65:68])
                date=dt.datetime(year,1,1)+dt.timedelta(doy-1)
                #print(date)
                
                dtime=(date-basedate).total_seconds()/86400.
                timeo[itime]=dtime
                #min temp
                #tmn_path = os.path.join(root,f)
                #print(f)
                tmn=gdal.Open(f)
                a=tmn.ReadAsArray()  #data
                tmno[itime,:,:]=a
                itime=itime+1
                print(date)
                
    nco.close()
    
