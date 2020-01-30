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



