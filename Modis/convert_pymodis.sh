#!/bin/bash

# Analysis showed that there are something wrong with converting modis hdf
# files to Tif format and reprojecting it using HegTool provided by NASA.
# In this script I use the PyModis package for such conversion. It provides the
# same exact results as APPEEARS tool.

Path_in="/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/"

Path_out="/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/Tifs/"


#cp $Path_in/dir_list.txt .


cat dir_list.txt | while read dir

	do
		
		ls $Path_in$dir/*.hdf > filenames_$dir.txt

		cat filenames_$dir.txt | while read line

		
		do 
		
			fname_hdf=$(basename $line)
			fname_tif=${fname_hdf::-4}

			file_out=$Path_out$dir/$fname_tif
			
			modis_convert.py -s "( 1 1 1 1 1 1 )" -g 0.0041666667 -o $file_out -e 4326 $line
			
		
		
		done


	
	done
