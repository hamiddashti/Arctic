#!/bin/bash

# First run the make_dir.sh to create the folders for Tifs and also a text file called dir_list.txt which is the list of all directories (years)



#Path to the directory where modis hdf file (seperated by year) are stored.

Path_in="/data/ABOVE/MODIS/MCD15A3H_test/"
Path_in_sed="\/data\/ABOVE\/MODIS\/MCD15A3H_test\/" #use \ before any / in the path to avoid error in sed comment

Path_out="/data/ABOVE/MODIS/MCD15A3H_test/Tifs/"
Path_out_sed="\/data\/ABOVE\/MODIS\/MCD15A3H_test\/Tifs\/"


Product_par="MCD15A3H_par.par"  # This is a heg parameter file prepared for MCD15A3H products. in this file in/out paths and coord of corners are empty

cp $Path_in/dir_list.txt .

cat dir_list.txt | while read dir
 
  do

          ls $Path_in$dir/*.hdf > filenames_$dir.txt #Get the file name in each year directory

          cat filenames_$dir.txt | while read line

       		do
	   		#echo "We are converting the $line file"
  		 	
			
			./hegtool -h $line
   
	   		# The two lines below read the UL and LR coordinates from each file using the hegtool outputs saved in HegHdr.hdr
	                UL_coord=$(sed -n 's/GRID_UL_CORNER_LATLON=//p' HegHdr.hdr)
			LR_coord=$(sed -n 's/GRID_LR_CORNER_LATLON=//p' HegHdr.hdr)
   			
			
			cp /data/home/hamiddashti/mnt/nasa_above/Arctic/Modis/$Product_par .
   
			fname_hdf=$(basename $line)
   			fname_tif=${fname_hdf::-4}.tif
   			

   			sed --in-place "s/SPATIAL_SUBSET_UL_CORNER = (/& $UL_coord/" $Product_par    
   			sed --in-place "s/SPATIAL_SUBSET_LR_CORNER = (/& $LR_coord/" $Product_par
   
   			
			year="$dir\/"
			
   			sed -i "s/INPUT_FILENAME =/INPUT_FILENAME = $Path_in_sed$year$fname_hdf/g" $Product_par
			sed -i "s/OUTPUT_FILENAME =/OUTPUT_FILENAME = $Path_out_sed$year$fname_tif/g" $Product_par

   		
			
			./resample -P MCD15A3H_par.par
   
	done

   done
