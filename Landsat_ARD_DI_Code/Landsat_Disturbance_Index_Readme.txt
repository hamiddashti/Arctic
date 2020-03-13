Landsat ARD Disturbance Index Data Processing Chain

Step1: Untar the downloaded tar files
       
	   Code: LandsatARD_untar.R
	   
	   Note: Place all folders of the bulk-downloaded files into a single directory. Modify directory name, number of bulk orders and output directory accordingly (see comments in the code).
	   

Step2: Generate BRDF-normalized surface refletance and convert the results to plain binary files

      Code:all the R code under /Landsat_NBAR_code/InR/
	  
	  Note: Only need to modify Line 4-6 in Landsat.BRDF.normalize.v1.0.r based on the comments
	  
	  
Step3: Calculate VI and output results together with land surface temperature

      Code: GenerateARD_DataCube.R, Generate_L47ARD_DataCube.c and Generate_L8ARD_DataCube.c
	  
	  Note: Only need to modify Line 4-5 in GenerateARD_DataCube.R based on the comments in the code
	  
	  Usage: 
	  
	        1. Compile Generate_L47ARD_DataCube.c and Generate_L8ARD_DataCube.c into two different executable files from terminal
			
			   cc Generate_L47ARD_DataCube.c -lm -lz -o Generate_L47ARD_DataCube.exe
			   
			   cc Generate_L8ARD_DataCube.c -lm -lz -o Generate_L8ARD_DataCube.exe
			   
			   Repeat the above two commands if you have made any changes to the .c source files
			   
			2. In a terminal, type "Rscript GenerateARD_DataCube.R" and click enter
			   
Step4: Generate reference DI for each year starting 1984

       Code: Generate_Reference_DI.pl and Generate_Reference_DI.c
	  
	   Note: Only need to modify Line 3-7 in Generate_Reference_DI.pl  based on the comments in the code
	  
	   Usage: 
	   
	       1. Change the mode of Generate_Reference_DI.pl to executable by typing in terminal:
		   
		     chmod +x Generate_Reference_DI.pl
		   
		   2. Compile Generate_Reference_DI.c into an executable from terminal:
		   
		     cc Generate_Reference_DI.c -lm -lz -o Generate_Reference_DI.exe
			 
		   3. In a terminal, type "./Generate_Reference_DI.pl" and click enter
		   
Step5: Generate instant DI
       
	   Code: Generate_ARD_LGDI.R and Generate_ARD_LGDI_v2.c
	  
	   Note: Only need to modify Line 4-6 in Generate_ARD_LGDI.R  based on the comments in the code
	  
	   Usage: 
	   
		   1. Compile Generate_ARD_LGDI_v2.c into an executable from terminal:
		   
		     cc Generate_ARD_LGDI_v2.c -lm -lz -o Generate_ARD_LGDI_v2.exe
			 
		   3. In a terminal, type "Rscript Generate_ARD_LGDI.R" and click enter