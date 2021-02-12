# **Preparing albedo data**
Here I explain the codes, steps and locations for processing albedo data

	• Download data
		○ The data are located on the ORNL DAAC: 
		ABoVE: MODIS-Derived Daily Mean Blue Sky Albedo for Northern North America, 2000-2017
		From <https://daac.ornl.gov/ABOVE/guides/Albedo_Boreal_North_America.html> 
		
		○ Original data is on wheat: 
		/data/ABOVE/MODIS/blue_sky_albedo/orders/df99d6e7e26fb666143dbf2de2de4707/Albedo_Boreal_North_America/data_geographic/
		
		○ The processing scripts are on Github and 
		P:\nasa_above\codes\Arctic\Modis

	• Preprocessing Steps
		○ step1_clipping_stacking_albedo.py --->  Clip the original albedo using the  extended ABoVE domain and stack them over time (daily)
		○ step2_preprocessing_Albedo.py ------->  Take the growing seasonal, monthly, and annual means
		○ step3_rematching.py ----------------------> Match the processed albedo (500m) with LST (1000m) 
