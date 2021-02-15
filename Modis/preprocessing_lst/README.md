# Preparing LST Data

Downlaod Day and Night LST Data  
-----------

* The LST data were downloaded from https://lpdaacsvc.cr.usgs.gov/appeears/ in ready to use nc format and geographic CRS.  
* Data are saved in '/xdisk/davidjpmoore/hamiddashti/nasa_above_data/Final_data/LST and '/data/ABOVE/MODIS/MYD21A2'.

Processing
---------
* **preprocessing_LST.py** ----> Filter LST based on the quality flag, take the seasonal, monthly and annual mean. 
* The resulted files are in /xdisk/davidjpmoore/hamiddashti/data/LST_Final/LST/ and /data/ABOVE/MODIS/MYD21A2/LST_Final/LST.  
**Note all the other products (e.g. Albedo, ET) matched with these LST data.**  
