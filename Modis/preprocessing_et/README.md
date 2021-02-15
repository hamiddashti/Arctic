# Preprocessing ET 
## Download ET products
* The original dataset is called "PML_V2: Coupled Evapotranspiration and Gross Primary Product"
* Download link:  
  [PML_V2: Coupled Evapotranspiration and Gross Primary Product](https://developers.google.com/earth-engine/datasets/catalog/CAS_IGSNRR_PML_V2) 
  * Dataset is 500m and 8-days resolutions and it includes:  
    * Ec ------------> Vegetation transpiration.
    * Es ------------> Soil evaporation.
    * Ei ------------> Interception from vegetation canopy.
    * ET_water ------> Water body, snow and ice evaporation. Penman evapotranspiration is regarded as actual evaporation for them.
## Processing scripts
* preprocessing_ET.py ---------> Resampling to monthly, seasonal and annual. 
* Final data are located in 
  * /data/ABOVE/MODIS/ET
  * /xdisk/davidjpmoore/hamiddashti/nasa_above_data/Final_data/ET_Final
