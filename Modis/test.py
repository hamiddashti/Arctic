# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:08:04 2019

@author: hamiddashti
"""
"""
from handy_funs import getTile 

print ("UpperEast tile" + str(getTile(66.5078812,174.7286377)))
print ("UpperWest tile" + str(getTile(73.0940475,-62.1090126)))
print ("LowerEast tile" + str(getTile(48.6725121,-94.2617569)))
print ("LowerWest tile" + str(getTile(45.3240967,-149.5862427)))
"""
# ---------------Import libraries ---------------------------------------------
# Cleaning console and variable explorer in Spyder
import os
clear = lambda: os.system('cls')

import requests as r
import getpass
import time
import cgi
import json
import pandas as pd
import geopandas as gpd
import xarray
import numpy as np
import hvplot.xarray
import holoviews as hv
import pprint

# Set input directory, change working directory
inDir = os.getcwd() + os.sep  # Set input directory to the current working directory
os.chdir(inDir)               # Change to working directory
###############################################################################



#--------------------------Login to AppEEARS using the API --------------------

# Set the AρρEEARS API to a variable 
api = 'https://lpdaacsvc.cr.usgs.gov/appeears/api/'  
                                                     


# Insert API URL, call login service, provide credentials & return json
login_response = r.post(f"{api}/login", auth=('hamiddashti', 'Iran1140')).json()
login_response

###############################################################################



# ---------- Get a sense of what data is available on AppEEARS ----------------
# request all products in the product service
product_response = r.get('{}product'.format(api)).json()
                       
# Print no. products available in AppEEARS  
print('AρρEEARS currently supports {} products.'.format(len(product_response)))  


# Create a dictionary indexed by product name & version
products = {p['ProductAndVersion']: p for p in product_response} 

# get info for a product(e.g. LAI)
products['MCD15A3H.006']  

# Make list of all products (including version)
prodNames = {p['ProductAndVersion'] for p in product_response} 

 # Make for loop to search list of products 'Description' for a keyword (e.g. Temperature) 
for p in prodNames:                                                          
    if 'Temperature' in products[p]['Description']:
        pprint.pprint(products[p])                             
        
        
prods = ['MCD15A3H.006']     # Start a list for products to be requested, beginning with MCD15A3H.006
#prods.append('MOD11A2.006')  # Append the MOD11A2.006 8 day LST product to the list of products desired
#prods.append('SRTMGL1.003')  # Append the SRTMGL1.003 product to the list of products desired
prods

# Request layers for the 2nd product (index 1) in the list: MOD11A2.006
lai_response = r.get('{}product/{}'.format(api, prods[0])).json()  
list(lai_response.keys())

lai_response['Lai_500m'] # Print layer response

 # Create tupled list linking desired product with desired layers
layers = [(prods[0],'Lai_500m'),(prods[0],'FparLai_QC')] 

#lai_response = r.get('{}product/{}'.format(api, prods[0])).json()  # Request layers for the 1st product (index 0) in the list: MCD15A3H.006
#list(lai_response.keys())   
#lai_response['Lai_500m']['Description']  # Make sure the correct layer is requested

#layers.append((prods[0],'Lai_500m')) # Append to tupled list linking desired product with desired layers
#dem_response = r.get('{}product/{}'.format(api, prods[2])).json()  # Request layers for the 3rd product (index 2) in the list: SRTMGL1.003
#list(dem_response.keys())  
#layers.append((prods[2],'Band1')) # Append to tupled list linking desired product with desired layers

prodLayer = []
for l in layers:
    prodLayer.append({
            "layer": l[1],
            "product": l[0]
          })
prodLayer

token = login_response['token']             # Save login token to a variable
# Create a header to store token information, needed to submit a request
head = {'Authorization': 'Bearer {}'.format(token)}  

##############################################################################


# ----------------------- Load the shapefile of the study area ---------------

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#  NOTE!! the shapefile should be in geographic projection systm

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# Read in shapefile as dataframe using geopandas
nps = gpd.read_file('P:\\nasa_above\Study_area\\Above2.shp') 
print(nps.head())                       # Print first few lines of dataframe

# Extract a feature from shp file if there are multiple features. 
# the AppEEARS currently doesn't support multiple features. 

nps_gc = nps[nps['Region']=='Extended Region'].to_json() 
nps_gc = json.loads(nps_gc) 


# Call to spatial API, return projs as json
projections = r.get('{}spatial/proj'.format(api)).json()  
projections                       # Print projections and information

projs = {}                                  # Create an empty dictionary
for p in projections: projs[p['Name']] = p  # Fill dictionary with `Name` as keys
list(projs.keys())                          # Print dictionary keys

projs['geographic']
###############################################################################


#-----------------Crreate a task and submit it for data download ---
# User-defined name of the task: 'NPS Vegetation Area' used in example
task_name = 'Modis download test run' 

task_type = ['point','area']        # Type of task, area or point
proj = projs['geographic']['Name'] 
#proj = "albers_ard_conus"   # Set output projection 
outFormat = ['geotiff', 'netcdf4']  # Set output file format type
startDate = '07-01-2017'            # Start of the date range for which to extract data: MM-DD-YYYY
endDate = '07-31-2017'              # End of the date range for which to extract data: MM-DD-YYYY
recurring = False                   # Specify True for a recurring date range
#yearRange = [2000,2016]        # if recurring = True, set yearRange, change start/end date to MM-DD


task = {
    'task_type': task_type[1],
    'task_name': task_name,
    'params': {
         'dates': [
         {
             'startDate': startDate,
             'endDate': endDate
         }],
         'layers': prodLayer,
         'output': {
                 'format': {
                         'type': outFormat[1]}, 
                         'projection': proj},
         'geo': nps_gc,
    }
}


# Post json to the API task service, return response as json
task_response = r.post('{}task'.format(api), json=task, headers=head).json()  
task_response 


###############################################################################

# -----------------Check the status of the request ----------------------------

# Limit API response to most recent entries, return as pretty json
params = {'limit': 1, 'pretty': True} 

# Query task service, setting params and header 
tasks_response = r.get('{}task'.format(api), params=params, headers=head).json() 
tasks_response 

###############################################################################










