# -*- coding: utf-8 -*-
def getTile(lat, lon):

    """
    Created on Thu Nov 14 08:23:48 2019

    @author: hamiddashti

    This script is for mass downloading of the modis: 
    
        - Get the tile number based on the coordinates 
   
    """
    import numpy as np
    """
    Modis provides a text file containing the range of latitude and longitude coordinates 
    for each tile. We will load this data into a numpy two dimensional array. 
    Next, we will define the coordinates of the point we would like to convert.
    # https://modis-land.gsfc.nasa.gov/pdf/sn_bound_10deg.txt
    """
    data = np.genfromtxt('sn_bound_10deg.txt',skip_header = 7,
                     skip_footer = 3)
    #lat = lat
    #lon = lon

    # Loop through the text file to see if our coordinate is in that range
    in_tile = False
    i = 0
    while(not in_tile):
        in_tile =  lat>= data[i, 4] and lat <= data[i, 5] and lon >= data[i, 2] and lon <= data[i, 3]
        i += 1
    
    vert = "vertical is = "+ str(data[i-1, 0])
    horiz = "horizontal is = " + str(data[i-1, 1])
    
    return vert, horiz;


#---------------------------------- Example 
#print (modis_tile(42,134))
