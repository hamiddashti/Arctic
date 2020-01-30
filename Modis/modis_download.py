# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 12:22:34 2019
@author: hamiddashti
"""

import sys
sys.path.append('/data/home/hamiddashti/mnt/nasa_above/Arctic/Modis')
import modis_functions

folder = 'MOTA'
product = 'MCD15A3H.006'
out_dir = '/data/ABOVE/MODIS/MCD15A3H/'

#out_dir = '/run/user/1008/gvfs/smb-share:server=gaea,share=projects,user=hamiddashti/nasa_above/working/modis_analyses/my_data/'
"""
tiles=['h12v01','h13v01','h14v01','h15v01','h16v01',
    
    'h10v02','h11v02','h12v02','h13v02','h14v02','h15v02',
    
    'h09v03','h10v03','h11v03','h12v03','h13v03','h14v03',
    
    'h08v04','h09v04','h10v04','h11v04','h12v04','h13v04','h14v04']
"""

tiles=['h09v02']
#dates format= "Y.m.d"
dates = ["2003.01.01","2019.12.31"]


modis_functions.modis_wget_generator(product,folder,tiles,dates,out_dir)
