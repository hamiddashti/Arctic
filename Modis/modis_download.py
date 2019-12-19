# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 14:49:29 2019

@author: hamiddashti
"""

import pandas as pd
import numpy as np

#https://e4ftl01.cr.usgs.gov/MOTA/MCD15A3H.006/2002.07.04/MCD15A3H.A2002185.h11v06.006.2015149102740.hdf

#wget --user=hamiddashti --password=Iran1140 -p /tmp -r -nd --no-parent -A "*h11v06.006*.hdf" http://e4ftl01.cr.usgs.gov/MOTA/MCD15A3H.006/2002.07.04/


url = 'https://e4ftl01.cr.usgs.gov/'
folder = 'MOTA'

product = 'MCD15A3H.006'
out = '/run/user/1008/gvfs/smb-share:server=gaea,share=projects,user=hamiddashti/nasa_above/working/modis_analyses/my_data'


T=['h12v01','h13v01','h14v01','h15v01','h16v01',

'h10v02','h11v02','h12v02','h13v02','h14v02','h15v02',

'h09v03','h10v03','h11v03','h12v03','h13v03','h14v03',

'h08v04','h09v04','h10v04','h11v04','h12v04','h13v04','h14v04']


#import pymodis
#modis_download.py -U hamiddashti -P Iran1140 -t h03v10 -s MOTA -p MCD15A3H.006 -f 2002-07-04 -O ./tmp


#T=['h12v01']

dates=pd.date_range(start='7/28/2002', end='7/29/2002',freq = '4D')
df = pd.DataFrame(dict(date_given=dates))
df['day_of_year'] = df['date_given'].dt.dayofyear
df['date_given'] = df['date_given'].dt.strftime('%Y.%m.%d')
df['year'] = pd.to_datetime(df['date_given'])
df['year'] = df['year'].dt.year

c = '006'

tmp1 = 'wget --user=hamiddashti --password=Iran1140 -p ' +out+ ' -r -nd --no-parent -A'
#tmp2 = ' "*' + T[0]+'*.hdf" ' + url+folder+'/'+product+'/'+str(df['date_given'][0])+'/'
#tmp = tmp1+tmp2


name = []
for i in np.arange(len(T)):
    for j in np.arange(len(dates)):
        tmp2 = ' "*' + T[i]+'*.hdf" ' + url+folder+'/'+product+'/'+str(df['date_given'][j])+'/'    
        name_tmp = tmp1+tmp2
        name.append(name_tmp)
        

outname = out+'/above.sh'

with open(outname, 'w') as filehandle:
    for listitem in name:
        filehandle.write('%s\n' % listitem)






























    
    
