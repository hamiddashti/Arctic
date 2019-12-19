''' 

Created May 17, 2013

@ bill.smith

Downloads data from LPDACC

'''

import numpy as np

import os

###wget###

#wget http://e4ftl01.cr.usgs.gov/MOLA/MYD09A1.005/2002.07.04/MYD09A1.A2002185.h23v05.005.2007172113836.hdf



P = ['MCD43A4']

C = '006'

T=['h12v01','h13v01','h14v01','h15v01',

'h10v02','h11v02','h12v02','h13v02','h14v02','h15v02',

'h09v03','h10v03','h11v03','h12v03','h13v03','h14v03',

'h08v04','h09v04','h10v04','h11v04','h12v04','h13v04','h14v04']

D = ['01.01','01.17','02.02','02.18','03.05','03.06','03.21','03.22',

	'04.06','04.07','04.22','04.23','05.08','05.09','05.24','05.25',

	'06.09','06.10','06.25','06.26','07.11','07.12','07.27','07.28',

	'08.12','08.13','08.28','08.29','09.13','09.14','09.29','09.30',

	'10.15','10.16','10.31','11.01','11.16','11.17','12.02','12.03',

	'12.18','12.19']   

Y = range(2018,2020)

DownloadDir = 'https://e4ftl01.cr.usgs.gov/MOTA/MCD43A4.006/'

OUT = 'P:\\nasa_above\\working\\modis_analyses'

#########IF RAW TILES NEED TO BE DOWNLOADED#######################

f = open('/Users/Bill/Desktop/'+'DAAC2DISK_ABOVE_'+P[0]+'_16day.bat','w')

for i in np.arange(len(T)):

	for m in np.arange(len(Y)):

		for n in np.arange(len(D)):

			cmd = "".join(["wget --user wkolby2 --password Ice-Nine9 -nd --no-parent -P ",OUT,str(Y[m])," -r -A '*",T[i],"*.hdf'  https://e4ftl01.cr.usgs.gov/MOTA/",P[0],".",C,"/",str(Y[m]),".",D[n],"/",'\n'])

			f.write(cmd)

f.close()

os.system('chmod -R 777 '+'/Users/Bill/Desktop/'+'DAAC2DISK_ABOVE_'+P[0]+'_16day.bat')