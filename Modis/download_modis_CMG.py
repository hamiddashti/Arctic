

import pandas as pd
import numpy as np
from datetime import datetime
#import os

product = "MYD11C3.006"
folder = "MOLA"
dates = ["2000.01.01","2014.12.31"]
out_dir = '/data/ABOVE/MODIS/MYD11C3/'

def return_url(url):
	#import time
	try:
		import urllib.request as urllib2
	except ImportError:
		import urllib2
	#import logging

	#LOG = logging.getLogger( __name__ )
	#the_day_today = time.asctime().split()[0]
	#the_hour_now = int(time.asctime().split()[3].split(":")[0])
	#if the_day_today == "Wed" and 14 <= the_hour_now <= 17:
	#    LOG.info("Sleeping for %d hours... Yawn!" % (18 - the_hour_now))
	#    time.sleep(60 * 60 * (18 - the_hour_now))

	req = urllib2.Request("%s" % (url), None)
	html = urllib2.urlopen(req).readlines()
	return html

#wget --user=hamiddashti --password=Iran1140 -p /tmp -r -nd --no-parent -A "*h11v06.006*.hdf" http://e4ftl01.cr.usgs.gov/MOTA/MCD15A3H.006/2002.07.04/
url = 'https://e4ftl01.cr.usgs.gov/'

"""
dates=pd.date_range(start=start_date, end=end_date,freq = '4D')
df = pd.DataFrame(dict(date_given=dates))
df['day_of_year'] = df['date_given'].dt.dayofyear
df['date_given'] = df['date_given'].dt.strftime('%Y.%m.%d')
df['year'] = pd.to_datetime(df['date_given'])
df['year'] = df['year'].dt.year

year1 = df['year'][0]
year2 = df['year'][len(df)-1]
years = np.arange(year1,year2+1,1)
"""
url_tmp = url+folder+"/"+product+"/"
html = return_url(url_tmp)	


#dates = ["2002.07.23", "2002.08.05"]
start = datetime.strptime(dates[0],'%Y.%m.%d')
end = datetime.strptime(dates[1],'%Y.%m.%d')
modis_date = []
for line in html:
	if line.decode().find("href") >=0 and \
					line.decode().find("[DIR]") >= 0:
						
						the_date = line.decode().split('href="')[1].split('"')[0].strip("/") 
						tmp_date = datetime.strptime(the_date,'%Y.%m.%d')
						if (start <= tmp_date <= end):
							modis_date.append(tmp_date)
	   
df = pd.DataFrame(dict(dates_available=modis_date))
df['day_of_year'] = df['dates_available'].dt.dayofyear
df['dates_available'] = df['dates_available'].dt.strftime('%Y.%m.%d')
df['year'] = pd.to_datetime(df['dates_available'])
df['year'] = df['year'].dt.year
year1 = df['year'][0]
year2 = df['year'][len(df)-1]
years = np.arange(year1,year2+1,1)


#for k in np.arange(len(years)):
#    tmp_dir = str(years[k])
#    new_dir = out_dir+tmp_dir
#    os.mkdir(new_dir)

f_path = []
for n in np.arange(len(df)):
	f_tmp = out_dir + str(df['year'][n])
	f_path.append(f_tmp)  

name = []
for j in np.arange(len(modis_date)):
	tmp1 = 'wget --user=hamiddashti --password=Iran1140 -P ' +f_path[j]+ ' -r -nd --no-parent -A'
	tmp2 = ' "*' + '*.hdf" ' + url+folder+'/'+product+'/'+str(df['dates_available'][j])+'/ -q'    
	name_tmp = tmp1+tmp2
	name.append(name_tmp)
		

total_line=len(name)
line10th = np.ceil(len(name)/100)
progress = np.arange(0,total_line,line10th)


acc = 0
x=1
for k in np.arange(len(progress)):
	if (acc==0):
		name.insert(0,'start_time="$(date -u +%s)"'+ '; echo Downloading the ' + product +" started\n\n")
		acc +=1
	else:
		ins_char = 'echo ' + str(x)+' percent of requested data is downloaded'
		some_info = ins_char+'; end_time="$(date -u +%s)"'+'; elapsed="$(($end_time-$start_time))"'+'; echo "Total of $elapsed seconds elapsed for downloading"'
		name.insert(int(progress[k]+acc),some_info)
		acc +=1
		x +=1
	
name.insert(len(name)+1 , 'echo ' + 'Downloading finished.Data stored in: ' + out_dir)

outname = out_dir+'/Download_script.sh'

with open(outname, 'w') as filehandle:
	for listitem in name:
		filehandle.write('%s\n' % listitem)