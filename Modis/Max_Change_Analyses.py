if __name__ == "__main__":
	import xarray as xr
	import numpy as np
	from importlib import reload
	import pandas as pd
	import modis_functions
	import dask

	in_dir = "/data/ABOVE/Final_data/"
	out_dir = "/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/LUC_max_change/"
	#out_dir = "F:\\working\\LUC\\"

	# -------------  Initialization -------------------------------------
	# total_pixels = 2692 * 8089
	years = pd.date_range(start="2004", end="2015", freq="A").year
	classes = ["EF", "DF", "shrub", "herb", "sparse", "wetland", "water"]
	#da = np.zeros((7, 7, int)
	#conversions = xr.DataArray(conversions, dims=["original", "converted", "year"])
	#conversions.to_netcdf(out_dir+'conversions.nc')
	# -------------------------------------------------------------------

	# This is the final LULC of the region
	lulc = xr.open_dataarray(in_dir+"LULC_2003_2014.nc")

	# Take the difference between years (e.g. t(i+1)-t(i))
	#for k in np.arange(0,len(years)+1):
		#loop over years

	def cal_max(k):
		
		da = np.zeros((7,7),int)
		da = xr.DataArray(da, dims=["original", "converted"])
		
		print(str(years[k]))
		for idx in np.arange(1, len(classes)+1):
		# for idx in np.arange(1, 2):
			# loop over classes
			#print("Conversion of " + classes[idx-1])
			# Looping over classes
			diff = lulc.isel(years=k + 1) - lulc.isel(years=k)

			# We choose the next two years to make sure a converted class is still the majority class in those years

			#lulc.isel(years=k).to_netcdf(out_dir+'LUC2003.nc')
			#lulc.isel(years=k+1).to_netcdf(out_dir+'LUC2004.nc')
			#lulc.isel(years=k + 2).to_netcdf(out_dir+'LUC2005.nc')
			#diff.to_netcdf(out_dir+'diff.nc')

			lulck2 = lulc.isel(years=k + 2)
			#lulck3 = lulc.isel(years=k + 3)
			original_class = diff.sel(band=idx)
			original_class_change = diff.where(original_class < -50)

			#original_class_change.to_netcdf(out_dir+'original_class_change.nc')
			orig_to_other = xr.apply_ufunc(
				#modis_functions.class_to_what,
				modis_functions.class_to_what,
				original_class_change,
				input_core_dims=[["band"]],
				kwargs={"CType": classes[idx-1],"idx":idx},
				vectorize=True,
			)

			#orig_to_other.to_netcdf(out_dir+classes[j-1] + "_to_other_" + str(years[k]) + ".nc")
			lulck2_tmp = lulck2.where(~xr.ufuncs.isnan(orig_to_other))
			#lulck3_tmp = lulck3.where(~xr.ufuncs.isnan(orig_to_other))
			# lulck2_tmp.to_netcdf('lulck2_tmp.nc')
			# lulck3_tmp.to_netcdf('lulck3_tmp.nc')
			lulck2_max = xr.apply_ufunc(
				#modis_functions.argmax, lulck2_tmp, input_core_dims=[["band"]], vectorize=True
				modis_functions.argmax, lulck2_tmp, input_core_dims=[["band"]], vectorize=True
			)

			# lulck3_max = xr.apply_ufunc(
			# 	modis_functions.argmax, lulck3_tmp, input_core_dims=[["band"]], vectorize=True
			# )

			# lulck2_max.to_netcdf(out_dir+'lulck2_max.nc')

			# lulck3_max.to_netcdf('lulck3_max.nc')

			#lulck2_max = lulck2_max.drop("years")
			# lulck3_max = lulck3_max.drop("years")

			#concat = xr.concat([orig_to_other, lulck2_max, lulck3_max], "tmp")
			#concat_std = concat.std(dim="tmp")
			# concat_std = concat_std.where(concat_std==0)
			# concat_std.to_netcdf('concat_std.nc')
			# final_orig_to_other = orig_to_other.where(concat_std == 0)

			final_orig_to_other = orig_to_other.where(orig_to_other==(lulck2_max+1))

			final_orig_to_other.to_netcdf(
				out_dir+"Final_" + classes[idx-1] + "_" + "to_other_" + str(years[k]) + ".nc"
			)

			#da = xr.open_dataarray(out_dir+'conversions.nc')
			da[idx-1, 0] = final_orig_to_other.where(final_orig_to_other == 1).count()
			da[idx-1, 1] = final_orig_to_other.where(final_orig_to_other == 2).count()
			da[idx-1, 2] = final_orig_to_other.where(final_orig_to_other == 3).count()
			da[idx-1, 3] = final_orig_to_other.where(final_orig_to_other == 4).count()
			da[idx-1, 4] = final_orig_to_other.where(final_orig_to_other == 5).count()
			da[idx-1, 5] = final_orig_to_other.where(final_orig_to_other == 6).count()
			da[idx-1, 6] = final_orig_to_other.where(final_orig_to_other == 7).count()
		
		da = xr.DataArray(da, dims=["original", "converted"])
		da.to_netcdf(out_dir+'da_'+str(years[k])+'.nc')
		# print('year:'+str(years[k])+' da:')
		# print(da)
		# #da.to_netcdf(out_dir+'da')
			
	#da.to_netcdf(out_dir+"class_conversion.nc")
# for k in range(0,2):
# 	cal_max(k)


	dask.config.set(scheduler='processes')
	lazy_results=[]
	for k in np.arange(0,len(years)-1):
		lazy_result = dask.delayed(cal_max)(k)
		lazy_results.append(lazy_result)
	

	from dask.diagnostics import ProgressBar
	with ProgressBar():
		futures = dask.persist(*lazy_results)
		results = dask.compute(*futures)
	
	fnames = [out_dir+'da_'+str(k)+'.nc' for k in range(2004,2014)]
	new_year = years[:-1]

	conversions = xr.concat([xr.open_dataarray(f) for f in fnames], dim=years[:-1])
	conversions = conversions.rename({'concat_dim':'years'})
	conversions.to_netcdf('conversions.nc')

"""
def argmax(x):
	import numpy as np
	if np.isnan(x).all():
		return np.nan
	else:
		return max(range(len(x)), key=lambda y: x[y])



def class_to_what(x,CType):
	import numpy as np
	if CType == 'EF':
		n=[]
		exceptions = 0 #Excluding EF from the conversion type
		if (x[exceptions]<-50):
			max1 = np.nanmax(np.delete(x, exceptions))
			#max1 = x[a1[a2]] 
			if (abs(x[j-1])-abs(max1))<10:
				I = np.where(x==max1)
				if I[0]==1:
					n=2 			# EF to DF
				elif I[0]==2:
					n=3 			# EF to shrub
				elif I[0]==3:
					n=4			# EF to Herbacuous
				elif I[0]==4:
					n=5			# EF to Sparse/Barren
				elif I[0]==5:
					n=6			# EF to wetlands
				elif I[0]==6:
					n=7			# EF to water
			else:
				n=np.nan
		else:
			n=np.nan
		return n

	elif CType == 'DF':
		n=[]
		exceptions = 1 #Index of the class to be studied for change
		if (x[exceptions]<-50):
			max1 = np.nanmax(np.delete(x, exceptions))
			#max1 = x[a1[a2]] 
			if (abs(x[j-1])-abs(max1))<10:
				I = np.where(x==max1)
				if I[0]==0:
					n=1			# DF to EF
				elif I[0]==2:
					n=3		# DF to shrub
				elif I[0]==3:
					n=4		# DF to Herbasuous
				elif I[0]==4:
					n=5		# DF to Sparse/Barren
				elif I[0]==5:
					n=6		# DF to wetlands
				elif I[0]==6:
					n=7			# EF to water
			else:
				n=np.nan
		else:
			n=np.nan
		return n
	elif CType == 'shrub':
		n=[]
		exceptions = 2 #Index of the class to be studied for change
		if (x[exceptions]<-50):
			max1 = np.nanmax(np.delete(x, exceptions))
			#max1 = x[a1[a2]] 
			if (abs(x[j-1])-abs(max1))<10:
				I = np.where(x==max1)
				if I[0]==0:
					n=1			# DF to EF
				elif I[0]==1:
					n=2		# DF to shrub
				elif I[0]==3:
					n=4		# DF to Herbasuous
				elif I[0]==4:
					n=5		# DF to Sparse/Barren
				elif I[0]==5:
					n=6		# DF to wetlands
				elif I[0]==6:
					n=7			# EF to water
			else:
				n=np.nan
		else:
			n=np.nan
		return n
	elif CType == 'herb':
		n=[]
		exceptions = 3 #Index of the class to be studied for change
		if (x[exceptions]<-50):
			max1 = np.nanmax(np.delete(x, exceptions))
			#max1 = x[a1[a2]] 
			if (abs(x[j-1])-abs(max1))<10:
				I = np.where(x==max1)
				if I[0]==0:
					n=1		# DF to EF
				elif I[0]==1:
					n=2		# DF to shrub
				elif I[0]==2:
					n=3		# DF to Herbasuous
				elif I[0]==4:
					n=5		# DF to Sparse/Barren
				elif I[0]==5:
					n=6		# DF to wetlands
				elif I[0]==6:
					n=7			# EF to water
			else:
				n=np.nan
		else:
			n=np.nan
		return n
	elif CType == 'sparse':
		n=[]
		exceptions = 4 #Index of the class to be studied for change
		if (x[exceptions]<-50):
			max1 = np.nanmax(np.delete(x, exceptions))
			#max1 = x[a1[a2]] 
			if (abs(x[j-1])-abs(max1))<10:
				I = np.where(x==max1)
				if I[0]==0:
					n=1			# DF to EF
				elif I[0]==1:
					n=2		# DF to shrub
				elif I[0]==2:
					n=3		# DF to Herbasuous
				elif I[0]==3:
					n=4		# DF to Sparse/Barren
				elif I[0]==5:
					n=6		# DF to wetlands
				elif I[0]==6:
					n=7			# EF to water
			else:
				n=np.nan
		else:
			n=np.nan
		return n
	elif CType == 'wetland':
		n=[]
		exceptions = 5 #Index of the class to be studied for change
		if (x[exceptions]<-50):
			max1 = np.nanmax(np.delete(x, exceptions))
			#max1 = x[a1[a2]] 
			if (abs(x[j-1])-abs(max1))<10:
				I = np.where(x==max1)
				if I[0]==0:
					n=1			# DF to EF
				elif I[0]==1:
					n=2		# DF to shrub
				elif I[0]==2:
					n=3		# DF to Herbasuous
				elif I[0]==3:
					n=4		# DF to Sparse/Barren
				elif I[0]==4:
					n=5		# DF to wetlands
				elif I[0]==6:
					n=7			# EF to water
			else:
				n=np.nan
		else:
			n=np.nan
		return n
	elif CType == 'water':
		n=[]
		exceptions = 6 #Index of the class to be studied for change
		if (x[exceptions]<-50):
			max1 = np.nanmax(np.delete(x, exceptions))
			#max1 = x[a1[a2]] 
			if (abs(x[j-1])-abs(max1))<10:
				I = np.where(x==max1)
				if I[0]==0:
					n=1			# DF to EF
				elif I[0]==1:
					n=2		# DF to shrub
				elif I[0]==2:
					n=3		# DF to Herbasuous
				elif I[0]==3:
					n=4		# DF to Sparse/Barren
				elif I[0]==4:
					n=5		# DF to wetlands
				elif I[0]==5:
					n=6		# DF to wetlands
			else:
				n=np.nan
		else:
			n=np.nan
		return n
	else:
		print('No class provided')

"""