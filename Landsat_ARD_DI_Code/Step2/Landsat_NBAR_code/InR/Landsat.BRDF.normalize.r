
# source("Landsat.BRDF.normalize.r");

# Hankui wrote it on Nov 22 2017 
require("raster")
source("model.r")
FILL_VALUE_L <- -9999;

# /***************************************************************************
# * calculate nadir normalized reflectance (scaled by 10000) for band b data
# * band:	   input Landsat surface reflectance to be normalized (scaled by 10000)
# * band_sz:   solar zenith angle of the Landsat surface refletance (in degree 0~90)
# * band_vz:   viewing zenith angle of the Landsat surface refletance (in degree 0~8)
# * band_sa:   solar azimuth angle of the Landsat surface refletance (in degree 0~360)
# * band_va:   viewing azimuth angle of the Landsat surface refletance (in degree 0~360)
# * b:         band index 

# * return value (Landsat_output): output normalized Landsat surface reflectance (scaled by 10000)

# ***************************************************************************/
NBAR_calculate_global_perband <- function (band, band_sz=band_sz, band_vz=band_vz, band_va=band_va, band_sa=band_sa, b = 1 ){ 
	Landsat_input <- band;
	Landsat_output <- band; 
	
	index <- !is.na(band);
	if (sum(index)!=0) { 
		solar_zenith <- band_sz[index]/100;
		view_zenith  <- band_vz[index]/100;
		relative_azimuth <- (band_va[index]-band_sa[index])/100;
		solar_zenith_output <- solar_zenith;
		Landsat_input <- Landsat_input[index];
		
		srf1 <- calc_refl_noround(pars_array[b, ], view_zenith, solar_zenith, relative_azimuth); 
		srf0 <- calc_refl_noround(pars_array[b, ], rep(0, length(view_zenith)), solar_zenith_output, rep(0, length(view_zenith))  );
		
		Landsat_output[index] <- as.integer((srf0/srf1*Landsat_input[index]));
		
		# Landsat_output[index] <- NBAR_calculate_global_perband(Landsat_input, view_zenith, solar_zenith, relative_azimuth, solar_zenith_output, b=b);
	}
	Landsat_output
}

ptm <- proc.time();

# //*******************************************************************************************
# // initialization 
input_sz_name <- "/gpfs/data2/temp.test/brdf.test/LT05_CU_004002_19860819_20170711_C01_V01_SOZ4.tif";
input_sa_name <- "/gpfs/data2/temp.test/brdf.test/LT05_CU_004002_19860819_20170711_C01_V01_SOA4.tif";
input_vz_name <- "/gpfs/data2/temp.test/brdf.test/LT05_CU_004002_19860819_20170711_C01_V01_SEZ4.tif";
input_va_name <- "/gpfs/data2/temp.test/brdf.test/LT05_CU_004002_19860819_20170711_C01_V01_SEA4.tif";

input_file_name1 <- "/gpfs/data2/temp.test/brdf.test/LT05_CU_004002_19860819_20170711_C01_V01_SRB1.tif";
input_file_name2 <- "/gpfs/data2/temp.test/brdf.test/LT05_CU_004002_19860819_20170711_C01_V01_SRB2.tif";
input_file_name3 <- "/gpfs/data2/temp.test/brdf.test/LT05_CU_004002_19860819_20170711_C01_V01_SRB3.tif";
input_file_name4 <- "/gpfs/data2/temp.test/brdf.test/LT05_CU_004002_19860819_20170711_C01_V01_SRB4.tif";
input_file_name5 <- "/gpfs/data2/temp.test/brdf.test/LT05_CU_004002_19860819_20170711_C01_V01_SRB5.tif";
input_file_name6 <- "/gpfs/data2/temp.test/brdf.test/LT05_CU_004002_19860819_20170711_C01_V01_SRB7.tif";
outputdir <- "/gpfs/data2/temp.test/brdf.test/R.hank" 
# dir.create(outputdir, showWarnings = TRUE, recursive = FALSE, mode = "0777");
if (!dir.exists(outputdir)) dir.create(outputdir, showWarnings = TRUE, recursive = FALSE, mode = "0777");

output_file_name1 <- paste(outputdir, "/LT05_CU_004002_19860819_20170711_C01_V01_NBRinR1.tif", sep='');
output_file_name2 <- paste(outputdir, "/LT05_CU_004002_19860819_20170711_C01_V01_NBRinR2.tif", sep='');
output_file_name3 <- paste(outputdir, "/LT05_CU_004002_19860819_20170711_C01_V01_NBRinR3.tif", sep='');
output_file_name4 <- paste(outputdir, "/LT05_CU_004002_19860819_20170711_C01_V01_NBRinR4.tif", sep='');
output_file_name5 <- paste(outputdir, "/LT05_CU_004002_19860819_20170711_C01_V01_NBRinR5.tif", sep='');
output_file_name6 <- paste(outputdir, "/LT05_CU_004002_19860819_20170711_C01_V01_NBRinR7.tif", sep='');

cat("Reading input data.....", "\n"); 
rasterb <- raster(input_sz_name, layer=1);
nrows <- nrow(rasterb);
ncols <- ncol(rasterb);

band_sz <- as.matrix(raster(input_sz_name, layer=1)); 
band_sa <- as.matrix(raster(input_sa_name, layer=1)); 
band_vz <- as.matrix(raster(input_vz_name, layer=1)); 
band_va <- as.matrix(raster(input_va_name, layer=1)); 

band1 <- as.matrix(raster(input_file_name1, layer=1)); 
band2 <- as.matrix(raster(input_file_name2, layer=1)); 
band3 <- as.matrix(raster(input_file_name3, layer=1)); 
band4 <- as.matrix(raster(input_file_name4, layer=1)); 
band5 <- as.matrix(raster(input_file_name5, layer=1)); 
band6 <- as.matrix(raster(input_file_name6, layer=1)); 

dims <- dim(band1);
width <- dims[1];
lengt <- dims[2];


# //*******************************************************************************************
# // BRDF normalization
# band1_one <- array( band1, dim=( nrows*ncols, 1) ) 
band1_one <- as.vector(band1);
band2_one <- as.vector(band2);
band3_one <- as.vector(band3);
band4_one <- as.vector(band4);
band5_one <- as.vector(band5);
band6_one <- as.vector(band6);

band_sz <- as.vector(band_sz);
band_sa <- as.vector(band_sa);
band_vz <- as.vector(band_vz);
band_va <- as.vector(band_va);


band1_one <- NBAR_calculate_global_perband(band1, band_sz=band_sz, band_vz=band_vz, band_va=band_va, band_sa=band_sa, b = 1);
band2_one <- NBAR_calculate_global_perband(band2, band_sz=band_sz, band_vz=band_vz, band_va=band_va, band_sa=band_sa, b = 2);
band3_one <- NBAR_calculate_global_perband(band3, band_sz=band_sz, band_vz=band_vz, band_va=band_va, band_sa=band_sa, b = 3);
band4_one <- NBAR_calculate_global_perband(band4, band_sz=band_sz, band_vz=band_vz, band_va=band_va, band_sa=band_sa, b = 4);
band5_one <- NBAR_calculate_global_perband(band5, band_sz=band_sz, band_vz=band_vz, band_va=band_va, band_sa=band_sa, b = 5);
band6_one <- NBAR_calculate_global_perband(band6, band_sz=band_sz, band_vz=band_vz, band_va=band_va, band_sa=band_sa, b = 6);

cat("Producing NBAR band 1 .....\n"); band1 <- matrix(band1_one, nrow = nrows,ncol = ncols)
cat("Producing NBAR band 2 .....\n"); band2 <- matrix(band2_one, nrow = nrows,ncol = ncols)
cat("Producing NBAR band 3 .....\n"); band3 <- matrix(band3_one, nrow = nrows,ncol = ncols)
cat("Producing NBAR band 4 .....\n"); band4 <- matrix(band4_one, nrow = nrows,ncol = ncols)
cat("Producing NBAR band 5 .....\n"); band5 <- matrix(band5_one, nrow = nrows,ncol = ncols)
cat("Producing NBAR band 6 .....\n"); band6 <- matrix(band6_one, nrow = nrows,ncol = ncols)

# for (i in c(1:width)) {
	# if (i%%500==0) cat("\tNow BRDF row", i, "\n"); 
	# band1[i, ] <- NBAR_for_line_i_band_b (band1, i=i, b = 1 ); 
	# band2[i, ] <- NBAR_for_line_i_band_b (band2, i=i, b = 2 ); 
	# band3[i, ] <- NBAR_for_line_i_band_b (band3, i=i, b = 3 ); 
	# band4[i, ] <- NBAR_for_line_i_band_b (band4, i=i, b = 4 ); 
	# band5[i, ] <- NBAR_for_line_i_band_b (band5, i=i, b = 5 ); 
	# band6[i, ] <- NBAR_for_line_i_band_b (band6, i=i, b = 6 ); 
# }

# //*******************************************************************************************
# // save nbar tiff file
cat("writeRaster output data.....", "\n"); 
writeRaster(raster(band1), filename=output_file_name1, format="GTiff", datatype='INT2S', overwrite=TRUE);
writeRaster(raster(band2), filename=output_file_name2, format="GTiff", datatype='INT2S', overwrite=TRUE);
writeRaster(raster(band3), filename=output_file_name3, format="GTiff", datatype='INT2S', overwrite=TRUE);
writeRaster(raster(band4), filename=output_file_name4, format="GTiff", datatype='INT2S', overwrite=TRUE);
writeRaster(raster(band5), filename=output_file_name5, format="GTiff", datatype='INT2S', overwrite=TRUE);
writeRaster(raster(band6), filename=output_file_name6, format="GTiff", datatype='INT2S', overwrite=TRUE);

time.my <- proc.time() - ptm;
cat(paste("\n", sprintf("%.1f",time.my[1]/60),"minutes\n"));


# source("Landsat.BRDF.normalize.r");



## For back up 
# /***************************************************************************
# * calculate nadir normalized reflectance (scaled by 10000)
# * Landsat_input:		 input six bands (in the order of 1, 2, 3, 4, 5 and 7) Landsat surface reflectance to be normalized (scaled by 10000)
# * Landsat_output:		 output normalized six bands (in the order of 1, 2, 3, 4, 5 and 7) Landsat surface reflectance (scaled by 10000)
# * view_zenith:         viewing zenith angle of the Landsat surface refletance (in degree 0~8)
# * solar_zenith:        solar zenith angle of the Landsat surface refletance (in degree 0~90)
# * relative_azimuth:    relative azimuth angle of obtained by (soloar_azimuth-viewing_azimuth) or (viewing_azimuth-soloar_azimuth) (in degree -180~180)
# * solar_zenith_output: output solar zenith angle the "Landsat_input" to be normalize to
# ***************************************************************************/
NBANDS_L <- 6; 
NBAR_calculate_global_per_pixel <- function(Landsat_input, view_zenith, solar_zenith,
		relative_azimuth, solar_zenith_output)
{

	# //input kernel values
	# double nbarkerval[3];
	nbarkerval <- CalculateKernels(view_zenith*DE2RA, solar_zenith*DE2RA, relative_azimuth*DE2RA);

	# //output kernel values
	# double nbarkerval_i[3];
	nbarkerval_i <- CalculateKernels(0*DE2RA, solar_zenith_output*DE2RA, 180.0*DE2RA);
	Landsat_output <- Landsat_input;
	# // process for each band
	# double nbar,modis_srf;
	# int16 pars[3];
	# int i = 0;
	# int b;
	for (b in c(1:NBANDS_L))
	{
		# for (i=0;i<3;i++)
		# {
			# pars[i] = pars_12m_global[b][i];
		# }

		# // it does not matter what is the relative_azimuth value when the view zenith is zero tested by Hankui
		# nbar = nbarkerval_i[0] * pars[0] +  nbarkerval_i[1] * pars[1] +  nbarkerval_i[2] * pars[2];
		# modis_srf = nbarkerval[0] * pars[0] +  nbarkerval[1] * pars[1] +  nbarkerval[2] * pars[2];
		modis_srf <- sum(nbarkerval*pars_array[b, ]);
		nbar <- sum(nbarkerval_i*pars_array[b, ]);
		
		if (is.na(Landsat_input[b]) || nbar <=0 || modis_srf<=0 || Landsat_input[b]==FILL_VALUE_L) {
			Landsat_output[b] = FILL_VALUE_L;
		} else
			# //ratio based normalization
			Landsat_output[b] =  as.integer((nbar/modis_srf*Landsat_input[b]));
	}
	Landsat_output
}




# Landsat_input <- vector(, length=NBANDS_L);
# Landsat_output <- Landsat_input;
# for (i in c(1:width)) {
	# if (i%%500==0) cat("Now BRDF row", i, "\n"); 
	# for (j in c(1:lengt)) {
		# Landsat_input[1] <- band1[i,j];
		# Landsat_input[2] <- band2[i,j];
		# Landsat_input[3] <- band3[i,j];
		# Landsat_input[4] <- band4[i,j];
		# Landsat_input[5] <- band5[i,j];
		# Landsat_input[6] <- band6[i,j];
		# solar_zenith <- band_sz[i,j]/100;
		# view_zenith <- band_vz[i,j]/100;
		# relative_azimuth <- (band_va[i,j]-band_sa[i,j])/100;
		# solar_zenith_output <- solar_zenith;
		# if (sum(is.na(Landsat_input))!=NBANDS_L) {
			# Landsat_output <- NBAR_calculate_global(Landsat_input, view_zenith, solar_zenith,
				# relative_azimuth, solar_zenith_output); 
		
			
			# band1[i,j] <- Landsat_output[1] ;
			# band2[i,j] <- Landsat_output[2] ;
			# band3[i,j] <- Landsat_output[3] ;
			# band4[i,j] <- Landsat_output[4] ;
			# band5[i,j] <- Landsat_output[5] ;
			# band6[i,j] <- Landsat_output[6] ;			
		# }
		
		
	# }
# }