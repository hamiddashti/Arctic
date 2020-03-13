rm(list=ls())

######### Begin things need to be changed ###########
ARD_dir <- '/data/home/yand/LandsatARD/H007V014_SRER/' # The root directory where the bulk-downloaded files are located
NBAR_code_dir <- '/data/home/yand/LandsatARD/H007V014_SRER/Landsat_NBAR_code/InR/' # directory of the BRDF correction code
TIFF_dir <- paste(ARD_dir,'GeoTIFF4',sep="") # output directory

######### End things need to be changed ###########

setwd(NBAR_code_dir)

SR_dir <- dir(path=TIFF_dir,pattern='_SR',full.names = T)
nImgs <- length(SR_dir)

for(k in seq(1,nImgs)){
#for(k in seq(383,nImgs)){
  
  # # **** six surface reflectance tiff files
  rootfn <- substr(SR_dir[k],(nchar(TIFF_dir)+2),(nchar(SR_dir[k])-3))
  
  ST_dir <- paste(TIFF_dir,'/',rootfn,'_ST/',sep='')
  TA_dir <- paste(TIFF_dir,'/',rootfn,'_TA/',sep='')
  NBAR_dir <- paste(TIFF_dir,'/NBAR_',rootfn,'/',sep='')
  
  if(dir.exists(TA_dir)){
    print(TA_dir)
    #print(length(nImgs))
    
    if (!dir.exists(NBAR_dir)) dir.create(NBAR_dir, showWarnings = TRUE, recursive = FALSE, mode = "0777");
    #if(!dir.exists(NBAR_dir)){
    #    dir.create(NBAR_dir)
    #}
    
    # Get input surface reflectance file names
    SR_TIFF_FULL <- list.files(path=SR_dir[k], pattern='_SRB',full.names = T)
    SR_TIFF <- list.files(path=SR_dir[k], pattern='_SRB',full.names = F)
    input_file_name <- rep(NA,6)
    sensor <- substr(SR_TIFF[1],1,4)
    if(sensor=='LC08'){
      band_idx <- seq(2,7)
    }else{
      band_idx <- seq(1,6)
    }
    input_file_name <- SR_TIFF_FULL[band_idx]
    
    #Get surface reflectance QA file names
    SRQA_TIFF_FULL <- list.files(path=SR_dir[k], pattern='QA.tif',full.names = T)
    if(sensor=='LC08'){
      QA_names <- c('_LINEAGEQA','_PIXELQA','_RADSATQA','_SRAEROSOLQA')
      QA_datatypes <- c('INT1U','INT2U','INT2U','INT1U')
    }else{
      QA_names <- c('_LINEAGEQA','_PIXELQA','_RADSATQA','_SRATMOSOPACITYQA','_SRCLOUDQA')
      QA_datatypes <- c('INT1U','INT2U','INT1U','INT2S','INT1U')
    }
    
    
    # Get input sensor and solar angle file names
    Angular_FULL <- list.files(path=TA_dir, pattern='_S',full.names = T)
    input_va_name <- Angular_FULL[1]
    input_vz_name <- Angular_FULL[2]
    input_sa_name <- Angular_FULL[3]
    input_sz_name <- Angular_FULL[4]
    
    #Set output NBAR file names
    
    
    output_file_name <- rep(NA,6)
    for(j in seq(1,6)){
      output_file_name[j] <- paste(NBAR_dir,rootfn,'_NBAR_B',j,sep='')
    }
    
    
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
        #solar_zenith_output <- rep(30,length(solar_zenith)) modify this line if you want a fixed solar zenith output
        Landsat_input <- Landsat_input[index];
        
        srf1 <- calc_refl_noround(pars_array[b, ], view_zenith, solar_zenith, relative_azimuth); 
        srf0 <- calc_refl_noround(pars_array[b, ], rep(0, length(view_zenith)), solar_zenith_output, rep(0, length(view_zenith))  );
        
        Landsat_output[index] <- as.integer((srf0/srf1*Landsat_input));
        
        # Landsat_output[index] <- NBAR_calculate_global_perband(Landsat_input, view_zenith, solar_zenith, relative_azimuth, solar_zenith_output, b=b);
      }
      Landsat_output
    }
    
    ptm <- proc.time();
    
   
    cat("Reading input data.....", "\n"); 
    rasterb <- raster(input_sz_name, layer=1);
    crs <- crs(rasterb)
    # extent <- extent(rasterb)
    nrows <- nrow(rasterb);
    ncols <- ncol(rasterb);
    
    band_sz <- as.matrix(raster(input_sz_name, layer=1)); 
    band_sa <- as.matrix(raster(input_sa_name, layer=1)); 
    band_vz <- as.matrix(raster(input_vz_name, layer=1)); 
    band_va <- as.matrix(raster(input_va_name, layer=1)); 
    
    band1 <- as.matrix(raster(input_file_name[1], layer=1)); 
    band2 <- as.matrix(raster(input_file_name[2], layer=1)); 
    band3 <- as.matrix(raster(input_file_name[3], layer=1)); 
    band4 <- as.matrix(raster(input_file_name[4], layer=1)); 
    band5 <- as.matrix(raster(input_file_name[5], layer=1)); 
    band6 <- as.matrix(raster(input_file_name[6], layer=1)); 
    
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
    
    
    cat("Producing NBAR band 1 .....\n"); band1_one <- NBAR_calculate_global_perband(band1, band_sz=band_sz, band_vz=band_vz, band_va=band_va, band_sa=band_sa, b = 1);
    cat("Producing NBAR band 2 .....\n"); band2_one <- NBAR_calculate_global_perband(band2, band_sz=band_sz, band_vz=band_vz, band_va=band_va, band_sa=band_sa, b = 2);
    cat("Producing NBAR band 3 .....\n"); band3_one <- NBAR_calculate_global_perband(band3, band_sz=band_sz, band_vz=band_vz, band_va=band_va, band_sa=band_sa, b = 3);
    cat("Producing NBAR band 4 .....\n"); band4_one <- NBAR_calculate_global_perband(band4, band_sz=band_sz, band_vz=band_vz, band_va=band_va, band_sa=band_sa, b = 4);
    cat("Producing NBAR band 5 .....\n"); band5_one <- NBAR_calculate_global_perband(band5, band_sz=band_sz, band_vz=band_vz, band_va=band_va, band_sa=band_sa, b = 5);
    cat("Producing NBAR band 6 .....\n"); band6_one <- NBAR_calculate_global_perband(band6, band_sz=band_sz, band_vz=band_vz, band_va=band_va, band_sa=band_sa, b = 6);
    
    band1 <- matrix(band1_one, nrow = nrows,ncol = ncols)
    band2 <- matrix(band2_one, nrow = nrows,ncol = ncols)
    band3 <- matrix(band3_one, nrow = nrows,ncol = ncols)
    band4 <- matrix(band4_one, nrow = nrows,ncol = ncols)
    band5 <- matrix(band5_one, nrow = nrows,ncol = ncols)
    band6 <- matrix(band6_one, nrow = nrows,ncol = ncols)
    
    
    # //*******************************************************************************************
    # // save nbar binary file
    cat("writeRaster NBAR binary files.....", "\n"); 
    writeRaster(raster(band1,crs=crs,template=rasterb), filename=output_file_name[1], format="ENVI", datatype='INT2S', overwrite=TRUE);
    writeRaster(raster(band2,crs=crs,template=rasterb), filename=output_file_name[2], format="ENVI", datatype='INT2S', overwrite=TRUE);
    writeRaster(raster(band3,crs=crs,template=rasterb), filename=output_file_name[3], format="ENVI", datatype='INT2S', overwrite=TRUE);
    writeRaster(raster(band4,crs=crs,template=rasterb), filename=output_file_name[4], format="ENVI", datatype='INT2S', overwrite=TRUE);
    writeRaster(raster(band5,crs=crs,template=rasterb), filename=output_file_name[5], format="ENVI", datatype='INT2S', overwrite=TRUE);
    writeRaster(raster(band6,crs=crs,template=rasterb), filename=output_file_name[6], format="ENVI", datatype='INT2S', overwrite=TRUE);

    # // save sun-sensor geometry binary file
    band_sz <- matrix(band_sz,nrow = nrows,ncol = ncols);
    band_sa <- matrix(band_sa,nrow = nrows,ncol = ncols);
    band_vz <- matrix(band_vz,nrow = nrows,ncol = ncols);
    band_va <- matrix(band_va,nrow = nrows,ncol = ncols);
    
    writeRaster(raster(band_sz,crs=crs,template=rasterb), filename=paste(NBAR_dir,rootfn,'_SZ',sep=''), format="ENVI", datatype='INT2S', overwrite=TRUE);
    writeRaster(raster(band_sa,crs=crs,template=rasterb), filename=paste(NBAR_dir,rootfn,'_SA',sep=''), format="ENVI", datatype='INT2S', overwrite=TRUE);
    writeRaster(raster(band_vz,crs=crs,template=rasterb), filename=paste(NBAR_dir,rootfn,'_VZ',sep=''), format="ENVI", datatype='INT2S', overwrite=TRUE);
    writeRaster(raster(band_va,crs=crs,template=rasterb), filename=paste(NBAR_dir,rootfn,'_VA',sep=''), format="ENVI", datatype='INT2S', overwrite=TRUE);
    
    
    # // save NBAR QA binary file
    cat("writeRaster NBAR QA files.....", "\n");
    for(j in seq(1,length(SRQA_TIFF_FULL))){
      rasterb <- raster(SRQA_TIFF_FULL[j], layer=1);
      crs <- crs(rasterb)
      QA_arr <- as.matrix(raster(SRQA_TIFF_FULL[j], layer=1)); 
      # extent <- extent(rasterb)
      writeRaster(raster(QA_arr,crs=crs,template=rasterb), filename=paste(NBAR_dir,rootfn,QA_names[j],sep=''), format="ENVI", datatype=QA_datatypes[j], overwrite=TRUE);
    }
    
    
    # // save LST and QA binary file
    cat("writeRaster LST and QA binary files.....", "\n");
    if(dir.exists(ST_dir)){
      ST_TIFF_FULL <- list.files(path=ST_dir, pattern='_ST',full.names = T)
      for(j in seq(1,2)){
        if(j==1){
          ST_name <- '_ST'
        }else{
          ST_name <- '_STQA'
        }
        rasterb <- raster(ST_TIFF_FULL[j], layer=1);
        crs <- crs(rasterb)
        ST_arr <- as.matrix(raster(ST_TIFF_FULL[j], layer=1)); 
        writeRaster(raster(ST_arr,crs=crs,template=rasterb), filename=paste(NBAR_dir,rootfn,ST_name,sep=''), format="ENVI", datatype='INT2S', overwrite=TRUE);
      }
    }
    
    time.my <- proc.time() - ptm;
    cat(paste("\n", sprintf("%.1f",time.my[1]/60),"minutes\n"))    
    
  }#if(dir.exists(TA_dir))
}


     



