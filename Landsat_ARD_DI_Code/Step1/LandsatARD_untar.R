rm(list=ls())

######### Begin things need to be changed ###########
ARD_dir <- 'G:/Landsat_DI/H007V014_SRER/' # The root directory where the bulk-downloaded files are located
norder <- 4 # number of bulk download folders
TIFF_dir <- paste(ARD_dir,'GeoTIFF/',sep="") # Create a folder "GeoTIFF" under the same root directory. This is the output directory for the untarred GeoTIFF files
######### End things need to be changed ###########

subdir <- list.dirs(path = ARD_dir, full.names = F, recursive = F)
subdir <- grep('Bulk', subdir, value = TRUE)

for(k in seq(1,norder)){
#for(k in seq(1,1)){
  
  tar_dir<- paste(ARD_dir,subdir[k],'/U.S. Landsat 4-8 ARD/',sep="")
  
  #print(tar_dir)
  setwd(tar_dir)
  xml_arr <- list.files(pattern='.xml')
  
  # Untar the Landsat ARD

  #if(F){}
    for(j in seq(1,length(xml_arr[]))){
    #for(j in seq(1,1)){
      rootfn <- substr(xml_arr[j],1,(nchar(xml_arr[j])-4))
      
      sensor <- substr(xml_arr[j],1,4)
      tile <- substr(xml_arr[j],9,14)
      year <- substr(xml_arr[j],16,19)
      month <- substr(xml_arr[j],20,21)
      day <- substr(xml_arr[j],22,23)
      datestr <- paste(year,'-',month,'-',day, ' 10:15:00 MST', sep="")
      doy <- strftime(datestr,format="%j")
      
      REF_folder <- paste(TIFF_dir, year, '_', doy, '_',tile,'_', sensor,'_SR','/', sep="")
      REF_tar <- paste(tar_dir,rootfn,'_SR.tar',sep="")
      tryCatch(untar(REF_tar,exdir = REF_folder),finally=next)
    }
  
  
  #if(F){}
    for(j in seq(1,length(xml_arr[]))){
      rootfn <- substr(xml_arr[j],1,(nchar(xml_arr[j])-4))
      
      sensor <- substr(xml_arr[j],1,4)
      tile <- substr(xml_arr[j],9,14)
      year <- substr(xml_arr[j],16,19)
      month <- substr(xml_arr[j],20,21)
      day <- substr(xml_arr[j],22,23)
      datestr <- paste(year,'-',month,'-',day, ' 10:15:00 MST', sep="")
      doy <- strftime(datestr,format="%j")
      
      
      LST_folder <- paste(TIFF_dir, year, '_', doy, '_',tile,'_', sensor,'_ST','/', sep="")
      LST_tar <- paste(tar_dir,rootfn,'_ST.tar',sep="")
      tryCatch(untar(LST_tar,exdir = LST_folder),finally=next)
      
    }
  
    for(j in seq(1,length(xml_arr[]))){
    #for(j in seq(1,1)){
      rootfn <- substr(xml_arr[j],1,(nchar(xml_arr[j])-4))
      
      sensor <- substr(xml_arr[j],1,4)
      tile <- substr(xml_arr[j],9,14)
      year <- substr(xml_arr[j],16,19)
      month <- substr(xml_arr[j],20,21)
      day <- substr(xml_arr[j],22,23)
      datestr <- paste(year,'-',month,'-',day, ' 10:15:00 MST', sep="")
      doy <- strftime(datestr,format="%j")
      
      TOA_folder <- paste(TIFF_dir, year, '_', doy, '_',tile,'_', sensor,'_TA','/', sep="")
      TOA_tar <- paste(tar_dir,rootfn,'_TA.tar',sep="")
      tryCatch(untar(TOA_tar,exdir = TOA_folder),finally=next)
    }
  
}