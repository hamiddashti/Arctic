rm(list=ls())

######### Begin things need to be changed ###########
ARD_dir <- '/data/ABOVE/LANDSAT/ARD/h08v03/' # The root directory where the bulk-downloaded files are located

norder <- 1 # If you have placed all the binary files generated in step2 in a single folder, set norder to 1.
######### End things need to be changed ###########

NBAR_list <- paste(ARD_dir,"NBAR_file_list.txt",sep="")
NBAR_QA_list <- paste(ARD_dir,"NBAR_QA_list.txt",sep="")
LST_list <- paste(ARD_dir,"LST_file_list.txt",sep="")

outdir <- paste(ARD_dir,"VI_LST/",sep="")
if (!dir.exists(outdir)) dir.create(outdir, showWarnings = TRUE, recursive = FALSE, mode = "0777");

for(k in seq(1,norder)){
  Input_dir <- paste(ARD_dir,"GeoTIFF",sep="")
  NBAR_dirs <- dir(path=Input_dir,pattern='NBAR_',full.names = T) 
  nfolder <- length(NBAR_dirs)
  
  for(j in seq(1,nfolder)){
     sensor <- substr(NBAR_dirs[j],(nchar(NBAR_dirs[j])-1),nchar(NBAR_dirs[j]))
     filestr <- substr(NBAR_dirs[j],(nchar(NBAR_dirs[j])-20),nchar(NBAR_dirs[j]))
     
     sr_files <- list.files(path=NBAR_dirs[j], pattern=glob2rx("*_NBAR_*.envi"),full.names = T)
     lst_files <- list.files(path=NBAR_dirs[j], pattern=glob2rx("*_ST*.envi"),full.names = T)
     pqa <- list.files(path=NBAR_dirs[j], pattern=glob2rx("*_PIXELQA.envi"),full.names = T)
     radsatqa <- list.files(path=NBAR_dirs[j], pattern=glob2rx("*_RADSATQA.envi"),full.names = T)
     
     if((length(sr_files)==6)&&(length(pqa)==1)&&(length(radsatqa)==1)){
       
       if(length(lst_files)==2){
         LST_mode <- 1
       }else{
         LST_mode <- 0
       }
       
       in_header <- paste(substr(sr_files[1],1,nchar(sr_files[1])-5),".hdr",sep="")
       
       out_datacube <- paste(outdir,"ARDCube",filestr,".dat",sep="")
       print(out_datacube)
       out_header <- paste(outdir,"ARDCube",filestr,".dat.hdr",sep="")
       system(paste("cp", in_header,out_header,sep=" "))
       system(paste("sed -i 's/","bands   = 1/bands   = 12/'"," ",out_header,sep=""))
       system(paste("sed -i 's/","interleave = bsq/interleave = bip/'"," ",out_header,sep=""))
       system(paste("sed -i 's/","Band 1}/NDVI, NIRv, EVI, EVI2, SAVI, MSAVI, NDMI, NBR, NBR2, VI_QA, LST, LST_QA}/'"," ",out_header,sep=""))
       
       #system(paste("mv",in_header,out_header,sep=" "))
       
       if(sensor=="08"){
         
         AEQA <- list.files(path=NBAR_dirs[j], pattern=glob2rx("*_SRAEROSOLQA.envi"),full.names = T)
         
         if(length(AEQA)==1){
           
           system(paste("ls ",NBAR_dirs[j],"/*NBAR*.envi"," > ",NBAR_list,sep=""))
           
           system(paste("ls ",pqa," > ",NBAR_QA_list,sep=""))
           system(paste("ls ",radsatqa," >> ",NBAR_QA_list,sep=""))
           system(paste("ls ",AEQA," >> ",NBAR_QA_list,sep=""))
           
           if(LST_mode==1){
             system(paste("ls ",NBAR_dirs[j],"/*_ST*.envi"," > ",LST_list,sep=""))
             system(paste(ARD_dir,"Generate_L8ARD_DataCube.exe ",NBAR_list," ",NBAR_QA_list," ",LST_mode," ",out_datacube," ",LST_list,sep=""))
           }else{
             system(paste(ARD_dir,"Generate_L8ARD_DataCube.exe ",NBAR_list," ",NBAR_QA_list," ",LST_mode," ",out_datacube,sep=""))
           }
           
         }
         
       }else{
         
         system(paste("ls ",NBAR_dirs[j],"/*NBAR*.envi"," > ",NBAR_list,sep=""))
         
         system(paste("ls ",pqa," > ",NBAR_QA_list,sep=""))
         system(paste("ls ",radsatqa," >> ",NBAR_QA_list,sep=""))
         #system(paste("ls ",AEQA," >> ",NBAR_QA_list,sep=""))
         
         if(LST_mode==1){
           system(paste("ls ",NBAR_dirs[j],"/*_ST*.envi"," > ",LST_list,sep=""))
           system(paste(ARD_dir,"Generate_L47ARD_DataCube.exe ",NBAR_list," ",NBAR_QA_list," ",LST_mode," ",out_datacube," ",LST_list,sep=""))
         }else{
           system(paste(ARD_dir,"Generate_L47ARD_DataCube.exe ",NBAR_list," ",NBAR_QA_list," ",LST_mode," ",out_datacube,sep=""))
         }
         
         
         
       }
       
     }
  } 

  
  
  
  
}
