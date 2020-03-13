rm(list=ls())

######### Begin things need to be changed ###########
ARD_dir <- '/data/home/yand/LandsatARD/H007V014_SRER/'# the root directory
REF_VI_str <- 'EVI' # VI used to calculate reference DI
vidx <- 2 # VI to be used to calcuate DI for the current year, which needs to be consistent with VI for reference DI
######### End things need to be changed ###########

REF_DI_list <- paste(ARD_dir,"Reference_DI_list.txt",sep="")
Annual_VILST_list <- paste(ARD_dir,"Annual_VILST_list.txt",sep="")
REF_DI_dir <- paste(ARD_dir,"ARD_REF_DI/",sep="")
VILST_dir <- paste(ARD_dir,"VI_LST/",sep="")
out_dir <- paste(ARD_dir,"ARD_DI/",sep="")
template_header <- paste(ARD_dir,"LGDI_header_template.hdr",sep="")
  
if(vidx==0){
  VI_str <- "NDVI"
}else if(vidx==1){
  VI_str <- "NIRv"
}else if(vidx==2){
  VI_str <- "EVI"
}else if(vidx==3){
  VI_str <- "EVI2"
}else if(vidx==4){
  VI_str <- "SAVI"
}else if(vidx==5){
  VI_str <- "MSAVI"
}else if(vidx==6){
  VI_str <- "NDMI"
}else if(vidx==7){
  VI_str <- "NBR"
}else if(vidx==8){
  VI_str <- "NBR2"
}

for(k in seq(1986,2019)){
  
   print(k)
  
   for(i in seq(1984,(k-1))){
     REF_DI_filename <- paste(REF_DI_dir,"REF_DI_",i,"_",REF_VI_str,".dat",sep="")
     
     if(i==1984){
       system(paste("ls ", REF_DI_filename," > ",REF_DI_list,sep=""))
     }else{
       system(paste("ls ", REF_DI_filename," >> ",REF_DI_list,sep=""))
     }
   }
  
    system(paste("ls ",VILST_dir,"ARDCube_*",k,"*.dat"," > ",Annual_VILST_list,sep=""))
    
    out_file <- paste(out_dir,"LGDI_",k,"_",VI_str,".dat",sep="")
    out_header <- paste(out_file,".hdr",sep="")
    system(paste("cp ",template_header," ",out_header,sep=""))
    
    system(paste("./Generate_ARD_LGDI_v2.exe ",REF_DI_list," ",Annual_VILST_list," ",vidx," ",out_file,sep=""))
    
    
  
}

