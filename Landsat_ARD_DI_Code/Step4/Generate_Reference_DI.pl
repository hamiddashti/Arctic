#!/usr/bin/perl

$ARD_dir = "/data/home/yand/LandsatARD/H007V014_SRER/"; # root directory
$annual_asc = join("",$ARD_dir,"cur_year_asc");
$VI_indir = join("",$ARD_dir,"VI_LST/"); # the directory for VI and LST datacubes
$VI_outdir = join("",$ARD_dir,"ARD_REF_DI/"); # ouput directory for reference DI
$VI_idx = 2; # The index of the VI that will be used to calculate reference VI

if($VI_idx==0){
   $VI = "NDVI";
}elsif($VI_idx==1){
   $VI = "NIRv";
}elsif($VI_idx==2){
   $VI = "EVI";
}elsif($VI_idx==3){
   $VI = "EVI2";
}elsif($VI_idx==4){
   $VI = "SAVI";
}elsif($VI_idx==5){
   $VI = "MSAVI";
}elsif($VI_idx==6){
   $VI = "NDMI";
}elsif($VI_idx==7){
   $VI = "NBR";
}elsif($VI_idx==8){
   $VI = "NBR2";
}


for($yy=1984;$yy<=2019;$yy++){
#for($yy=2002;$yy<=2002;$yy++){
 
   print "$yy\n";
   $filein = join("",$VI_indir,"*",$yy,"*.dat");
   system("ls $filein > $annual_asc");
   
   $fileout = join("",$VI_outdir,"REF_DI_",$yy,"_",$VI,".dat");

   system("./Generate_Reference_DI.exe $annual_asc $VI_idx $fileout");

}


