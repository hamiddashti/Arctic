#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <string.h>
#include <errno.h>


#define nb 12
#define max_count 200 
#define ncols 5000
#define nrows 5000 
#define npix 1000
#define NoData -32768
#define DI_FILL -1.0
#define LGDI_threshold 0.65
#define VI_SCF 0.0001
#define LST_SCF 0.1
#define year_ini 1984
#define debug_j 1367
#define debug_i 201

int main(int argc, char** argv)
{
  FILE *fpre_asc, *fcur_asc, *fDI;
  FILE *fpre[max_count],*fcur[max_count];
  int npre, ncur,year_cur,nyear,vidx,nblock,i,j,k;
  char imgname[500];

    if((fpre_asc=fopen(argv[1],"r"))==NULL)
     {
       printf("Failure open the file list for multi-year mean ARD cube\n");
       exit(1);
     }

  npre = 0;

   while(fscanf(fpre_asc,"%s",imgname)!=EOF)
       {
         if((fpre[npre]=fopen(imgname,"rb"))==NULL)
           {
             //printf("%d\n",ndat);
             printf("Cannot open file %s.error %s\n",imgname,strerror(errno));
             exit(1);
           }
	     npre++;    
       }

 if((fcur_asc=fopen(argv[2],"r"))==NULL)
   {
       printf("Failure open the file list for multi-year mean ARD cube\n");
       exit(1);
   }

  ncur = 0;


   while(fscanf(fcur_asc,"%s",imgname)!=EOF)
       {
         if((fcur[ncur]=fopen(imgname,"rb"))==NULL)
           {
             //printf("%d\n",ndat);
             printf("Cannot open file %s.error %s\n",imgname,strerror(errno));
             exit(1);
           }
	     ncur++;    
       }


   vidx = atoi(argv[3]);

   if((fDI=fopen(argv[4],"wb"))==NULL)
     {
       printf("Failure open the file list for multi-year mean ARD cube\n");
       exit(1);
     }



  float REF_DI[npre][npix];
  short cur_temp[nb*npix];
  float cur_data[ncur][npix][4];
  float LST_max[npix],VI_max[npix],k0[npix],nvalid[npix],DI_sum[npix],STD_sum[npix],DI_mean[npix],DI_std[npix],DI_squ[npix],DI_inst[npix],DI_cur[npix],DI_bin[npix];

  nblock = (int)((ncols*1.0)*(nrows*1.0)/(npix*1.0));

for(i=0;i<nblock;i++){

//printf("Block %d\n",i);

 for(k=0;k<npre;k++){
     fread(REF_DI[k],npix,sizeof(float),fpre[k]);
 }//for(k=0;k<pre;k++)

  for(k=0;k<ncur;k++)
       {
          fread(cur_temp,nb*npix,sizeof(short),fcur[k]);

          for(j=0;j<npix;j++){

             if(cur_temp[j*nb+vidx]!=NoData){
                cur_data[k][j][0] = cur_temp[j*nb+vidx]*1.0*VI_SCF;
               }

                cur_data[k][j][1] = cur_temp[j*nb+9]*1.0;//QA

            if(cur_temp[j*nb+10]!=NoData){
               cur_data[k][j][2] = cur_temp[j*nb+10]*1.0*LST_SCF-273.15;//LST
             }

           if(cur_temp[j*nb+11]!=NoData){
               cur_data[k][j][3] = cur_temp[j*nb+11]*1.0*0.01;//LST_QA
            }

          }//for(j=0;j<npix;j++)

  }//for(k=0;k<ncur;k++)



for(j=0;j<npix;j++){

    DI_cur[j] = DI_FILL;
    DI_mean[j] = DI_FILL;
    DI_std[j] = DI_FILL;
    DI_inst[j] = DI_FILL;
    DI_bin[j] = DI_FILL;

    LST_max[j] = -9999.0;
    VI_max[j] = -9999.0;
    k0[j] = 0;

    for(k=0;k<ncur;k++){

       if(cur_data[k][j][2]>LST_max[j]){
          LST_max[j] = cur_data[k][j][2];
          k0[j] = k;
       }

    }//for(k=0;k<ncur;k++)


       if(LST_max[j]>0){

         for(k=(k0[j]+1);k<ncur;k++){
            if((cur_data[k][j][0]>VI_max[j])&&(cur_data[k][j][1]==20.0||cur_data[k][j][1]==21.0)){
               VI_max[j] = cur_data[k][j][0];
            }
         }
       }


       if(LST_max[j]>0&&VI_max[j]>0){
          DI_cur[j] = LST_max[j]/VI_max[j];
       }



      nvalid[j] = 0.0;
      DI_sum[j] = 0.0;
      DI_squ[j] = 0.0;

for(k=0;k<npre;k++){

    if(REF_DI[k][j]>0){

      nvalid[j] = nvalid[j]+1;
      DI_sum[j] = DI_sum[j] + REF_DI[k][j];
    }
}


if(nvalid[j]>0){

   DI_mean[j] = DI_sum[j]/nvalid[j];

if(DI_mean[j]>0){

  if(DI_cur[j]>0){

   DI_inst[j] = DI_cur[j]/DI_mean[j];
  }

for(k=0;k<npre;k++){

    if(REF_DI[k][j]>0){

      //nvalid[j] = nvalid[j]+1;
      DI_squ[j] = DI_squ[j] + (REF_DI[k][j] - DI_mean[j])*(REF_DI[k][j] - DI_mean[j]);
    }
}


if(DI_squ[j]>0){


DI_std[j] = sqrt(DI_squ[j]/nvalid[j]);

if(DI_cur[j]>0){

if((DI_cur[j]-DI_mean[j]) > (2.0*DI_std[j])){

    DI_bin[j] = 1.0;
}

}


}


}//if(DI_mean[j]>0)


}//if(nvalid[j]>0)



fwrite(&DI_inst[j],1,sizeof(float),fDI);
fwrite(&DI_cur[j],1,sizeof(float),fDI);
fwrite(&DI_mean[j],1,sizeof(float),fDI);
fwrite(&DI_std[j],1,sizeof(float),fDI);
fwrite(&DI_bin[j],1,sizeof(float),fDI);

}//for(j=0;j<npix;j++)



}//for(i=0;i<nblock;i++)


for(k=0;k<npre;k++)
  { 
    fclose(fpre[k]);
  }

for(k=0;k<ncur;k++)
  { 
     fclose(fcur[k]);
  }

fclose(fpre_asc);
fclose(fcur_asc);
fclose(fDI);
return(0);
}
