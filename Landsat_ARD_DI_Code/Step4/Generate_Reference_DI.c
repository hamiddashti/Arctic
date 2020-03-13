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

int main(int argc, char** argv)
{
  FILE *fDI,*fyear_asc,*fpre[max_count];
  int i,j,k,vidx,ndat,nblock,tt;
  short pre_temp[nb*npix];

  char imgname[500],ascname[500];

   if((fyear_asc=fopen(argv[1],"r"))==NULL)
     {
       printf("Failure open the file list for multi-year mean ARD cube\n");
       exit(1);
     }

   vidx = atoi(argv[2]);

   if((fDI=fopen(argv[3],"wb"))==NULL)
     {
       printf("Failure open the file list for multi-year mean ARD cube\n");
       exit(1);
     }


   ndat=0;

   while(fscanf(fyear_asc,"%s",imgname)!=EOF)
       {
         if((fpre[ndat]=fopen(imgname,"rb"))==NULL)
           {
             //printf("%d\n",ndat);
             //printf("Cann't open the file for NBAR QA input file 1\n");
             printf("Cannot open file %s.error %s\n",imgname,strerror(errno));
             exit(1);
           }
	     ndat++;    
       }

   //printf("ndat = %d\n",ndat);

   float pre_data[ndat][npix][4];
   float LST_max[npix],VI_max[npix],DI_pre[npix];
   nblock = (int)((ncols*1.0)*(nrows*1.0)/(npix*1.0));

  for(i=0;i<nblock;i++){
     for(k=0;k<ndat;k++){

      fread(pre_temp,nb*npix,sizeof(short),fpre[k]);

     for(j=0;j<npix;j++){

        if(pre_temp[j*nb+vidx]!=NoData){
           pre_data[k][j][0] = pre_temp[j*nb+vidx]*1.0*VI_SCF; //VI
          }

          pre_data[k][j][1] = pre_temp[j*nb+9]*1.0;//QA

         if(pre_temp[j*nb+10]!=NoData){
            pre_data[k][j][2] = pre_temp[j*nb+10]*1.0*LST_SCF-273.15;//LST
         }

         if(pre_temp[j*nb+11]!=NoData){
            pre_data[k][j][3] = pre_temp[j*nb+11]*1.0*0.01;//LST_QA
         }

      }//for(j=0;j<npix;j++)

     }//for(k=0;k<ndat;k++)


     for(j=0;j<npix;j++){

      DI_pre[j] = DI_FILL;
      LST_max[j] = -9999.0;
      VI_max[j] = -9999.0;

         for(k=0;k<ndat;k++){
               /*
               if(i==17975&&j==873){
                 printf("k=%d LST=%f\n",k,pre_data[k][j][2]);
               }*/

            if(pre_data[k][j][2]>LST_max[j]){
               LST_max[j] = pre_data[k][j][2];
            }

            if((pre_data[k][j][0]>VI_max[j])&&(pre_data[k][j][1]==20.0||pre_data[k][j][1]==21.0)){
               VI_max[j] = pre_data[k][j][0];
            }
         }

         if(LST_max[j]>0&&VI_max[j]>0){
           DI_pre[j] = LST_max[j]/VI_max[j];
         }
         /*
         if(i==17975&&j==873){
           printf("LST_max=%f VI_max=%f DI=%f\n",LST_max[j],VI_max[j],DI_pre[j]);
         }*/

     }//for(j=0;j<npix;j++)


    fwrite(DI_pre,npix,sizeof(float),fDI);

  }//for(i=0;i<nblock;i++){


 for(k=0;k<ndat;k++){
     fclose(fpre[k]);
 }

 fclose(fyear_asc);
 fclose(fDI);
 return(0);
}



