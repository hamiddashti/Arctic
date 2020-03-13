#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <string.h>

#define nb 6 
#define nqa 2 
#define nst 2
#define ncols 5000
#define nrows 5000 
#define nout 12
#define NoData -32768

void Decode_PQA(unsigned short quo, short *BitArr, short nbits);
void Decode_AEQA(unsigned char quo_char, short *BitArr, short nbits);
short Calculate_VI(short band1, short band2, short band3, short band4, short band5, short VI_idx);

int main(int argc, char** argv)
{
 FILE *fNBAR_asc, *fQA_asc, *fLST_asc;
 FILE *fNBAR[nb], *fQA[nqa], *fLST[nst];
 FILE *fcube;
 
 int i,j,k,mode;
 char imgname[500];

 short BLU[ncols],GRN[ncols],RED[ncols];
 short NIR[ncols],SWIR1[ncols],SWIR2[ncols];

 short LST[ncols],LST_QA[ncols];

 unsigned short PQA[ncols];
 unsigned char RADSAT[ncols];

 //short PQA_bits[16], RADSAT_bits[16], AEQA_bits[8];
   short PQA_bits[16], RADSAT_bits[8];

 //short data[nout][ncols];
 short NDVI,NIRv,EVI,EVI2,SAVI,MSAVI,NDMI,NBR,NBR2,VI_QA;

 // Read NBAR binary files
 if((fNBAR_asc=fopen(argv[1],"r"))==NULL)
  {
    printf("Cann't open the file for NBAR input file list \n");
    exit(1);
  }

/**/
 for(k=0;k<nb;k++)
    { 
      fscanf(fNBAR_asc,"%s",imgname);
      
      if((fNBAR[k]=fopen(imgname,"rb"))==NULL)
       {
          printf("Cann't open the file for NBAR input file\n");
          exit(1);
      }
 
    }

 // Read NBAR QA binary files
 if((fQA_asc=fopen(argv[2],"r"))==NULL)
  {
    printf("Cann't open the file for NBAR QA input file list \n");
    exit(1);
  }

/**/
 for(k=0;k<nqa;k++)
    { 
      fscanf(fQA_asc,"%s",imgname);
      
      if((fQA[k]=fopen(imgname,"rb"))==NULL)
       {
          printf("Cann't open the file for NBAR QA input file\n");
          exit(1);
      }
 
    }


// Determine if there are missing files from mode
  mode = atoi(argv[3]);
  //printf("Successful!\n");

  
   /**/if((fcube=fopen(argv[4],"wb"))==NULL)
     {
          printf("Missing filename for the output Data Cube\n");
          exit(1);
     }


 // Read LST binary files

if(mode==1){

 if((fLST_asc=fopen(argv[5],"r"))==NULL)
  {
    printf("Cann't open the file for LST input file list \n");
    exit(1);
  }

/**/
 for(k=0;k<nst;k++)
    { 
      fscanf(fLST_asc,"%s",imgname);
      
      if((fLST[k]=fopen(imgname,"rb"))==NULL)
       {
          printf("Cann't open the file for LST input file\n");
          exit(1);
      }
 
    }

}//if(mode==1)



/**/
for(i=0;i<nrows;i++){
    //printf("Successful2!\n");
    fread(BLU,ncols,sizeof(short),fNBAR[0]);
    fread(GRN,ncols,sizeof(short),fNBAR[1]);
    fread(RED,ncols,sizeof(short),fNBAR[2]);
    fread(NIR,ncols,sizeof(short),fNBAR[3]);
    fread(SWIR1,ncols,sizeof(short),fNBAR[4]);
    fread(SWIR2,ncols,sizeof(short),fNBAR[5]);

    fread(PQA,ncols,sizeof(short),fQA[0]);
    fread(RADSAT,ncols,sizeof(char),fQA[1]);
    //fread(AEQA,ncols,sizeof(char),fQA[2]);
 
    if(mode==1){
    fread(LST,ncols,sizeof(short),fLST[0]);
    fread(LST_QA,ncols,sizeof(short),fLST[1]); 
    }
    for(j=0;j<ncols;j++){

    NDVI = NoData;
    NIRv = NoData;
    EVI = NoData;
    EVI2 = NoData;
    SAVI = NoData;
    MSAVI = NoData;
    NDMI = NoData;
    NBR = NoData;
    NBR2 = NoData;
    VI_QA = 0;//NoData


    if((BLU[j]!=NoData)&&(RED[j]!=NoData)&&(NIR[j]!=NoData)&&(SWIR1[j]!=NoData)&&(SWIR2[j]!=NoData)){
        
        Decode_PQA(PQA[j],PQA_bits,16);
        //Decode_PQA(RADSAT[j],RADSAT_bits,8);
        Decode_AEQA(RADSAT[j],RADSAT_bits,8);
       

        if((PQA_bits[0]==1)||(PQA_bits[2]==1)){
           if(PQA_bits[0]==1){
             VI_QA = 10; //fill
           }else{
             VI_QA = 11; //water
           }
        }else{//land

         if((RADSAT[j]==0)||(RADSAT_bits[1]!=1&&RADSAT_bits[3]!=1&&RADSAT_bits[4]!=1&&RADSAT_bits[5]!=1&&RADSAT_bits[7]!=1)){
           if((PQA_bits[1]==1)){
             VI_QA = 20; // best quality land pixels
           }else if((PQA_bits[1]==0)){
             VI_QA = 23; // cloud/snow contaminated land pixels
           }
           NDVI = Calculate_VI(BLU[j],RED[j],NIR[j],SWIR1[j],SWIR2[j],0);
           NIRv = Calculate_VI(BLU[j],RED[j],NIR[j],SWIR1[j],SWIR2[j],1);
           EVI = Calculate_VI(BLU[j],RED[j],NIR[j],SWIR1[j],SWIR2[j],2);
           EVI2 = Calculate_VI(BLU[j],RED[j],NIR[j],SWIR1[j],SWIR2[j],3);
           SAVI = Calculate_VI(BLU[j],RED[j],NIR[j],SWIR1[j],SWIR2[j],4);
           MSAVI = Calculate_VI(BLU[j],RED[j],NIR[j],SWIR1[j],SWIR2[j],5);
           NDMI = Calculate_VI(BLU[j],RED[j],NIR[j],SWIR1[j],SWIR2[j],6);
           NBR = Calculate_VI(BLU[j],RED[j],NIR[j],SWIR1[j],SWIR2[j],7);
           NBR2 = Calculate_VI(BLU[j],RED[j],NIR[j],SWIR1[j],SWIR2[j],8);
         }else{
          
           VI_QA = 24; // sensor saturation

         }//if(RADSAT[j]==0)
         
        }//if((PQA_bits[0]==1)||(PQA_bits[2]==1))

    }// if((RED[j]!=NoData)&&(NIR[j]!=NoData)&&(SWIR1[j]!=NoData)&&(SWIR2[j]!=NoData))

       fwrite(&NDVI,1,sizeof(short),fcube);
       fwrite(&NIRv,1,sizeof(short),fcube);

       fwrite(&EVI,1,sizeof(short),fcube);
       fwrite(&EVI2,1,sizeof(short),fcube);

       fwrite(&SAVI,1,sizeof(short),fcube);
       fwrite(&MSAVI,1,sizeof(short),fcube);

       fwrite(&NDMI,1,sizeof(short),fcube);
       fwrite(&NBR,1,sizeof(short),fcube);
       fwrite(&NBR2,1,sizeof(short),fcube);
      
       fwrite(&VI_QA,1,sizeof(short),fcube);

       if(mode==0){
         LST[j]=NoData;
         LST_QA[j]=NoData;
       }

       fwrite(&LST[j],1,sizeof(short),fcube);
       fwrite(&LST_QA[j],1,sizeof(short),fcube);
      
   }

}//for(i=0;i<nrows;i++)

/**/
for(k=0;k<nb;k++){

     if(mode==1){
     if(k<nst){
         fclose(fLST[k]);
       }
     }
     if(k<nqa){
         fclose(fQA[k]);
       }

      fclose(fNBAR[k]);
   }


 fclose(fcube);
 return(0);
}

void Decode_PQA(unsigned short quo, short *BitArr, short nbits)
{
  int k;
  short rem;

  for(k=0;k<nbits;k++)
      {
        BitArr[k]=0;
      }

  k=0;

  while(quo>0)
       {
         rem = quo % 2;

         BitArr[k]=rem;

         //printf("k=%d quo=%d rem=%d BitArr=%d\n",k,quo,rem,BitArr[k]);

         k++;

         quo = (quo -rem)/2;

       }// while(quo>0)
  
}

void Decode_AEQA(unsigned char quo_char, short *BitArr, short nbits)
{
  int k;
  short rem;
  unsigned short quo;

  quo = (unsigned short)quo_char;

  for(k=0;k<nbits;k++)
      {
        BitArr[k]=0;
      }

  k=0;

  while(quo>0)
       {
         rem = quo % 2;

         BitArr[k]=rem;

         //printf("k=%d quo=%d rem=%d BitArr=%d\n",k,quo,rem,BitArr[k]);

         k++;

         quo = (quo -rem)/2;

       }// while(quo>0)
}


short Calculate_VI(short band1, short band2, short band3, short band4, short band5, short VI_idx){

    short VI;
    float SCF = 10000.0;
    float BLU = (band1*1.0)/SCF;
    float RED = (band2*1.0)/SCF;
    float NIR = (band3*1.0)/SCF;
    float SWIR1 = (band4*1.0)/SCF;
    float SWIR2 = (band5*1.0)/SCF;
    float soil_offset = 0.08;

    if(VI_idx==0){//NDVI
      VI = (short)((NIR - RED)*SCF/(NIR + RED));
    }else if(VI_idx==1){//NIRv
      //VI = (short)((NIR - RED)*NIR*SCF/(NIR + RED));
      VI = (short)((((NIR - RED)/(NIR + RED))-soil_offset)*NIR*SCF);
    }else if(VI_idx==2){//EVI
      VI = (short)((NIR - RED)*2.5*SCF/(NIR + RED*6.0 - BLU*7.5 + 1.0));
    }else if(VI_idx==3){//EVI2
      VI = (short)((NIR - RED)*2.5*SCF/(NIR + RED*2.4 + 1.0));
    }else if(VI_idx==4){//SAVI
      VI = (short)((NIR - RED)*1.5*SCF/(NIR + RED + 0.5));
    }else if(VI_idx==5){//MSAVI
      VI = (short)((2.0*NIR + 1.0 - sqrt((2.0*NIR + 1.0)*(2.0*NIR + 1.0) - 8.0*(NIR - RED)))*SCF/2.0);
    }else if(VI_idx==6){//NDMI
      VI = (short)((NIR - SWIR1)*SCF/(NIR + SWIR1));
    }else if(VI_idx==7){//NBR
      VI = (short)((NIR - SWIR2)*SCF/(NIR + SWIR2));
    }else if(VI_idx==8){//NBR2
      VI = (short)((SWIR1 - SWIR2)*SCF/(SWIR1 + SWIR2));
    }

return(VI);
}



