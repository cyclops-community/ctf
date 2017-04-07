#include <ctf.hpp>
#include <stdio.h>
using namespace CTF;

static void trashCache(float* trash1, float* trash2, int nTotal){
   for(int i = 0; i < nTotal; i ++) 
      trash1[i] += 0.99 * trash2[i];
}

int equal_(const double*A, const double*B, int total_size){
  int error = 0;
   const double*Atmp= A;
   const double*Btmp= B;
   for(int i=0;i < total_size ; ++i){
      if( Atmp[i] != Atmp[i] || Btmp[i] != Btmp[i]  || isinf(Atmp[i]) || isinf(Btmp[i]) ){
         error += 1; //test for NaN or Inf
         continue;
      }
      double Aabs = (Atmp[i] < 0) ? -Atmp[i] : Atmp[i];
      double Babs = (Btmp[i] < 0) ? -Btmp[i] : Btmp[i];
      double max = (Aabs < Babs) ? Babs : Aabs;
      double diff = (Aabs - Babs);
      diff = (diff < 0) ? -diff : diff;
      if(diff > 0){
         double relError = (diff / max);
         if(relError > 4e-5){
//            printf("%.3e %.3e\n",relError, max);
            error += 1;
         }
      }
   }
   return (error == 0) ? 1 : 0;
}

void example(int argc, char** argv){
  

  World dw(MPI_COMM_WORLD, argc, argv);
  int dimA = 4;
  int shapeA[dimA];
  int sizeA[] = {24,16,24,16};
  for( int i = 0; i < dimA; i++ )
    shapeA[i] = NS;
  int dimB = 4;
  int shapeB[dimB];
  int sizeB[] = {24,16,24,16};
  for( int i = 0; i < dimB; i++ )
    shapeB[i] = NS;
  int dimC = 6;
  int shapeC[dimC];
  int sizeC[] = {24,16,16,24,16,16};
  for( int i = 0; i < dimC; i++ )
   shapeC[i] = NS;

  float *trash1, *trash2;
  int nTotal = 1024*1024*100;
  trash1 = (float*) malloc(sizeof(float)*nTotal);
  trash2 = (float*) malloc(sizeof(float)*nTotal);
  //* Creates distributed tensors initialized with zeros
  Tensor<double> A(dimA, sizeA, shapeA, dw);
  Tensor<double> B(dimB, sizeB, shapeB, dw);
  Tensor<double> C(dimC, sizeC, shapeC, dw);

  int64_t sizeAtotal;
  double* Adata = A.get_raw_data(&sizeAtotal);
  double *Acopy = (double*) malloc(sizeof(double)*sizeAtotal);
  int64_t sizeBtotal;
  double* Bdata = B.get_raw_data(&sizeBtotal);
  double *Bcopy = (double*) malloc(sizeof(double)*sizeBtotal);
  int64_t sizeCtotal;
  double* Cdata = C.get_raw_data(&sizeCtotal);
  double *Ccopy = (double*) malloc(sizeof(double)*sizeCtotal);
  for(int64_t i = 0; i < sizeAtotal; i++){
     Adata[i] = (((i+1)*13 % 1000) - 500.) / 1000.;
     Acopy[i] = (((i+1)*13 % 1000) - 500.) / 1000.;
  }
  for(int64_t i = 0; i < sizeBtotal; i++){
     Bdata[i] = (((i+1)*7 % 1000) - 500.) / 1000.;
     Bcopy[i] = (((i+1)*7 % 1000) - 500.) / 1000.;
  }
  for(int64_t i = 0; i < sizeCtotal; i++){
     Cdata[i] = (((i+1)*17 % 1000) - 500.) / 1000.;
     Ccopy[i] = (((i+1)*17 % 1000) - 500.) / 1000.;
  }


  double minTime = 1e100;
  for (int i=0; i<3; i++){
     trashCache(trash1, trash2, nTotal);
     double t = MPI_Wtime();
     C["abcdef"] = A["dfgb"]*B["geac"];
     t = MPI_Wtime() - t;
     minTime = (minTime < t) ? minTime : t;
  }

  Adata = A.get_raw_data(&sizeAtotal);
  Bdata = B.get_raw_data(&sizeBtotal);
  Cdata = C.get_raw_data(&sizeCtotal);

  if( !equal_(Adata, Acopy, sizeAtotal) )
     fprintf(stderr, "ERROR: A has changed\n");
  if( !equal_(Bdata, Bcopy, sizeBtotal) )
     fprintf(stderr, "ERROR: B has changed\n");

  double flops = 2.E-9 * 905969664.000000; 
  printf("abcdef-dfgb-geac %.2lf seconds/GEMM, %.2lf GF\n",minTime, flops/minTime);
 
  free(trash1);
  free(trash2);
} 

int main(int argc, char ** argv){

  MPI_Init(&argc, &argv);

  example(argc, argv);

  MPI_Finalize();
  return 0;
}
