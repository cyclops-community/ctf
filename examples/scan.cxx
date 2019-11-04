/** \addtogroup examples 
  * @{ 
  * \defgroup scan scan
  * @{ 
  * \brief scan iterative method using gemv and spmv
  */

#include <ctf.hpp>
#include <float.h>
using namespace CTF;

template <typename dtype>
void rec_scan(Tensor<dtype> & V){

  if (V.order == 1){
    Matrix<dtype> W(2, V.lens[0], V.lens[0], *V.wrld, *V.sr);
    dtype mulid = ((dtype*)V.sr->mulid())[0];
    W["ij"], [=](dtype & a){ a=mulid; };
    int ssym[] = {SH, NS};
    int nsym[] = {NS, NS};
    Tensor<dtype> W1(W, ssym);
    Tensor<dtype> W2(W1, nsym);
    V["i"] = W2["ji"]*V["j"];
  } else {
    Tensor<dtype> V2(V.order-1, V.lens, *V.wrld, *V.sr);
    char str[V.order];
    for (int i=0; i<V.order; i++){ str[i] = 'a'+i; }
    V2[str+1] += V[str];
    rec_scan(V2);
    
    Matrix<dtype> W(2, V.lens[V.order-1], V.lens[V.order-1], *V.wrld, *V.sr);
    dtype mulid = ((dtype*)V.sr->mulid())[0];
    W["ij"], [=](dtype & a){ a=mulid; };
    int hsym[] = {SH, NS};
    int nsym[] = {NS, NS};
    Tensor<dtype> W1(W, hsym);
    Tensor<dtype> W2(W1, nsym);
    char str2[V.order];
    memcpy(str2+1, str+1, V.order-1);
    str2[0] = 'a'+V.order;
    char strW[2] = {str2[0],'a'};
    V[str]  = W2[strW]*V[str2];
    V[str] += V2[str+1]; 
  }
}

template<typename dtype>
void scan(Vector<dtype> & v, int logn){
  int64_t np;
  int64_t * inds;
  double * data;

  int lens[logn];
  std::fill(lens, lens+logn, 2);

  // represent vector to scan as 2-by-...-by-2 tensor  
  Tensor<dtype> V(logn, lens, *v.wrld, *v.sr);

  v.get_local_data(&np, &inds, &data);
  V.write(np, inds, data);

  free(inds);
  delete [] data;

  rec_scan(V);

  // put the data from the tensor back into the vector
  V.get_local_data(&np, &inds, &data);
  v.write(np, inds, data);
  
  free(inds);
  delete [] data;
}

int scan_test(int     logn,
              World & dw){

  Vector<> v(1<<logn, dw);

  srand48(dw.rank*27);
  v.fill_random(0.0, 1.0);
  
  double * start_data;

  int64_t nn;
  v.get_all_data(&nn, &start_data);

  scan(v, logn);

  double data[1<<logn];

  v.read_all(data);

  int pass = 1;
  for (int i=1; i<1<<logn; i++){
    if (std::abs(data[i] - start_data[i-1] - data[i-1]) >= 1.E-9*(1<<logn)) pass = 0;
  }
  if (dw.rank == 0){
    if (pass) 
      printf("{ scan via tensor contractions } passed \n");
    else
      printf("{ scan via tensor contractions } failed \n");
  }

  delete [] start_data;
  return pass;
} 


#ifndef TEST_SUITE
char* getCmdOption(char ** begin,
                   char ** end,
                   const   std::string & option){
  char ** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end){
    return *itr;
  }
  return 0;
}


int main(int argc, char ** argv){
  int rank, np, logn, pass;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-logn")){
    logn = atoi(getCmdOption(input_str, input_str+in_num, "-logn"));
    if (logn < 0) logn = 4;
  } else logn = 4;


  {
    World dw(argc, argv);

    if (rank == 0){
      printf("Running scan on dimension %d vector\n",1<<logn);
    }
    pass = scan_test(logn, dw);
    assert(pass);
  }

  MPI_Finalize();
  return 0;
}
/**
 * @} 
 * @}
 */

#endif
