/** Copyright (c) 2014, Edgar Solomonik, all rights reserved.
  * \addtogroup examples 
  * @{ 
  * \addtogroup quantum informatics
  * @{ 
  * \brief high-order nonsymmetric contractions
  */


#include <ctf.hpp>
using namespace CTF;

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
  int rank, np, d, L;
  int const in_num = argc;
  char ** input_str = argv;
  double *pairs, *opairs;
  int64_t *indices, *oindices, npair, onpair;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
    
  World dw(argc, argv);

  if (getCmdOption(input_str, input_str+in_num, "-d")){
    d = atoi(getCmdOption(input_str, input_str+in_num, "-d"));
    if (d < 0) d = 2;
  } else d = 2;


  L = 15;

  {
    Matrix<> h(d*d, d*d, NS, dw);
    h.get_local_data(&npair, &indices, &pairs);
    for (int i=0; i<npair; i++) {
      srand48(indices[i]*23);
      pairs[i] = drand48();
    }
    h.write(npair, indices, pairs);
    delete [] pairs;
    free(indices);

    {
      int size[L/2+1];
      int shape[L/2+1];

      for (int i=0; i<L/2; i++) size[i] = d*d;
      size[L/2] = d;
      for (int i=0; i<L/2+1; i++) shape[i] = NS;
      Tensor<> v_in(L/2+1, size, shape, dw);
    
      double t_io_start = MPI_Wtime();

      v_in.get_local_data(&npair, &indices, &pairs);
      for (int i=0; i<npair; i++) {
        srand48(indices[i]);
        pairs[i] = drand48();
      }
      v_in.write(npair, indices, pairs);
      delete [] pairs;
      free(indices);
      Tensor<> v_out(L/2+1, size, shape, dw);
      
      double t_io_end = MPI_Wtime();

      double t_start = MPI_Wtime();

      Timer_epoch ep("folded contractions set 1");
      ep.begin();

      v_out["acegikmo"] =  h["ax"]*v_in["xcegikmo"];
      v_out["acegikmo"] += h["cx"]*v_in["axegikmo"];
      v_out["acegikmo"] += h["ex"]*v_in["acxgikmo"];

      v_out["acegikmo"] += h["gx"]*v_in["acexikmo"];
      v_out["acegikmo"] += h["ix"]*v_in["acegxkmo"];

      v_out["acegikmo"] += h["kx"]*v_in["acegixmo"];
      v_out["acegikmo"] += h["mx"]*v_in["acegikxo"];

      ep.end();

      double t_end = MPI_Wtime();
      if (rank == 0)
        printf("First set of folded contractions took %lf sec\n", t_end-t_start);

      double t_io_start2 = MPI_Wtime();
      
      v_in.get_local_data(&npair, &indices, &pairs);
      v_out.get_local_data(&onpair, &oindices, &opairs);

      double t_io_end2 = MPI_Wtime();
      if (rank == 0)
        printf("IO for first set of folded contractions took %lf sec\n", t_io_end-t_io_start + (t_io_end2-t_io_start2));
    }

    {
      int size[L/2+1];
      int shape[L/2+1];

      size[0] = d;
      for (int i=0; i<L/2; i++) size[i+1] = d*d;
      for (int i=0; i<L/2+1; i++) shape[i] = NS;
      Tensor<> v_in(L/2+1, size, shape, dw);
      
      double t_io_start = MPI_Wtime();
      v_in.write(npair, indices, pairs);
      delete [] pairs;
      free(indices);
      Tensor<> v_out(L/2+1, size, shape, dw);
      v_out.write(onpair, oindices, opairs);
      double t_io_end = MPI_Wtime();

      double t_start = MPI_Wtime();

      Timer_epoch ep("folded contractions set 1");
      ep.begin();
   
      v_out["abdfhjln"] += h["bx"]*v_in["axdfhjln"];
      v_out["abdfhjln"] += h["dx"]*v_in["abxfhjln"];

      v_out["abdfhjln"] += h["fx"]*v_in["abdxhjln"];
      v_out["abdfhjln"] += h["hx"]*v_in["abdfxjln"];
      v_out["abdfhjln"] += h["jx"]*v_in["abdfhxln"];

      v_out["abdfhjln"] += h["lx"]*v_in["abdfhjxn"];
      v_out["abdfhjln"] += h["nx"]*v_in["abdfhjlx"];
     
      ep.end();

      double t_end = MPI_Wtime();

      double t_norm_start = MPI_Wtime();
      double norm = v_out.norm2();
      double t_norm_end = MPI_Wtime();
      if (rank == 0){
        printf("second set of folded contractions took %lf seconds\n",t_end-t_start);
        printf("IO for second set of folded contractions took %lf sec\n", t_io_end-t_io_start);
        printf("norm of v_out is %lf\n",norm);
        printf("calculating norm of v_out took %lf sec\n", t_norm_end-t_norm_start);
      }
    }
  }

  {
    int size[L];
    int shape[L];

    for (int i=0; i<L; i++) size[i] = d;
    for (int i=0; i<L; i++) shape[i] = NS;
    Tensor<> v_in(L, size, shape, dw);
    Tensor<> h(4, size, shape, dw);
    double t_io_start = MPI_Wtime();
    v_in.get_local_data(&npair, &indices, &pairs);
    for (int i=0; i<npair; i++) {
      srand48(indices[i]);
      pairs[i] = drand48();
    }
    v_in.write(npair, indices, pairs);
    delete [] pairs;
    free(indices);
    h.get_local_data(&npair, &indices, &pairs);
    for (int i=0; i<npair; i++) {
      srand48(indices[i]*23);
      pairs[i] = drand48();
    }
    h.write(npair, indices, pairs);
    delete [] pairs;
    free(indices);
    Tensor<> v_out(L, size, shape, dw);

    double t_io_end = MPI_Wtime();

    double t_start = MPI_Wtime();

    Timer_epoch ep("contractions");
    ep.begin();

    v_out["abcdefghijklmno"] = h["abxy"]*v_in["xycdefghijklmno"];
    v_out["abcdefghijklmno"] += h["bcxy"]*v_in["axydefghijklmno"];
    v_out["abcdefghijklmno"] += h["cdxy"]*v_in["abxyefghijklmno"];
    v_out["abcdefghijklmno"] += h["dexy"]*v_in["abcxyfghijklmno"];
    v_out["abcdefghijklmno"] += h["efxy"]*v_in["abcdxyghijklmno"];

    v_out["abcdefghijklmno"] += h["fgxy"]*v_in["abcdexyhijklmno"];
    v_out["abcdefghijklmno"] += h["ghxy"]*v_in["abcdefxyijklmno"];
    v_out["abcdefghijklmno"] += h["hixy"]*v_in["abcdefgxyjklmno"];
    v_out["abcdefghijklmno"] += h["ijxy"]*v_in["abcdefghxyklmno"];
    v_out["abcdefghijklmno"] += h["jkxy"]*v_in["abcdefghixylmno"];

    v_out["abcdefghijklmno"] += h["klxy"]*v_in["abcdefghijxymno"];
    v_out["abcdefghijklmno"] += h["lmxy"]*v_in["abcdefghijkxyno"];
    v_out["abcdefghijklmno"] += h["mnxy"]*v_in["abcdefghijklxyo"];
    v_out["abcdefghijklmno"] += h["noxy"]*v_in["abcdefghijklmxy"];
    ep.end();

    double t_end = MPI_Wtime();

    double t_norm_start = MPI_Wtime();
    double norm = v_out.norm2();
    double t_norm_end = MPI_Wtime();
    if (rank == 0){
      printf("unfolded contractions took %lf seconds\n", t_end-t_start);
      printf("unfolded IO took %lf seconds\n", t_io_end-t_io_start);
      printf("unfolded norm of v_out is %lf\n", norm);
      printf("calculating norm of v_out took %lf sec\n", t_norm_end-t_norm_start);
    }
  }
  return 0;
}
/**
 * @} 
 * @}
 */
