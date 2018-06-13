/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

/** \addtogroup examples 
  * @{ 
  * \defgroup CCSD CCSD
  * @{ 
  * \brief A Coupled Cluster Singles and Doubles contraction code extracted from Aquarius
  */

#include <ctf.hpp>
using namespace CTF;

double divide(double a, double b){
  return a/b;
}


class Integrals {
  public:
  World * dw;
  Tensor<> * aa;
  Tensor<> * ii;
  Tensor<> * ab;
  Tensor<> * ai;
  Tensor<> * ia;
  Tensor<> * ij;
  Tensor<> * abcd;
  Tensor<> * abci;
  Tensor<> * aibc;
  Tensor<> * aibj;
  Tensor<> * abij;
  Tensor<> * ijab;
  Tensor<> * aijk;
  Tensor<> * ijak;
  Tensor<> * ijkl;

  Integrals(int no, int nv, World &dw_){
    int shapeASAS[] = {AS,NS,AS,NS};
    int shapeASNS[] = {AS,NS,NS,NS};
    int shapeNSNS[] = {NS,NS,NS,NS};
    int shapeNSAS[] = {NS,NS,AS,NS};
    int vvvv[]      = {nv,nv,nv,nv};
    int vvvo[]      = {nv,nv,nv,no};
    int vovv[]      = {nv,no,nv,nv};
    int vovo[]      = {nv,no,nv,no};
    int vvoo[]      = {nv,nv,no,no};
    int oovv[]      = {no,no,nv,nv};
    int vooo[]      = {nv,no,no,no};
    int oovo[]      = {no,no,nv,no};
    int oooo[]      = {no,no,no,no};
    
    dw = &dw_;
    
    aa = new CTF_Vector(nv,dw_);
    ii = new CTF_Vector(no,dw_);
    
    ab = new CTF_Matrix(nv,nv,AS,dw_,"Vab",1);
    ai = new CTF_Matrix(nv,no,NS,dw_,"Vai",1);
    ia = new CTF_Matrix(no,nv,NS,dw_,"Via",1);
    ij = new CTF_Matrix(no,no,AS,dw_,"Vij",1);

    abcd = new Tensor<>(4,vvvv,shapeASAS,dw_,"Vabcd",1);
    abci = new Tensor<>(4,vvvo,shapeASNS,dw_,"Vabci",1);
    aibc = new Tensor<>(4,vovv,shapeNSAS,dw_,"Vaibc",1);
    aibj = new Tensor<>(4,vovo,shapeNSNS,dw_,"Vaibj",1);
    abij = new Tensor<>(4,vvoo,shapeASAS,dw_,"Vabij",1);
    ijab = new Tensor<>(4,oovv,shapeASAS,dw_,"Vijab",1);
    aijk = new Tensor<>(4,vooo,shapeNSAS,dw_,"Vaijk",1);
    ijak = new Tensor<>(4,oovo,shapeASNS,dw_,"Vijak",1);
    ijkl = new Tensor<>(4,oooo,shapeASAS,dw_,"Vijkl",1);
  }

  ~Integrals(){
    delete aa;
    delete ii;
    
    delete ab;
    delete ai;
    delete ia;
    delete ij;
    
    delete abcd;
    delete abci;
    delete aibc;
    delete aibj;
    delete abij;
    delete ijab;
    delete aijk;
    delete ijak;
    delete ijkl;
  }

  void fill_rand(){
    int i, rank;
    int64_t j, sz, * indices;
    double * values;
    
    Tensor<> * tarr[] =  {aa, ii, ab, ai, ia, ij, 
                            abcd, abci, aibc, aibj, 
                            abij, ijab, aijk, ijak, ijkl};
    MPI_Comm comm = dw->comm;
    MPI_Comm_rank(comm, &rank);

    srand48(rank*13);

    for (i=0; i<15; i++){
      tarr[i]->get_local_data(&sz, &indices, &values);
//      for (j=0; j<sz; j++) values[j] = drand48()-.5;
      for (j=0; j<sz; j++) values[j] = ((indices[j]*16+i)%13077)/13077. -.5;
      tarr[i]->write(sz, indices, values);
      free(indices), delete [] values;
    }
  }
  
  Idx_Tensor operator[](char const * idx_map_){
    int i, lenm, no, nv;
    lenm = strlen(idx_map_);
    char new_idx_map[lenm+1];
    new_idx_map[lenm]='\0';
    no = 0;
    nv = 0;
    for (i=0; i<lenm; i++){
      if (idx_map_[i] >= 'a' && idx_map_[i] <= 'h'){
        new_idx_map[i] = 'a'+nv;
        nv++;
      } else if (idx_map_[i] >= 'i' && idx_map_[i] <= 'n'){
        new_idx_map[i] = 'i'+no;
        no++;
      }
    }
//    printf("indices %s are %s\n",idx_map_,new_idx_map);
    if (0 == strcmp("a",new_idx_map)) return (*aa)[idx_map_];
    if (0 == strcmp("i",new_idx_map)) return (*ii)[idx_map_];
    if (0 == strcmp("ab",new_idx_map)) return (*ab)[idx_map_];
    if (0 == strcmp("ai",new_idx_map)) return (*ai)[idx_map_];
    if (0 == strcmp("ia",new_idx_map)) return (*ia)[idx_map_];
    if (0 == strcmp("ij",new_idx_map)) return (*ij)[idx_map_];
    if (0 == strcmp("abcd",new_idx_map)) return (*abcd)[idx_map_];
    if (0 == strcmp("abci",new_idx_map)) return (*abci)[idx_map_];
    if (0 == strcmp("aibc",new_idx_map)) return (*aibc)[idx_map_];
    if (0 == strcmp("aibj",new_idx_map)) return (*aibj)[idx_map_];
    if (0 == strcmp("abij",new_idx_map)) return (*abij)[idx_map_];
    if (0 == strcmp("ijab",new_idx_map)) return (*ijab)[idx_map_];
    if (0 == strcmp("aijk",new_idx_map)) return (*aijk)[idx_map_];
    if (0 == strcmp("ijak",new_idx_map)) return (*ijak)[idx_map_];
    if (0 == strcmp("ijkl",new_idx_map)) return (*ijkl)[idx_map_];
    printf("Invalid integral indices\n");
    assert(0);
//shut up compiler
    return (*aa)[idx_map_];
  }
};

class Amplitudes {
  public:
  Tensor<> * ai;
  Tensor<> * abij;
  World * dw;

  Amplitudes(int no, int nv, World &dw_){
    dw = &dw_;
    int shapeASAS[] = {AS,NS,AS,NS};
    int vvoo[]      = {nv,nv,no,no};

    ai = new CTF_Matrix(nv,no,NS,dw_,"Tai",1);
    abij = new Tensor<>(4,vvoo,shapeASAS,dw_,"Tabij",1);
  }

  ~Amplitudes(){
    delete ai;
    delete abij;
  }

  Idx_Tensor operator[](char const * idx_map_){
    if (strlen(idx_map_) == 4) return (*abij)[idx_map_];
    else return (*ai)[idx_map_];
  }
  
  void fill_rand(){
    int i, rank;
    int64_t j, sz, * indices;
    double * values;
    
    Tensor<> * tarr[] =  {ai, abij};
    MPI_Comm comm = dw->comm;
    MPI_Comm_rank(comm, &rank);

    srand48(rank*25);

    for (i=0; i<2; i++){
      tarr[i]->get_local_data(&sz, &indices, &values);
//      for (j=0; j<sz; j++) values[j] = drand48()-.5;
      for (j=0; j<sz; j++) values[j] = ((indices[j]*13+i)%13077)/13077. -.5;
      tarr[i]->write(sz, indices, values);
      free(indices), delete [] values;
    }
  }
};

void ccsd(Integrals   &V,
          Amplitudes  &T,
          int sched_nparts = 0){
  int rank;   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#ifdef SCHEDULE_CCSD
  double timer = MPI_Wtime();
  tCTF_Schedule<double> sched(V.dw);
  sched.set_max_partitions(sched_nparts);
  sched.record();
#endif

  Tensor<> T21 = Tensor<>(T.abij);
  T21["abij"] += .5*T["ai"]*T["bj"];

  Tensor<> tFme(*V["me"].parent);
  Idx_Tensor Fme(&tFme,"me");
  Fme += V["me"];
  Fme += V["mnef"]*T["fn"];
  
  Tensor<> tFae(*V["ae"].parent);
  Idx_Tensor Fae(&tFae,"ae");
  Fae += V["ae"];
  Fae -= Fme*T["am"];
  Fae -=.5*V["mnef"]*T["afmn"];
  Fae += V["anef"]*T["fn"];

  Tensor<> tFmi(*V["mi"].parent);
  Idx_Tensor Fmi(&tFmi,"mi");
  Fmi += V["mi"];
  Fmi += Fme*T["ei"];
  Fmi += .5*V["mnef"]*T["efin"];
  Fmi += V["mnfi"]*T["fn"];

  Tensor<> tWmnei(*V["mnei"].parent);
  Idx_Tensor Wmnei(&tWmnei,"mnei");
  Wmnei += V["mnei"];
  Wmnei += V["mnei"];
  Wmnei += V["mnef"]*T["fi"];
  
  Tensor<> tWmnij(*V["mnij"].parent);
  Idx_Tensor Wmnij(&tWmnij,"mnij");
  Wmnij += V["mnij"];
  Wmnij -= V["mnei"]*T["ej"];
  Wmnij += V["mnef"]*T21["efij"];

  Tensor<> tWamei(*V["amei"].parent);
  Idx_Tensor Wamei(&tWamei,"amei");
  Wamei += V["amei"];
  Wamei -= Wmnei*T["an"];
  Wamei += V["amef"]*T["fi"];
  Wamei += .5*V["mnef"]*T["afin"];
  
  Tensor<> tWamij(*V["amij"].parent);
  Idx_Tensor Wamij(&tWamij,"amij");
  Wamij += V["amij"];
  Wamij += V["amei"]*T["ej"];
  Wamij += V["amef"]*T["efij"];

  Tensor<> tZai(*V["ai"].parent);
  Idx_Tensor Zai(&tZai,"ai");
  Zai += V["ai"];
  Zai -= Fmi*T["am"]; 
  Zai += V["ae"]*T["ei"]; 
  Zai += V["amei"]*T["em"];
  Zai += V["aeim"]*Fme;
  Zai += .5*V["amef"]*T21["efim"];
  Zai -= .5*Wmnei*T21["eamn"];
  
  Tensor<> tZabij(*V["abij"].parent);
  Idx_Tensor Zabij(&tZabij,"abij");
  Zabij += V["abij"];
  Zabij += V["abei"]*T["ej"];
  Zabij += Wamei*T["ebmj"];
  Zabij -= Wamij*T["bm"]; 
  Zabij += Fae*T["ebij"];
  Zabij -= Fmi*T["abmj"];
  Zabij += .5*V["abef"]*T21["efij"];
  Zabij += .5*Wmnij*T21["abmn"];
 
#ifdef SCHEDULE_CCSD 
  if (rank == 0) {
    printf("Record: %lf\n",
            MPI_Wtime()-timer);
  }

  timer = MPI_Wtime();
  tCTF_ScheduleTimer schedule_time = sched.execute();
#endif


  Tensor<> Dai(2, V.ai->lens, V.ai->sym, *V.dw);
  int sh_sym[4] = {SH, NS, SH, NS};
  Tensor<> Dabij(4, V.abij->lens, sh_sym, *V.dw);
  Dai["ai"] += V["i"];
  Dai["ai"] -= V["a"];
 
  Dabij["abij"] += V["i"];
  Dabij["abij"] += V["j"];
  Dabij["abij"] -= V["a"];
  Dabij["abij"] -= V["b"];


  Function<> fctr(&divide);

  T.ai->contract(1.0, *(Zai.parent), "ai", Dai, "ai", 0.0, "ai", fctr);
  T.abij->contract(1.0, *(Zabij.parent), "abij", Dabij, "abij", 0.0, "abij", fctr);
#ifdef SCHEDULE_CCSD
  if (rank == 0) {
    printf("Schedule comm down: %lf\n", schedule_time.comm_down_time);
    printf("Schedule execute: %lf\n", schedule_time.exec_time);
    printf("Schedule imbalance, wall: %lf\n", schedule_time.imbalance_wall_time);
    printf("Schedule imbalance, accum: %lf\n", schedule_time.imbalance_acuum_time);
    printf("Schedule comm up: %lf\n", schedule_time.comm_up_time);
    printf("Schedule total: %lf\n", schedule_time.total_time);
    printf("All execute: %lf\n",
            MPI_Wtime()-timer);
  }
#endif
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
  int rank, np, niter, no, nv, sched_nparts, i;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-no")){
    no = atoi(getCmdOption(input_str, input_str+in_num, "-no"));
    if (no < 0) no= 4;
  } else no = 4;
  if (getCmdOption(input_str, input_str+in_num, "-nv")){
    nv = atoi(getCmdOption(input_str, input_str+in_num, "-nv"));
    if (nv < 0) nv = 6;
  } else nv = 6;
  if (getCmdOption(input_str, input_str+in_num, "-niter")){
    niter = atoi(getCmdOption(input_str, input_str+in_num, "-niter"));
    if (niter < 0) niter = 1;
  } else niter = 1;
  if (getCmdOption(input_str, input_str+in_num, "-nparts")){
    sched_nparts = atoi(getCmdOption(input_str, input_str+in_num, "-nparts"));
    if (sched_nparts < 0) sched_nparts = 0;
  } else sched_nparts = 0;

  {
    World dw(argc, argv);
    {
      Integrals V(no, nv, dw);
      V.fill_rand();
      Amplitudes T(no, nv, dw);
      Timer_epoch tccsd("CCSD");
      tccsd.begin();
      for (i=0; i<niter; i++){
        T.fill_rand();
        double d = MPI_Wtime();
        ccsd(V,T,sched_nparts);
        if (rank == 0)
          printf("(%d nodes) Completed %dth CCSD iteration in time = %lf, |T| is %lf\n",
              np, i, MPI_Wtime()-d, T.ai->norm2()+T.abij->norm2());
        else {
          T.ai->norm2();
          T.abij->norm2();
        }
        T["ai"] = (1./T.ai->norm2())*T["ai"];
        T["abij"] = (1./T.abij->norm2())*T["abij"];
      }
      tccsd.end();
    }
  }

  MPI_Finalize();
  return 0;
}
/**
 * @} 
 * @}
 */

#endif

