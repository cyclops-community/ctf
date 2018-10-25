/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "common.h"
#include "world.h"
#include "../shared/blas_symbs.h"
#include "../shared/lapack_symbs.h"
#include <stdlib.h>


namespace CTF_int{
  struct int2
  {
    int i[2];
    int2(int a, int b)
    {
      i[0] = a;
      i[1] = b;
    }
    operator const int*() const
    {
      return i;
    }
  };
}


namespace CTF {
  template<typename dtype>
  Matrix<dtype>::Matrix() : Tensor<dtype>() { 
    nrow = -1;
    ncol = -1;
  }

  template<typename dtype>
  Matrix<dtype>::Matrix(Matrix<dtype> const & A)
    : Tensor<dtype>(A) {
    nrow = A.nrow;
    ncol = A.ncol;
    symm = A.symm;
  }

  template<typename dtype>
  Matrix<dtype>::Matrix(Tensor<dtype> const & A)
    : Tensor<dtype>(A) {
    assert(A.order == 2);
    nrow = A.lens[0];
    ncol = A.lens[1];
    switch (A.sym[0]){
      case NS:
        symm=NS;
        break;
      case SY:
        symm=SY;
        break;
      case AS:
        symm=AS;
        break;
      default:
        IASSERT(0);
        break;
    }
  }


  template<typename dtype>
  Matrix<dtype>::Matrix(int                       nrow_,
                        int                       ncol_,
                        World &                   world_,
                        CTF_int::algstrct const & sr_,
                        char const *              name_,
                        int                       profile_)
    : Tensor<dtype>(2, false, CTF_int::int2(nrow_, ncol_),  CTF_int::int2(NS, NS),
                           world_, sr_, name_, profile_) {
    nrow = nrow_;
    ncol = ncol_;
    symm = NS;
  }

  template<typename dtype>
  Matrix<dtype>::Matrix(int                       nrow_,
                        int                       ncol_,
                        int                       atr_,
                        World &                   world_,
                        CTF_int::algstrct const & sr_,
                        char const *              name_,
                        int                       profile_)
    : Tensor<dtype>(2, (atr_&4)>0, CTF_int::int2(nrow_, ncol_), CTF_int::int2(atr_&3, NS), 
                           world_, sr_, name_, profile_) {
    nrow = nrow_;
    ncol = ncol_;
    symm = atr_&3;
  }

  template<typename dtype>
  Matrix<dtype>::Matrix(int                       nrow_,
                        int                       ncol_,
                        World &                   world_,
                        char const *              name_,
                        int                       profile_,
                        CTF_int::algstrct const & sr_)
    : Tensor<dtype>(2, false, CTF_int::int2(nrow_, ncol_), CTF_int::int2(NS, NS),
                           world_, sr_, name_, profile_) {
    nrow = nrow_;
    ncol = ncol_;
    symm = 0;
  }


  template<typename dtype>
  Matrix<dtype>::Matrix(int                       nrow_,
                        int                       ncol_,
                        int                       atr_,
                        World &                   world_,
                        char const *              name_,
                        int                       profile_,
                        CTF_int::algstrct const & sr_)
    : Tensor<dtype>(2, (atr_&4)>0, CTF_int::int2(nrow_, ncol_), CTF_int::int2(atr_&3, NS), 
                           world_, sr_, name_, profile_) {
    nrow = nrow_;
    ncol = ncol_;
    symm = atr_&3;
  }


  template<typename dtype>
  Matrix<dtype>::Matrix(int                       nrow_,
                        int                       ncol_,
                        char const *              idx,
                        Idx_Partition const &     prl,
                        Idx_Partition const &     blk,
                        int                       atr_,
                        World &                   world_,
                        CTF_int::algstrct const & sr_,
                        char const *              name_,
                        int                       profile_)
    : Tensor<dtype>(2, (atr_&4)>0, CTF_int::int2(nrow_, ncol_), CTF_int::int2(atr_&3, NS), 
                           world_, idx, prl, blk, name_, profile_, sr_) {
    nrow = nrow_;
    ncol = ncol_;
    symm = atr_&3;
  }

  template<typename dtype>
  void Matrix<dtype>::print_matrix(){
    int64_t nel;
    dtype * data = (dtype*)malloc(sizeof(dtype)*nrow*ncol);
    nel = this->read_all(data,true);
    if (this->wrld->rank == 0){
      for (int i=0; i<nrow; i++){
        for (int j=0; j<ncol; j++){
          this->sr->print((char*)&(data[j*nrow+i]));
          if (j!=ncol-1) printf(" ");
        }
        printf("\n");
      }
    }
    free(data);
  }

  template<typename dtype>
  void get_my_kv_pair(int            rank,
                      int            nrow,
                      int            ncol,
                      int            mb,
                      int            nb,
                      int            pr,
                      int            pc,
                      int            rsrc,
                      int            csrc,
                      int64_t &      nmyr,
                      int64_t &      nmyc,
                      Pair<dtype> *& pairs){
    nmyr = mb*(nrow/mb/pr);
    if ((nrow/mb)%pr > (rank+pr-rsrc)%pr){
      nmyr+=mb;
    }
    if (((nrow/mb)%pr) == (rank+pr-rsrc)%pr){
      nmyr+=nrow%mb;
    }
    nmyc = nb*(ncol/nb/pc);
    if ((ncol/nb)%pc > (rank/pr+pc-csrc)%pc){
      nmyc+=nb;
    }
    if (((ncol/nb)%pc) == (rank/pr+pc-csrc)%pc){
      nmyc+=ncol%nb;
    }
    //printf("nrow = %d ncol = %d nmyr = %ld, nmyc = %ld mb = %d nb = %d pr = %d pc = %d\n",nrow,ncol,nmyr,nmyc,mb,nb,pr,pc);
    pairs = new Pair<dtype>[nmyr*nmyc];
    int cblk = (rank/pr+pc-csrc)%pc;
    for (int64_t i=0; i<nmyc;  i++){
      int rblk = (rank+pr-rsrc)%pr;
      for (int64_t j=0; j<nmyr;  j++){
        pairs[i*nmyr+j].k = (cblk*nb+(i%nb))*nrow+rblk*mb+(j%mb);
    //    pairs[i*nmyr+j].d = *(dtype*)this->sr->addid();
        //printf("RANK = %d, pairs[%ld].k=%ld\m",rank,i*nmyr+j,pairs[i*nmyr+j].k);
        if ((j+1)%mb == 0) rblk += pr;
      }
      if ((i+1)%nb == 0) cblk += pc;
    }
  }

  template<typename dtype>
  void Matrix<dtype>::write_mat(int           mb,
                                int           nb,
                                int           pr,
                                int           pc,
                                int           rsrc,
                                int           csrc,
                                int           lda,
                                dtype const * data_){
    if (mb==1 && nb==1 && nrow%pr==0 && ncol%pc==0 && rsrc==0 && csrc==0){
      if (this->edge_map[0].np == pr && this->edge_map[1].np == pc){
        if (lda == nrow/pc){
          memcpy(this->data, (char*)data_, sizeof(dtype)*this->size);
        } else {
          for (int i=0; i<ncol/pc; i++){
            memcpy(this->data+i*lda*sizeof(dtype),(char*)(data_+i*lda), nrow*sizeof(dtype)/pr);
          }
        }
      } else {
        Matrix<dtype> M(nrow, ncol, mb, nb, pr, pc, rsrc, csrc, lda, data_);
        (*this)["ab"] = M["ab"];
      }
    } else {
      Pair<dtype> * pairs;
      int64_t nmyr, nmyc;
      get_my_kv_pair(this->wrld->rank, nrow, ncol, mb, nb, pr, pc, rsrc, csrc, nmyr, nmyc, pairs);

        //printf("lda = %d, nmyr =%ld, nmyc=%ld\n",lda,nmyr,nmyc);
      if (lda == nmyr){
        for (int64_t i=0; i<nmyr*nmyc; i++){
          pairs[i].d = data_[i];
        }
      } else {
        for (int64_t i=0; i<nmyc; i++){
          for (int64_t j=0; j<nmyr; j++){
            pairs[i*nmyr+j].d = data_[i*lda+j];
          }
        }
      }
      this->write(nmyr*nmyc, pairs);
      delete [] pairs;
    }
  }

  template<typename dtype>
  void Matrix<dtype>::read_mat(int     mb,
                               int     nb,
                               int     pr,
                               int     pc,
                               int     rsrc,
                               int     csrc,
                               int     lda,
                               dtype * data_){
    //FIXME: (1) can optimize sparse for this case (mapping cyclic), (2) can use permute to avoid sparse redistribution always
    if (!this->is_sparse && (mb==1 && nb==1 && nrow%pr==0 && ncol%pc==0 && rsrc==0 && csrc==0)){
      if (this->edge_map[0].np == pr && this->edge_map[1].np == pc){
        if (lda == nrow/pc){
          memcpy((char*)data_, this->data, sizeof(dtype)*this->size);
        } else {
          for (int i=0; i<ncol/pc; i++){
            memcpy((char*)(data_+i*lda), this->data+i*lda*sizeof(dtype), nrow*sizeof(dtype)/pr);
          }
        }
      } else {
        int plens[] = {pr, pc};
        Partition ip(2, plens);
        Matrix M(nrow, ncol, "ij", ip["ij"], Idx_Partition(), 0, *this->wrld, *this->sr);
        M["ab"] = (*this)["ab"];
        M.read_mat(mb, nb, pr, pc, rsrc, csrc, lda, data_);
      }
    } else {
      Pair<dtype> * pairs;
      int64_t nmyr, nmyc;
      get_my_kv_pair(this->wrld->rank, nrow, ncol, mb, nb, pr, pc, rsrc, csrc, nmyr, nmyc, pairs);

      this->read(nmyr*nmyc, pairs);
      if (lda == nmyr){
        for (int64_t i=0; i<nmyr*nmyc; i++){
          data_[i] = pairs[i].d;
          //printf("data %ld = %lf\n",i,data_[i]);
        }
      } else {
        for (int64_t i=0; i<nmyc; i++){
          for (int64_t j=0; j<nmyr; j++){
            data_[i*lda+j] = pairs[i*nmyr+j].d;
            //printf("data %ld %ld = %lf\n",i,j,data_[i*lda+j]);
          }
        }
      }
      delete [] pairs;
    }
  }

  template<typename dtype>
  void Matrix<dtype>::get_desc(int & ictxt, int *& desc){
    int pr, pc;
    pr = this->edge_map[0].calc_phase();       
    pc = this->edge_map[1].calc_phase();       

    char C = 'C';
    int ctxt;
    IASSERT(this->wrld->comm == MPI_COMM_WORLD);
    CTF_SCALAPACK::cblacs_get(-1, 0, &ctxt);
    CTF_int::grid_wrapper gw;
    gw.pr = pr;
    gw.pc = pc;
    std::set<CTF_int::grid_wrapper>::iterator s = CTF_int::scalapack_grids.find(gw);
    if (s != CTF_int::scalapack_grids.end()){
      ctxt = s->ctxt;
    } else {
      CTF_SCALAPACK::cblacs_gridinit(&ctxt, &C, pr, pc);
      gw.ctxt = ctxt;
      CTF_int::scalapack_grids.insert(gw);
    }
    ictxt = ctxt;

    desc = (int*)malloc(sizeof(int)*9);
    desc[0] = 1;
    desc[1] = ictxt;
    desc[2] = nrow;
    desc[3] = ncol;
    desc[4] = 1;
    desc[5] = 1;
    desc[6] = 0;
    desc[7] = 0;
    desc[8] = this->pad_edge_len[0]/pr;
  }

  template<typename dtype>
  void Matrix<dtype>::read_mat(int const * desc,
                               dtype *     data_){
    int ictxt = desc[1];
    int pr, pc, ipr, ipc;
    CTF_SCALAPACK::cblacs_gridinfo(ictxt, &pr, &pc, &ipr, &ipc);
    IASSERT(ipr == this->wrld->rank%pr);
    IASSERT(ipc == this->wrld->rank/pr);

    read_mat(desc[4],desc[5],pr,pc,desc[6],desc[7],desc[8],data_);
  }

  template<typename dtype>
  Matrix<dtype>::Matrix(int                       nrow_,
                        int                       ncol_,
                        int                       mb,
                        int                       nb,
                        int                       pr,
                        int                       pc,
                        int                       rsrc,
                        int                       csrc,
                        int                       lda,
                        dtype const *             data,
                        World &                   wrld_,
                        CTF_int::algstrct const & sr_,
                        char const *              name_,
                        int                       profile_)
    : Tensor<dtype>(2, false, CTF_int::int2(nrow_, ncol_),  CTF_int::int2(NS, NS),
                           wrld_, sr_, name_, profile_) {
    nrow = nrow_;
    ncol = ncol_;
    symm = NS;
    write_mat(mb,nb,pr,pc,rsrc,csrc,lda,data);
  }

  

  static inline Idx_Partition get_map_from_desc(int const * desc){

    int ictxt = desc[1];
    int pr, pc, ipr, ipc;
    CTF_SCALAPACK::cblacs_gridinfo(ictxt, &pr, &pc, &ipr, &ipc);
    return Partition(2,CTF_int::int2(pr, pc))["ij"];
  }

  template<typename dtype>
  Matrix<dtype>::Matrix(int const *               desc,
                        dtype const *             data_,
                        World &                   wrld_,
                        CTF_int::algstrct const & sr_,
                        char const *              name_,
                        int                       profile_)
    : Tensor<dtype>(2, false, CTF_int::int2(desc[2], desc[3]),  CTF_int::int2(NS, NS),
                           wrld_, "ij", get_map_from_desc(desc), Idx_Partition(), name_, profile_, sr_) {
    nrow = desc[2];
    ncol = desc[3];
    symm = NS;
    int ictxt = desc[1];
    int pr, pc, ipr, ipc;
    CTF_SCALAPACK::cblacs_gridinfo(ictxt, &pr, &pc, &ipr, &ipc);
    IASSERT(ipr == wrld_.rank%pr);
    IASSERT(ipc == wrld_.rank/pr);
    IASSERT(pr*pc == wrld_.np);
    //this->set_distribution("ij", Partition(2,CTF_int::int2(pr, pc))["ij"], Idx_Partition());
    write_mat(desc[4],desc[5],pr,pc,desc[6],desc[7],desc[8],data_);
  }

  template <typename dtype>
  int get_int_fromreal(dtype r){
    assert(0);
    return -1;
  }

  template <>
  inline int get_int_fromreal<float>(float r){
    return (int)r;
  }
  template <>
  inline int get_int_fromreal<double>(double r){
    return (int)r;
  }
  template <>
  inline int get_int_fromreal<std::complex<float>>(std::complex<float> r){
    return (int)r.real();
  }
  template <>
  inline int get_int_fromreal<std::complex<double>>(std::complex<double> r){
    return (int)r.real();
  }




  template<typename dtype>
  void Matrix<dtype>::qr(Matrix<dtype> & Q, Matrix<dtype> & R){

    int info;

    int m = this->nrow;
    int n = this->ncol;

    int * desca;// = (int*)malloc(9*sizeof(int));

    int ictxt;
    this->get_desc(ictxt, desca);
    dtype * A = (dtype*)malloc(this->size*sizeof(dtype));

    this->read_mat(desca, A);

    dtype * tau = (dtype*)malloc(n*sizeof(dtype));
    dtype dlwork;
    CTF_SCALAPACK::pgeqrf<dtype>(m,n,A,1,1,desca,tau,(dtype*)&dlwork,-1,&info);
    int lwork = get_int_fromreal<dtype>(dlwork);
    dtype * work = (dtype*)malloc(lwork*sizeof(dtype));
    CTF_SCALAPACK::pgeqrf<dtype>(m,n,A,1,1,desca,tau,work,lwork,&info);
 

    dtype * dQ = (dtype*)malloc(this->size*sizeof(dtype));
    memcpy(dQ,A,this->size*sizeof(dtype));

    Q = Matrix<dtype>(desca, dQ, (*(this->wrld)));
    if (m==n)
      R = Matrix<dtype>(Q);
    else {
      R = Matrix<dtype>(desca,dQ,*this->wrld,*this->sr);
      R = R.slice(0,m*(n-1)+n-1);
    }


    free(work);
    CTF_SCALAPACK::porgqr<dtype>(m,n,n,dQ,1,1,desca,tau,(dtype*)&dlwork,-1,&info);
    lwork = get_int_fromreal<dtype>(dlwork);
    work = (dtype*)malloc(lwork*sizeof(dtype));
    CTF_SCALAPACK::porgqr<dtype>(m,n,n,dQ,1,1,desca,tau,work,lwork,&info);
    Q = Matrix<dtype>(desca, dQ, (*(this->wrld)));
    free(work);
    free(tau);
    free(desca);
    //make upper-tri
    int syns[] = {SY, NS};
    Tensor<dtype> tR(R,syns);
    int nsns[] = {NS, NS};
    tR = Tensor<dtype>(tR,nsns);
    R = CTF::Matrix<dtype>(tR);
    //R["ij"] = R["ji"];
    free(A);
    free(dQ);
  }

  template<typename dtype>
  void Matrix<dtype>::svd(Matrix<dtype> & U, Vector<dtype> & S, Matrix<dtype> & VT,  int rank){

    int info;

    int m = this->nrow;
    int n = this->ncol;
    int k = std::min(m,n);


    int * desca;// = (int*)malloc(9*sizeof(int));
    int * descu = (int*)malloc(9*sizeof(int));
    int * descvt = (int*)malloc(9*sizeof(int));

    int ictxt;
    this->get_desc(ictxt, desca);

    int pr, pc;
    pr = this->edge_map[0].calc_phase();       
    pc = this->edge_map[1].calc_phase();       
    //CTF_SCALAPACK::cdescinit(desca, m, n, 1, 1, 0, 0, ictxt, m/(*(this->wrld)).np, &info);
    int64_t mpr = m/pr + (m % pr != 0);
    int64_t kpr = k/pr + (k % pr != 0);
    int64_t kpc = k/pc + (k % pc != 0);
    int64_t npc = n/pc + (n % pc != 0);
    CTF_SCALAPACK::cdescinit(descu, m, k, 1, 1, 0, 0, ictxt, mpr, &info);
    CTF_SCALAPACK::cdescinit(descvt, k, n, 1, 1, 0, 0, ictxt, kpr, &info);
    dtype * A = (dtype*)malloc(this->size*sizeof(dtype));


    dtype * u = (dtype*)new dtype[mpr*kpc];
    dtype * s = (dtype*)new dtype[k];
    dtype * vt = (dtype*)new dtype[kpr*npc];
    this->read_mat(desca, A);

    int lwork;
    dtype dlwork;
    CTF_SCALAPACK::pgesvd<dtype>('V', 'V', m, n, NULL, 1, 1, desca, NULL, NULL, 1, 1, descu, vt, 1, 1, descvt, &dlwork, -1, &info);  

    lwork = get_int_fromreal<dtype>(dlwork);
    dtype * work = (dtype*)malloc(sizeof(dtype)*lwork);

    CTF_SCALAPACK::pgesvd<dtype>('V', 'V', m, n, A, 1, 1, desca, s, u, 1, 1, descu, vt, 1, 1, descvt, work, lwork, &info);	

 
    U = Matrix<dtype>(descu, u, (*(this->wrld)));
    VT = Matrix<dtype>(descvt, vt, (*(this->wrld)));

    S = Vector<dtype>(k, (*(this->wrld)));
    int64_t sc;
    dtype * s_data = S.get_raw_data(&sc);

    int phase = S.edge_map[0].calc_phase();
    if ((int)((this->wrld->rank) < phase){
      for (int i = S.edge_map[0].calc_phys_rank(S.topo); i < k; i += phase) {
        s_data[i/phase] = s[i];
      } 
    }
    if (rank > 0 && rank < k) {
      S = S.slice(0, rank-1);
      U = U.slice(0, rank*(m)-1);
      VT = VT.slice(0, k*n-(k-rank+1));
    }

    free(A);
    delete [] u;
    delete [] s;
    delete [] vt;
    free(desca);
    free(descu);
    free(descvt);
    free(work);

  }
  

}
