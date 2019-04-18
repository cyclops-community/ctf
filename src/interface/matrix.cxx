/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "common.h"
#include "world.h"
#include "timer.h"
#include "../shared/blas_symbs.h"
#include "../shared/lapack_symbs.h"
#include <stdlib.h>
#include <algorithm>


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
                      char           layout_order,
                      int            rsrc,
                      int            csrc,
                      int64_t &      nmyr,
                      int64_t &      nmyc,
                      Pair<dtype> *& pairs){
    int ipr = (rank+pr-rsrc)%pr;
    if (layout_order == 'R')
      ipr = (rank/pc+pr-rsrc)%pr;
    int ipc = (rank/pr+pc-csrc)%pc;
    if (layout_order == 'R')
      ipc = (rank+pc-csrc)%pc;

    nmyr = mb*(nrow/mb/pr);
    if ((nrow/mb)%pr > ipr){
      nmyr+=mb;
    }
    if (((nrow/mb)%pr) == ipr){
      nmyr+=nrow%mb;
    }
    nmyc = nb*(ncol/nb/pc);
    if ((ncol/nb)%pc > ipc){
      nmyc+=nb;
    }
    if (((ncol/nb)%pc) == ipc){
      nmyc+=ncol%nb;
    }
    //printf("nrow = %d ncol = %d nmyr = %ld, nmyc = %ld mb = %d nb = %d pr = %d pc = %d\n",nrow,ncol,nmyr,nmyc,mb,nb,pr,pc);
    pairs = new Pair<dtype>[nmyr*nmyc];
    int cblk = ipc;
    for (int64_t i=0; i<nmyc;  i++){
      int rblk = ipr;
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
                                char          layout_order,
                                int           rsrc,
                                int           csrc,
                                int           lda,
                                dtype const * data_){
    bool is_order_same = true;
    if (layout_order == 'C'){
      if ((this->edge_map[0].type != CTF_int::PHYSICAL_MAP || this->edge_map[0].cdt != 0) && pc > 1)
        is_order_same = false;
    } else {
      if ((this->edge_map[1].type != CTF_int::PHYSICAL_MAP || this->edge_map[1].cdt != 0) && pr > 1)
        is_order_same = false;
    }
    IASSERT(is_order_same);

    if (is_order_same && mb==1 && nb==1 && nrow%pr==0 && ncol%pc==0 && rsrc==0 && csrc==0){
      if (this->edge_map[0].np == pr && this->edge_map[1].np == pc){
        if (lda == nrow/pr){
          memcpy(this->data, (char*)data_, sizeof(dtype)*this->size);
        } else {
          for (int64_t i=0; i<ncol/pc; i++){
            memcpy(this->data+i*lda*sizeof(dtype),(char*)(data_+i*lda), ((int64_t)nrow/pr)*sizeof(dtype));
          }
        }
      } else {
        Matrix<dtype> M(nrow, ncol, mb, nb, pr, pc, layout_order, rsrc, csrc, lda, data_);
        (*this)["ab"] = M["ab"];
      }
    } else {
      Pair<dtype> * pairs;
      int64_t nmyr, nmyc;
      get_my_kv_pair(this->wrld->rank, nrow, ncol, mb, nb, pr, pc, layout_order, rsrc, csrc, nmyr, nmyc, pairs);

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
                               char    layout_order,
                               int     rsrc,
                               int     csrc,
                               int     lda,
                               dtype * data_){
    //FIXME: (1) can optimize sparse for this case (mapping cyclic), (2) can use permute to avoid sparse redistribution always
    bool is_order_same = true;
    if (layout_order == 'C'){
      if ((this->edge_map[0].type != CTF_int::PHYSICAL_MAP || this->edge_map[0].cdt != 0) && pc > 1)
        is_order_same = false;
    } else {
      if ((this->edge_map[0].type != CTF_int::PHYSICAL_MAP || this->edge_map[1].cdt != 0) && pr > 1)
        is_order_same = false;
    }
    IASSERT(is_order_same);
    if (is_order_same && !this->is_sparse && (mb==1 && nb==1 && nrow%pr==0 && ncol%pc==0 && rsrc==0 && csrc==0)){
      if (this->edge_map[0].np == pr && this->edge_map[1].np == pc){
        if (lda == nrow/pr){
          memcpy((char*)data_, this->data, sizeof(dtype)*this->size);
        } else {
          for (int64_t i=0; i<ncol/pc; i++){
            memcpy((char*)(data_+i*lda), this->data+i*lda*sizeof(dtype), nrow*sizeof(dtype)/pr);
          }
        }
      } else {
        IASSERT(layout_order == 'C');
        int plens[] = {pr, pc};
        Partition ip(2, plens);
        Matrix M(nrow, ncol, "ij", ip["ij"], Idx_Partition(), 0, *this->wrld, *this->sr);
        M["ab"] = (*this)["ab"];
        M.read_mat(mb, nb, pr, pc, layout_order, rsrc, csrc, lda, data_);
      }
    } else {
      Pair<dtype> * pairs;
      int64_t nmyr, nmyc;
      get_my_kv_pair(this->wrld->rank, nrow, ncol, mb, nb, pr, pc, layout_order, rsrc, csrc, nmyr, nmyc, pairs);

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
  void Matrix<dtype>::get_desc(int & ictxt, int *& desc, char & layout_order){
    int pr, pc;
    pr = this->edge_map[0].calc_phase();       
    pc = this->edge_map[1].calc_phase();       
    IASSERT(this->wrld->np == pr*pc);

    layout_order = 'C';
    if (this->edge_map[1].type == CTF_int::PHYSICAL_MAP &&
        this->edge_map[1].np   >  1 &&
        this->edge_map[1].cdt  == 0)
      layout_order = 'R';
    int ctxt;
    IASSERT(this->wrld->comm == MPI_COMM_WORLD);
    CTF_SCALAPACK::cblacs_get(-1, 0, &ctxt);
    CTF_int::grid_wrapper gw;
    gw.pr = pr;
    gw.pc = pc;
    gw.layout = layout_order;
    std::set<CTF_int::grid_wrapper>::iterator s = CTF_int::scalapack_grids.find(gw);
    if (s != CTF_int::scalapack_grids.end()){
      ctxt = s->ctxt;
    } else {
      CTF_SCALAPACK::cblacs_gridinit(&ctxt, &layout_order, pr, pc);
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
                               dtype *     data_,
                               char        layout_order){
    int ictxt = desc[1];
    int pr, pc, ipr, ipc;
    CTF_SCALAPACK::cblacs_gridinfo(ictxt, &pr, &pc, &ipr, &ipc);

    read_mat(desc[4],desc[5],pr,pc,layout_order,desc[6],desc[7],desc[8],data_);
  }

  template<typename dtype>
  Matrix<dtype>::Matrix(int                       nrow_,
                        int                       ncol_,
                        int                       mb,
                        int                       nb,
                        int                       pr,
                        int                       pc,
                        char                      layout_order,
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
    write_mat(mb,nb,pr,pc,layout_order,rsrc,csrc,lda,data);
  }

  

  static inline Idx_Partition get_map_from_desc(int const * desc, char layout_order='C'){

    int ictxt = desc[1];
    int pr, pc, ipr, ipc;
    CTF_SCALAPACK::cblacs_gridinfo(ictxt, &pr, &pc, &ipr, &ipc);
    if (layout_order == 'C')
      return Partition(2,CTF_int::int2(pr, pc))["ij"];
    else
      return Partition(2,CTF_int::int2(pc, pr))["ji"];
  }

  template<typename dtype>
  Matrix<dtype>::Matrix(int const *               desc,
                        dtype const *             data_,
                        char                      layout_order,
                        World &                   wrld_,
                        CTF_int::algstrct const & sr_,
                        char const *              name_,
                        int                       profile_)
    : Tensor<dtype>(2, false, CTF_int::int2(desc[2], desc[3]),  CTF_int::int2(NS, NS),
                           wrld_, "ij", get_map_from_desc(desc,layout_order), Idx_Partition(), name_, profile_, sr_) {
    nrow = desc[2];
    ncol = desc[3];
    symm = NS;
    int ictxt = desc[1];
    int pr, pc, ipr, ipc;
    CTF_SCALAPACK::cblacs_gridinfo(ictxt, &pr, &pc, &ipr, &ipc);
    //IASSERT(ipr == wrld_.rank%pr);
    //IASSERT(ipc == wrld_.rank/pr);
    IASSERT(pr*pc == wrld_.np);
    //this->set_distribution("ij", Partition(2,CTF_int::int2(pr, pc))["ij"], Idx_Partition());
    write_mat(desc[4],desc[5],pr,pc,layout_order,desc[6],desc[7],desc[8],data_);
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
  void Matrix<dtype>::get_tri(Matrix<dtype> & T, bool lower, bool keep_diag){
    Timer t_get_tri("get_tri");
    t_get_tri.start();
    int min_mn = std::min(this->nrow,this->ncol);
    int sym_type = SH;
    if ((keep_diag && !lower) || (!keep_diag && lower)) sym_type = SY;
    int syns[] = {sym_type, NS};
    Tensor<dtype> F;
    Tensor<dtype> U;
    if (this->nrow != this->ncol){
      F = this->slice(0,((int64_t)nrow)*(min_mn-1) + min_mn-1);
      U = Tensor<dtype>(F,syns);
    } else {
      U = Tensor<dtype>(*this,syns);
    }
    int nsns[] = {NS, NS};
    U = Tensor<dtype>(U,nsns);
    if (T.nrow == -1){
      if (lower && this->nrow <= this->ncol){
        if (this->nrow == this->ncol)
          T = Tensor<dtype>(true, *this);
        else
          T = Tensor<dtype>(true, F);
      }
      else if (lower && this->nrow > this->ncol)
        T = Tensor<dtype>(false, *this);
      else if (!lower && this->nrow >= this->ncol){
        // do not copy from U to permit general mapping
        T = Tensor<dtype>(2, U.is_sparse, U.lens, *U.wrld, *U.sr);
        T["ij"] += U["ij"];
      } else
        T = Tensor<dtype>(false, *this);
    }
    if (lower){
      if (T.nrow == min_mn && T.ncol == min_mn)
        T["ij"] -= U["ij"];
      else
        F["ij"] -= U["ij"];
    }
    if (T.nrow != T.ncol){
      assert(T.nrow == this->nrow && T.ncol == this->ncol);
      if ((lower && T.nrow > T.ncol) ||
         (!lower && T.nrow < T.ncol))
        T["ij"] = this->operator[]("ij");
      else
        T.set_zero();
      T.slice(0,((int64_t)nrow)*(min_mn-1) + min_mn-1,*(dtype*)this->sr->addid(),F,0,((int64_t)min_mn)*(min_mn-1) + min_mn-1,*(dtype*)this->sr->mulid());
    }
    t_get_tri.stop();
  }

  template<typename dtype>
  void Matrix<dtype>::cholesky(Matrix<dtype> & L, bool lower){
    Timer t_cholesky("cholesky");
    t_cholesky.start();
    int info;
    int m = this->nrow;
    int n = this->ncol;
    IASSERT(m==n);

    int * desca;// = (int*)malloc(9*sizeof(int));

    int ictxt;
    char layout_order;
    this->get_desc(ictxt, desca, layout_order);
    dtype * A = (dtype*)malloc(this->size*sizeof(dtype));

    this->read_mat(desca, A, layout_order);

    char uplo = 'U';
    if (lower) uplo = 'L';

    CTF_SCALAPACK::ppotrf<dtype>(uplo,n,A,1,1,desca,&info);
    IASSERT(info == 0);

    Matrix<dtype> S(desca, A, layout_order, (*(this->wrld)));
    free(A);
    S.get_tri(L, lower);
    t_cholesky.stop();
  }
      
  template<typename dtype>
  void Matrix<dtype>::solve_tri(Matrix<dtype> & L, Matrix<dtype> & X, bool lower, bool from_left, bool transp_L){
    Timer t_solve_tri("solve_tri");
    t_solve_tri.start();
    int m = this->nrow;
    int n = this->ncol;

    int * desca;// = (int*)malloc(9*sizeof(int));
    int * descl;// = (int*)malloc(9*sizeof(int));

    int ictxt;
    char layout_order_A;
    this->get_desc(ictxt, desca, layout_order_A);

    int ictxt2;
    char layout_order_L;
    L.get_desc(ictxt2, descl, layout_order_L);
    dtype * dL;
    if  (ictxt != ictxt2){
      Partition part(this->topo->order, this->topo->lens);
      Matrix<dtype> L_r;
      if (this->topo->order == 2){
        if (this->edge_map[0].cdt == 0 && 
            this->edge_map[1].cdt == 1){ 
          L_r = Matrix<dtype>(L.nrow, L.ncol, "ij", part["ij"], Idx_Partition(), 0, *this->wrld, *this->sr);
  
        } else {
          IASSERT(this->edge_map[0].cdt == 1 && 
                  this->edge_map[1].cdt == 0);
          L_r = Matrix<dtype>(L.nrow, L.ncol, "ij", part["ji"], Idx_Partition(), 0, *this->wrld, *this->sr);
        }
      } else {
        IASSERT(this->topo->order == 1);
        if (this->edge_map[0].type == CTF_int::PHYSICAL_MAP){
          L_r = Matrix<dtype>(L.nrow, L.ncol, "ij", part["i"], Idx_Partition(), 0, *this->wrld, *this->sr);
        } else {
          L_r = Matrix<dtype>(L.nrow, L.ncol, "ij", part["j"], Idx_Partition(), 0, *this->wrld, *this->sr);
        }
      }
      L_r["ij"] += L["ij"];
      L_r.get_desc(ictxt2, descl, layout_order_L);
      IASSERT(ictxt == ictxt2);
      IASSERT(layout_order_A == layout_order_L);
      dL = (dtype*)malloc(L_r.size*sizeof(dtype));
      L_r.read_mat(descl, dL, layout_order_L);
    } else {
      IASSERT(layout_order_A == layout_order_L);
      dL = (dtype*)malloc(L.size*sizeof(dtype));
      L.read_mat(descl, dL, layout_order_L);
    }
    dtype * A = (dtype*)malloc(this->size*sizeof(dtype));

    this->read_mat(desca, A, layout_order_A);

    char SIDE = 'R';
    if (from_left) SIDE = 'L';
    char UPLO = 'U';
    if (lower) UPLO = 'L';
    char TRANS = 'N';
    if (transp_L) TRANS = 'T';
    char DIAG = 'N';

    CTF_SCALAPACK::ptrsm<dtype>(SIDE, UPLO, TRANS, DIAG, m, n, 1., dL, 1, 1, descl, A, 1, 1, desca);
    free(dL); 
    X = Matrix<dtype>(desca, A, layout_order_A, (*(this->wrld)));
    free(A);
    free(desca);
    free(descl);
    t_solve_tri.stop();
  }

  template<typename dtype>
  void Matrix<dtype>::qr(Matrix<dtype> & Q, Matrix<dtype> & R){

    Timer t_qr("QR");
    t_qr.start();
    int info;

    int m = this->nrow;
    int n = this->ncol;

    int * desca;// = (int*)malloc(9*sizeof(int));

    int ictxt;
    char layout_order;
    this->get_desc(ictxt, desca, layout_order);
    dtype * A = (dtype*)malloc(this->size*sizeof(dtype));

    this->read_mat(desca, A, layout_order);

    dtype * tau = (dtype*)malloc(((int64_t)n)*sizeof(dtype));
    dtype dlwork;
    CTF_SCALAPACK::pgeqrf<dtype>(m,n,A,1,1,desca,tau,(dtype*)&dlwork,-1,&info);
    int lwork = get_int_fromreal<dtype>(dlwork);
    dtype * work = (dtype*)malloc(((int64_t)lwork)*sizeof(dtype));
    CTF_SCALAPACK::pgeqrf<dtype>(m,n,A,1,1,desca,tau,work,lwork,&info);
 
    dtype * dQ = (dtype*)malloc(this->size*sizeof(dtype));
    memcpy(dQ,A,this->size*sizeof(dtype));
    free(A);

    Q = Matrix<dtype>(desca, dQ, layout_order, (*(this->wrld)));
    Q.get_tri(R);

    free(work);
    CTF_SCALAPACK::porgqr<dtype>(m,std::min(m,n),std::min(m,n),dQ,1,1,desca,tau,(dtype*)&dlwork,-1,&info);
    lwork = get_int_fromreal<dtype>(dlwork);
    work = (dtype*)malloc(((int64_t)lwork)*sizeof(dtype));
    CTF_SCALAPACK::porgqr<dtype>(m,std::min(m,n),std::min(m,n),dQ,1,1,desca,tau,work,lwork,&info);
    Q = Matrix<dtype>(desca, dQ, layout_order, (*(this->wrld)));
    free(dQ);
    if (m<n)
      Q = Q.slice(0,m*(m-1)+m-1);
    free(work);
    free(tau);
    free(desca);
    t_qr.stop();
  }

  template<typename dtype>
  void get_svd(dtype * A, dtype * U, dtype * S, dtype * VT, int & rank, double threshold){

  }

  template<typename dtype>
  void Matrix<dtype>::svd(Matrix<dtype> & U, Vector<dtype> & S, Matrix<dtype> & VT, int rank, double threshold){

    Timer t_svd("SVD");
    t_svd.start();
    int info;

    int m = this->nrow;
    int n = this->ncol;
    int k = std::min(m,n);


    int * desca;// = (int*)malloc(9*sizeof(int));
    int * descu = (int*)malloc(9*sizeof(int));
    int * descvt = (int*)malloc(9*sizeof(int));

    int ictxt;
    char layout_order;
    this->get_desc(ictxt, desca, layout_order);

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

    dtype * A = (dtype*)CTF_int::alloc(this->size*sizeof(dtype));


    dtype * u = (dtype*)CTF_int::alloc(sizeof(dtype)*mpr*kpc);
    dtype * vt = (dtype*)CTF_int::alloc(sizeof(dtype)*kpr*npc);
    this->read_mat(desca, A, layout_order);


    S = Vector<dtype>(k, (*(this->wrld)));
    int64_t sc;
    dtype * s_data = S.get_raw_data(&sc);

    int lwork;
    dtype dlwork;

    //if (typeid(dtype) == typeid(std::complex<float>)){
    //  float * s = (float*)CTF_int::alloc(sizeof(float)*k);
    //  CTF_SCALAPACK::pgesvd<dtype>('V', 'V', m, n, NULL, 1, 1, desca, NULL, NULL, 1, 1, descu, vt, 1, 1, descvt, &dlwork, -1, &info);  
    //  lwork = get_int_fromreal<dtype>(dlwork);
    //  float * work = (float*)CTF_int::alloc(sizeof(float)*((int64_t)lwork));
    //  CTF_SCALAPACK::pgesvd<dtype>('V', 'V', m, n, A, 1, 1, desca, s, u, 1, 1, descu, vt, 1, 1, descvt, work, lwork, &info);
    //  if (threshold > 0.0)
    //    rank = std::lower_bound(s, s+k, (float)threshold) - s;
    //  int phase = S.edge_map[0].calc_phase();
    //  if ((int)(this->wrld->rank) < phase){
    //    for (int i = S.edge_map[0].calc_phys_rank(S.topo); i < k; i += phase) {
    //      s_data[i/phase] = s[i];
    //    } 
    //  }
    //  CTF_int::cdealloc(s);
    //  CTF_int::cdealloc(work);
    //} else if (typeid(dtype) == typeid(std::complex<double>)){
    //  double * s = (double*)CTF_int::alloc(sizeof(double)*k);
    //  CTF_SCALAPACK::pgesvd<dtype>('V', 'V', m, n, NULL, 1, 1, desca, NULL, NULL, 1, 1, descu, vt, 1, 1, descvt, &dlwork, -1, &info);  
    //  lwork = get_int_fromreal<dtype>(dlwork);
    //  double * work = (double*)CTF_int::alloc(sizeof(double)*((int64_t)lwork));
    //  CTF_SCALAPACK::pgesvd<dtype>('V', 'V', m, n, A, 1, 1, desca, s, u, 1, 1, descu, vt, 1, 1, descvt, work, lwork, &info);
    //  if (threshold > 0.0)
    //    rank = std::lower_bound(s, s+k, (double)threshold) - s;
    //  int phase = S.edge_map[0].calc_phase();
    //  if ((int)(this->wrld->rank) < phase){
    //    for (int i = S.edge_map[0].calc_phys_rank(S.topo); i < k; i += phase) {
    //      s_data[i/phase] = s[i];
    //    } 
    //  }
    //  CTF_int::cdealloc(s);
    //  CTF_int::cdealloc(work);
    //} else {
    dtype * s = (dtype*)CTF_int::alloc(sizeof(dtype)*k);
    CTF_SCALAPACK::pgesvd<dtype>('V', 'V', m, n, NULL, 1, 1, desca, NULL, NULL, 1, 1, descu, vt, 1, 1, descvt, &dlwork, -1, &info);  
    lwork = get_int_fromreal<dtype>(dlwork);
    dtype * work = (dtype*)CTF_int::alloc(sizeof(dtype)*((int64_t)lwork));
    CTF_SCALAPACK::pgesvd<dtype>('V', 'V', m, n, A, 1, 1, desca, s, u, 1, 1, descu, vt, 1, 1, descvt, work, lwork, &info);
    if (threshold > 0.0){
      rank = std::upper_bound(s, s+k, (dtype)threshold, [](const dtype a, const dtype b){ return std::abs(a) > std::abs(b); }) - s;
      //printf("truncated value ");
      //this->sr->print((char*)(s+rank));
      //printf(", threshold was %lf\n",threshold);
    }
    int phase = S.edge_map[0].calc_phase();
    if ((int)(this->wrld->rank) < phase){
      for (int i = S.edge_map[0].calc_phys_rank(S.topo); i < k; i += phase) {
        s_data[i/phase] = s[i];
      } 
    }
    CTF_int::cdealloc(s);
    CTF_int::cdealloc(work);

    U = Matrix<dtype>(descu, u, layout_order, (*(this->wrld)));
    VT = Matrix<dtype>(descvt, vt, layout_order, (*(this->wrld)));

    if (rank > 0 && rank < k) {
      S = S.slice(0, rank-1);
      U = U.slice(0, rank*((int64_t)m)-1);
      VT = VT.slice(0, k*((int64_t)n)-(k-rank+1));
    }

    CTF_int::cdealloc(A);
    CTF_int::cdealloc(u);
    CTF_int::cdealloc(vt);
    free(desca);
    free(descu);
    free(descvt);
    t_svd.stop();

  }

    
  template<typename dtype>
  void Matrix<dtype>::svd_rand(Matrix<dtype> & U, Vector<dtype> & S, Matrix<dtype> & VT, int rank, int iter, int oversamp, Matrix<dtype> * U_guess){
    Timer t_svd("SVD_rand");
    t_svd.start();
    int max_rank = std::min(std::min(nrow,ncol), rank+oversamp);
    IASSERT(rank+oversamp <= std::min(nrow,ncol) || U_guess==NULL);
    bool del_U_guess = false;
    if (U_guess == NULL){
      del_U_guess = true;
      U_guess = new Matrix<dtype>(this->nrow, max_rank);
      U_guess->fill_random(-1.,1.);
      Matrix<dtype> Q, R;
      U_guess->qr(Q, R);
      U_guess->operator[]("ij") = Q["ij"];
    }
    for (int i=0; i<iter; i++){
      U_guess->operator[]("ir") = this->operator[]("ij") * this->operator[]("lj") * U_guess->operator[]("lr");
      Matrix<dtype> Q, R;
      U_guess->qr(Q,R);
      U_guess->operator[]("ij") = Q["ij"];
    }
    if (max_rank - rank > 0)
      U = U_guess->slice(0,this->nrow*rank-1);
    else
      U = Matrix<dtype>(*U_guess);
    if (del_U_guess)
      delete U_guess;
    Matrix<dtype> B(rank, this->ncol);
    B["ij"] = U["ki"]*this->operator[]("kj");
    Matrix<dtype> U1;
    B.svd(U1,S,VT,rank);
    U["ij"] = U["ik"] * U1["kj"];
    t_svd.stop();
  }
}
