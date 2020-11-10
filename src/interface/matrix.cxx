/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "common.h"
#include "world.h"
#include "timer.h"
#include "../shared/blas_symbs.h"
#include "../shared/lapack_symbs.h"
#include <stdlib.h>
#include <algorithm>


namespace CTF_int{
  void factorize(int n, int *nfactor, int **factor);

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
  struct int64_t2
  {
    int64_t i[2];
    int64_t2(int64_t a, int64_t b)
    {
      i[0] = a;
      i[1] = b;
    }
    operator const int64_t*() const
    {
      return i;
    }
  };

  inline int64_t estimate_qr_flops(int nrow, int ncol) {
    int64_t m = std::max(nrow, ncol);
    int64_t n = std::min(nrow, ncol);
    int64_t nflops = 2.*m*n*n-(2./3.)*n*n*n;
    return nflops;
  }

  inline int64_t estimate_svd_flops(int nrow, int ncol) {
    int64_t m = std::max(nrow, ncol);
    int64_t n = std::min(nrow, ncol);
    int64_t nflops;
    if (m >= (10./3.)*n)
      nflops = 6.*m*n*n+8.*n*n*n;
    else
      nflops = 8.*m*n*n+(4./3.)*n*n*n;
    return nflops;
  }
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
      case SH:
        symm=SH;
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
  Matrix<dtype>::Matrix(int64_t                   nrow_,
                        int64_t                   ncol_,
                        World &                   world_,
                        CTF_int::algstrct const & sr_,
                        char const *              name_,
                        int                       profile_)
    : Tensor<dtype>(2, false, CTF_int::int64_t2(nrow_, ncol_),  CTF_int::int2(NS, NS),
                           world_, sr_, name_, profile_) {
    nrow = nrow_;
    ncol = ncol_;
    symm = NS;
  }

  template<typename dtype>
  Matrix<dtype>::Matrix(int64_t                   nrow_,
                        int64_t                   ncol_,
                        int                       atr_,
                        World &                   world_,
                        CTF_int::algstrct const & sr_,
                        char const *              name_,
                        int                       profile_)
    : Tensor<dtype>(2, (atr_&4)>0, CTF_int::int64_t2(nrow_, ncol_), CTF_int::int2(atr_&3, NS),
                           world_, sr_, name_, profile_) {
    nrow = nrow_;
    ncol = ncol_;
    symm = atr_&3;
  }

  template<typename dtype>
  Matrix<dtype>::Matrix(int64_t                   nrow_,
                        int64_t                   ncol_,
                        World &                   world_,
                        char const *              name_,
                        int                       profile_,
                        CTF_int::algstrct const & sr_)
    : Tensor<dtype>(2, false, CTF_int::int64_t2(nrow_, ncol_), CTF_int::int2(NS, NS),
                           world_, sr_, name_, profile_) {
    nrow = nrow_;
    ncol = ncol_;
    symm = 0;
  }


  template<typename dtype>
  Matrix<dtype>::Matrix(int64_t                   nrow_,
                        int64_t                   ncol_,
                        int                       atr_,
                        World &                   world_,
                        char const *              name_,
                        int                       profile_,
                        CTF_int::algstrct const & sr_)
    : Tensor<dtype>(2, (atr_&4)>0, CTF_int::int64_t2(nrow_, ncol_), CTF_int::int2(atr_&3, NS),
                           world_, sr_, name_, profile_) {
    nrow = nrow_;
    ncol = ncol_;
    symm = atr_&3;
  }


  template<typename dtype>
  Matrix<dtype>::Matrix(int64_t                   nrow_,
                        int64_t                   ncol_,
                        char const *              idx,
                        Idx_Partition const &     prl,
                        Idx_Partition const &     blk,
                        int                       atr_,
                        World &                   world_,
                        CTF_int::algstrct const & sr_,
                        char const *              name_,
                        int                       profile_)
    : Tensor<dtype>(2, (atr_&4)>0, CTF_int::int64_t2(nrow_, ncol_), CTF_int::int2(atr_&3, NS),
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
                      int64_t        nrow,
                      int64_t        ncol,
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

    if (is_order_same && mb==1 && nb==1 && rsrc==0 && csrc==0){
      if (this->edge_map[0].np == pr && this->edge_map[1].np == pc){
        //assert(lda == (nrow+this->padding[0])/pr);
        //memcpy(this->data, (char*)data_, sizeof(dtype)*this->size);
        if (lda == nrow/pr){
          int64_t copy_ncol = ncol/pc;
          if (this->edge_map[1].calc_phys_rank(this->topo) < ncol%pc) copy_ncol++;
          memcpy(this->data, (char*)data_, sizeof(dtype)*lda*copy_ncol);
        } else {
          int64_t copy_len = nrow/pr;
          if (this->edge_map[0].calc_phys_rank(this->topo) < nrow%pr) copy_len++;
          int ipc = this->edge_map[1].calc_phys_rank(this->topo);
          for (int64_t i=0; ipc+i*pc<ncol; i++){
            memcpy(this->data+i*((nrow+this->padding[0])/pr)*sizeof(dtype),(char*)(data_+i*lda), copy_len*sizeof(dtype));
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
      if ((this->edge_map[1].type != CTF_int::PHYSICAL_MAP || this->edge_map[1].cdt != 0) && pr > 1)
        is_order_same = false;
    }
    IASSERT(is_order_same);
    //if (is_order_same && !this->is_sparse && (mb==1 && nb==1 && nrow%pr==0 && ncol%pc==0 && rsrc==0 && csrc==0))
    if (is_order_same && !this->is_sparse && (mb==1 && nb==1 && rsrc==0 && csrc==0)){
      if (this->sym[0] == NS && this->edge_map[0].np == pr && this->edge_map[1].np == pc){
        assert(lda == (nrow+this->padding[0])/pr);
        memcpy((char*)data_, this->data, sizeof(dtype)*this->size);
        //if (lda == (nrow+this->padding[0])/pr){
        //  memcpy((char*)data_, this->data, sizeof(dtype)*this->size);
        //} else {
        //  for (int64_t i=0; i*pc<ncol; i++){
        //    memcpy((char*)(data_+i*lda), this->data+i*((nrow+this->padding[0])/pr)*sizeof(dtype), nrow*sizeof(dtype)/pr);
        //  }
        //}
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
    IASSERT(this->sym[0] == NS);
    int pr, pc;
    pr = this->edge_map[0].calc_phys_phase();
    pc = this->edge_map[1].calc_phys_phase();
    IASSERT(this->wrld->np == pr*pc);

    layout_order = 'C';
    if (this->edge_map[1].type == CTF_int::PHYSICAL_MAP &&
        this->edge_map[1].np   >  1 &&
       (this->edge_map[0].type  != CTF_int::PHYSICAL_MAP
        || this->edge_map[0].cdt > this->edge_map[1].cdt))
      layout_order = 'R';
    int ctxt;
    CTF_SCALAPACK::cblacs_get(-1, 0, &ctxt);
    CTF_int::grid_wrapper gw;
    gw.pr = pr;
    gw.pc = pc;
    gw.layout = layout_order;
    std::set<CTF_int::grid_wrapper>::iterator s = CTF_int::scalapack_grids.find(gw);
    if (s != CTF_int::scalapack_grids.end()){
      ctxt = s->ctxt;
    } else {
      int tot_np;
      MPI_Comm_size(MPI_COMM_WORLD, &tot_np);
      if (tot_np == pr*pc){
        CTF_SCALAPACK::cblacs_gridinit(&ctxt, &layout_order, pr, pc);
        gw.ctxt = ctxt;
        CTF_int::scalapack_grids.insert(gw);
      } else {
        int * all_ranks = (int*)CTF_int::alloc(sizeof(int)*pr*pc);
        int myrank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        MPI_Allgather(&myrank, 1, MPI_INT, all_ranks, 1, MPI_INT, this->wrld->comm);
        if (layout_order == 'R'){
          int * all_ranksT = (int*)CTF_int::alloc(sizeof(int)*pr*pc);
          for (int ii=0; ii<pr; ii++){
            for (int jj=0; jj<pc; jj++){
              //all_ranksT[ii*pc+jj] = all_ranks[ii+jj*pr];
              all_ranksT[ii+jj*pr] = all_ranks[ii*pc+jj];
            }
          }
          CTF_int::cdealloc(all_ranks);
          all_ranks = all_ranksT;
        }
        CTF_int::grid_map_wrapper mgw;
        mgw.pr = pr;
        mgw.pc = pc;
        mgw.layout = layout_order;
        mgw.allranks = all_ranks;
        std::set<CTF_int::grid_map_wrapper>::iterator ms = CTF_int::scalapack_grid_maps.find(mgw);
        if (ms != CTF_int::scalapack_grid_maps.end()){
          ctxt = ms->ctxt;
          CTF_int::cdealloc(all_ranks);
        } else {
          CTF_SCALAPACK::cblacs_gridmap(&ctxt, all_ranks, pr, pr, pc);
          mgw.ctxt = ctxt;
          CTF_int::scalapack_grid_maps.insert(mgw);
        }
      }
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
  Matrix<dtype>::Matrix(int64_t                   nrow_,
                        int64_t                   ncol_,
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
    : Tensor<dtype>(2, false, CTF_int::int64_t2(nrow_, ncol_),  CTF_int::int2(NS, NS),
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
    //extract upper triangular matrix
    if (this->nrow != this->ncol){
      F = this->slice(0,((int64_t)nrow)*(min_mn-1) + min_mn-1);
      U = Tensor<dtype>(F,syns);
    } else {
      U = Tensor<dtype>(*this,syns);
    }
    int nsns[] = {NS, NS};
    U = Tensor<dtype>(U,nsns);
    // if T is uninitialized
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
    if (T.nrow == min_mn && T.ncol == min_mn) {
      if (lower) T["ij"] -= U["ij"];
    } else {
      if (lower)
        F["ij"] -= U["ij"];
      else
        F = U;
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
    if (this->sym[0] != NS){
      Matrix<dtype> A(this->nrow, this->ncol, *this->wrld);
      A["ij"] = this->operator[]("ij");
      return A.cholesky(L, lower);
    }
    int info;
    int m = this->nrow;
    int n = this->ncol;
    IASSERT(m==n);

    int * desca;// = (int*)malloc(9*sizeof(int));

    int ictxt;
    char layout_order;
    this->get_desc(ictxt, desca, layout_order);

    int pr, pc;
    pr = this->edge_map[0].calc_phase();
    pc = this->edge_map[1].calc_phase();
    //CTF_SCALAPACK::cdescinit(desca, m, n, 1, 1, 0, 0, ictxt, m/(*(this->wrld)).np, &info);
    int64_t mpr = m/pr + (m % pr != 0);
    int64_t npc = n/pc + (n % pc != 0);


    dtype * A = (dtype*)malloc(mpr*npc*sizeof(dtype));

    this->read_mat(desca, A, layout_order);

    char uplo = 'U';
    if (lower) uplo = 'L';

    Timer __t("SCALAPACK_PPOTRF");
    __t.start();
    CTF_SCALAPACK::ppotrf<dtype>(uplo,n,A,1,1,desca,&info);
    __t.stop();
    IASSERT(info == 0);

    Matrix<dtype> S(desca, A, layout_order, (*(this->wrld)));
    free(A);
    free(desca);
    S.get_tri(L, lower);
    t_cholesky.stop();
  }

  template<typename dtype>
  void map_matrix_to_my_context(Matrix<dtype> const * me, Matrix<dtype> & other, Matrix<dtype> & out){
    Partition part(me->topo->order, me->topo->lens);
    if (me->topo->order == 2){
      if (me->edge_map[0].cdt == 0 &&
          me->edge_map[1].cdt == 1){
        out = Matrix<dtype>(other.nrow, other.ncol, "ij", part["ij"], Idx_Partition(), 0, *me->wrld, *me->sr);

      } else {
        IASSERT(me->edge_map[0].cdt == 1 &&
                me->edge_map[1].cdt == 0);
        out = Matrix<dtype>(other.nrow, other.ncol, "ij", part["ji"], Idx_Partition(), 0, *me->wrld, *me->sr);
      }
    } else {
      IASSERT(me->topo->order == 1);
      if (me->edge_map[0].type == CTF_int::PHYSICAL_MAP){
        out = Matrix<dtype>(other.nrow, other.ncol, "ij", part["i"], Idx_Partition(), 0, *me->wrld, *me->sr);
      } else {
        out = Matrix<dtype>(other.nrow, other.ncol, "ij", part["j"], Idx_Partition(), 0, *me->wrld, *me->sr);
      }
    }
    out["ij"] += other["ij"];
  }

  template<typename dtype>
  void Matrix<dtype>::solve_tri(Matrix<dtype> & L, Matrix<dtype> & X, bool lower, bool from_left, bool transp_L){
    Timer t_solve_tri("solve_tri");
    t_solve_tri.start();
    if (this->sym[0] != NS){
      Matrix<dtype> B(this->nrow, this->ncol, *this->wrld);
      B["ij"] = this->operator[]("ij");
      return B.solve_tri(L, X, lower, from_left, transp_L);
    }
    if (L.sym[0] != NS){
      Matrix<dtype> LF(this->nrow, this->ncol, *this->wrld);
      LF["ij"] = L["ij"];
      return this->solve_tri(L, X, lower, from_left, transp_L);
    }

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
      map_matrix_to_my_context<dtype>(this, L, L_r);
      free(descl);
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

    int pr, pc;
    pr = this->edge_map[0].calc_phase();
    pc = this->edge_map[1].calc_phase();
    //CTF_SCALAPACK::cdescinit(desca, m, n, 1, 1, 0, 0, ictxt, m/(*(this->wrld)).np, &info);
    int64_t mpr = m/pr + (m % pr != 0);
    int64_t npc = n/pc + (n % pc != 0);


    dtype * A = (dtype*)malloc(mpr*npc*sizeof(dtype));

    this->read_mat(desca, A, layout_order_A);

    char SIDE = 'R';
    if (from_left) SIDE = 'L';
    char UPLO = 'U';
    if (lower) UPLO = 'L';
    char TRANS = 'N';
    if (transp_L) TRANS = 'T';
    char DIAG = 'N';

    Timer __t("SCALAPACK_PTRSM");
    __t.start();
    CTF_SCALAPACK::ptrsm<dtype>(SIDE, UPLO, TRANS, DIAG, m, n, 1., dL, 1, 1, descl, A, 1, 1, desca);
    __t.stop();
    free(dL);
    X = Matrix<dtype>(desca, A, layout_order_A, (*(this->wrld)));
    free(A);
    free(desca);
    free(descl);
    t_solve_tri.stop();
  }

  template<typename dtype>
  void Matrix<dtype>::solve_spd(Matrix<dtype> & M, Matrix<dtype> & X){
    Timer t_solve_spd("solve_spd");
    t_solve_spd.start();
    int p = M.wrld->np;
    int nfactor;
    int * factors;
    CTF_int::factorize(p, &nfactor, &factors);
    int target_pr = 1;
    int target_pc = 1;
    for (int i=0; i<nfactor; i++){
      target_pr *= factors[i];
      target_pc *= factors[i];
      if (i<nfactor-1 && factors[i] == factors[i+1]){
        i++;
      }
    }
    if (nfactor>0)
      CTF_int::cdealloc(factors);
    int virt_factor = target_pc/(p/target_pr);
    int pe_dims[2] = {target_pr, target_pc/virt_factor};
    int virt_dims[1] = {virt_factor};
    Partition proc_grid_2d(2,pe_dims);
    Partition virt_grid_1d(1,virt_dims);

    if (M.sym[0] != NS || M.edge_map[0].calc_phase() != M.edge_map[1].calc_phase()){
      Matrix<dtype> MM(M.nrow, M.ncol, "ij", proc_grid_2d["ij"], virt_grid_1d["j"], 0, *M.wrld);
      MM["ij"] = M.operator[]("ij");
      return this->solve_spd(MM, X);
    }

    int info;
    int m = M.nrow;
    int n = M.ncol;
    IASSERT(m==n);
    int nrhs = this->ncol;

    int * desca;// = (int*)malloc(9*sizeof(int));

    int ictxt;
    char layout_order;
    M.get_desc(ictxt, desca, layout_order);

    int * descb;// = (int*)malloc(9*sizeof(int));

    int ictxt2;
    char layout_order2;
    this->get_desc(ictxt2, descb, layout_order2);

    Matrix<dtype> * S;
    Matrix<dtype> * B;

    if (ictxt != ictxt2){
      S = &M;
      B = new Matrix<dtype>();
      map_matrix_to_my_context<dtype>(&M, *this, *B);
      //if (nrhs > 2*n){
      //  S = &M;
      //  B = new Matrix<dtype>();
      //  map_matrix_to_my_context<dtype>(&M, *this, *B);
      //} else {
      //  S = new Matrix<dtype>();
      //  B = this;
      //  map_matrix_to_my_context<dtype>(B, M, *S);
      //}
      free(desca);
      free(descb);
      S->get_desc(ictxt, desca, layout_order);
      B->get_desc(ictxt2, descb, layout_order2);
    } else {
      S = &M;
      B = this;
    }

    //printf("B is\n");
    //B->print_map();

    IASSERT(ictxt == ictxt2);
    IASSERT(layout_order == layout_order2);

    int ipr, ipc;
    int pr, pc;
    ipr = S->edge_map[0].calc_phys_rank(S->topo);
    ipc = S->edge_map[1].calc_phys_rank(S->topo);
    pr = S->edge_map[0].calc_phase();
    pc = S->edge_map[1].calc_phase();

    int prB, pcB;
    prB = B->edge_map[0].calc_phase();
    pcB = B->edge_map[1].calc_phase();

    //CTF_SCSLSPSCK::cdescinit(desca, m, n, 1, 1, 0, 0, ictxt, m/(*(S->wrld)).np, &info);
    int64_t mpr = m/pr + (m % pr != 0);
    int64_t npc = (n/pc + (n % pc != 0))*virt_factor;
    int64_t mprB = m/prB + (m % prB != 0);
    int64_t nrhspcB = nrhs/pcB + (nrhs % pcB != 0);

    // select b to be ceil(k/ceil(k/b)) which is the smallest block size with the same number of block as block size 64

//#define SCALAPACK_BSIZE 1024
//
//    int nloc_blk = ((mpr+SCALAPACK_BSIZE-1))/SCALAPACK_BSIZE;
//    // adjust to block size that does not require repadding if possible
//    if (nloc_blk > 1 && (mpr%nloc_blk) != 0 && (mpr%(nloc_blk-1)) == 0) nloc_blk--;
//    if ((mpr%nloc_blk) != 0 && (mpr%(nloc_blk+1)) == 0) nloc_blk++;
//    int mb = (mpr+nloc_blk-1)/nloc_blk;
//    int nb = std::min(mprB,std::min(npc,(int64_t)mb));
    int mb = mpr;
    int nb = npc/virt_factor;

    int64_t pad_mpr = ((mpr + mb - 1) / mb)*mb;
    int64_t pad_npc = ((npc + nb - 1) / nb)*nb;

    dtype * S_data_pad;
    if (m/pr == pad_mpr){
      S_data_pad = (dtype*)S->data;
    } else {
      S_data_pad = (dtype*)CTF_int::alloc(pad_mpr*pad_npc*sizeof(dtype));
      S->sr->copy(mpr, npc, (char const *)S->data, mb, (char*)S_data_pad, pad_mpr);
      for (int64_t i=m/pr+(ipr<m%pr); i<pad_mpr; i++){
        int64_t row_idx = i*pr+ipr;
        if ((row_idx%(pc/virt_factor)) == ipc){
          S_data_pad[i+((row_idx/pc)+((row_idx/(pc/virt_factor))%virt_factor)*nb)*pad_mpr] = 1.;
        }
      }
    }

    //dtype * A = (dtype*)malloc(pad_mpr*pad_npc*sizeof(dtype));
    //S->read_mat(desca, A, layout_order);

    //CTF_SCMLMPMCK::cdescinit(desca, m, n, 1, 1, 0, 0, ictxt, m/(*(M->wrld)).np, &info);

    int mbB = nb;
    int nrhsbB = nrhspcB; //std::min(nrhspcB,(int64_t)nb);

    int64_t pad_mprB = ((mprB + mbB - 1) / mbB)*mbB;
    int64_t pad_nrhspcB = ((nrhspcB + nrhsbB - 1) / nrhsbB)*nrhsbB;

    dtype * B_data_pad;
    if (m/prB == pad_mprB){// == mprB && pad_nrhspcB == nrhspcB)
      B_data_pad = (dtype*)B->data;
    } else {
      B_data_pad = (dtype*)CTF_int::alloc(pad_mprB*pad_nrhspcB*sizeof(dtype));
      B->sr->copy(mprB, nrhspcB, (char const *)B->data, mprB, (char *)B_data_pad, pad_mprB);
      if (B != this){
        delete B;
        B = this;
      }
    }

    IASSERT(mb==mbB);

    //Trick scalapack into treating the matrix as blocked and not cyclic
    //to do this, need to include padding, and make sure that S contains zeros on padded diagonal entries as above. This is because the first m columns of a cyclic matrix are different in the physical data than on the blocked matrix, so ScaLAPACK would not operate on the right data if we tell it to use m and nrhs as dims
    desca[2] = pad_mpr*pr;
    desca[3] = pad_npc*pc/virt_factor;
    desca[4] = mb;
    desca[5] = nb;
    descb[2] = pad_mprB*prB;
    descb[3] = pad_nrhspcB*pcB;
    descb[4] = mbB;
    descb[5] = nrhsbB;

    //printf("S block sizes are %ld by %ld and B block sizes are %ld by %ld\n",(int64_t)mb,(int64_t)nb,(int64_t)mbB,(int64_t)nrhsbB);
    //printf("desc23 are %ld %ld and %ld %ld\n",(int64_t)desca[2],(int64_t)desca[3],(int64_t)descb[2],(int64_t)descb[3]);
    
    Timer __t("SCALAPACK_PPOSV");
    __t.start();
    CTF_SCALAPACK::pposv<dtype>('L',pad_mprB*prB,nrhspcB*pcB,S_data_pad,1,1,desca,B_data_pad,1,1,descb,&info);
    __t.stop();
    //printf("info is %d\n",info);

    //desca[4] = 1;
    //desca[5] = 1;
    free(desca);
    descb[2] = m;
    descb[3] = nrhs;
    descb[4] = 1;
    descb[5] = 1;

    if (S->data != (char*)S_data_pad) CTF_int::cdealloc(S_data_pad);
    if (S != &M) delete S;

    if (pad_mpr == mpr){
      X = Matrix<dtype>(descb, B_data_pad, layout_order2, (*(this->wrld)));
    } else {
      dtype * X_data_pad = (dtype*)malloc(mprB*nrhspcB*sizeof(dtype));
      B->sr->copy(mprB, nrhspcB, (char const *)B_data_pad, pad_mprB, (char *)X_data_pad, mprB);
      X = Matrix<dtype>(descb, X_data_pad, layout_order2, (*(this->wrld)));
    }
    if (B_data_pad != (dtype*)B->data)
      CTF_int::cdealloc(B_data_pad);
    if (B != this){
      delete B;
      B = this;
    }

    free(descb);
    t_solve_spd.stop();
  }

  template<typename dtype>
  void Matrix<dtype>::qr(Matrix<dtype> & Q, Matrix<dtype> & R){

    Timer t_qr("QR");
    t_qr.start();
    if (this->sym[0] != NS){
      Matrix<dtype> A(this->nrow, this->ncol, *this->wrld);
      A["ij"] = this->operator[]("ij");
      return A.qr(Q,R);
    }
    int info;

    int m = this->nrow;
    int n = this->ncol;

    int * desca;// = (int*)malloc(9*sizeof(int));

    int ictxt;
    char layout_order;
    this->get_desc(ictxt, desca, layout_order);

    int pr, pc;
    pr = this->edge_map[0].calc_phase();
    pc = this->edge_map[1].calc_phase();
    //CTF_SCALAPACK::cdescinit(desca, m, n, 1, 1, 0, 0, ictxt, m/(*(this->wrld)).np, &info);
    int64_t mpr = m/pr + (m % pr != 0);
    int64_t npc = n/pc + (n % pc != 0);

    CTF_int::add_estimated_flops(CTF_int::estimate_qr_flops(m, n));

    dtype * A = (dtype*)malloc(mpr*npc*sizeof(dtype));

    this->read_mat(desca, A, layout_order);

    dtype * tau = (dtype*)malloc(((int64_t)n)*sizeof(dtype));
    dtype dlwork;
    Timer __t("SCALAPACK_PGEQRF");
    __t.start();
    CTF_SCALAPACK::pgeqrf<dtype>(m,n,A,1,1,desca,tau,(dtype*)&dlwork,-1,&info);
    int lwork = CTF_SCALAPACK::get_int_fromreal<dtype>(dlwork);
    dtype * work = (dtype*)malloc(((int64_t)lwork)*sizeof(dtype));
    CTF_SCALAPACK::pgeqrf<dtype>(m,n,A,1,1,desca,tau,work,lwork,&info);
    __t.stop();

    dtype * dQ = (dtype*)malloc(mpr*npc*sizeof(dtype));
    memcpy(dQ,A,mpr*npc*sizeof(dtype));
    free(A);

    Q = Matrix<dtype>(desca, dQ, layout_order, (*(this->wrld)));
    Q.get_tri(R);

    free(work);
    Timer __t2("SCALAPACK_PORGQR");
    __t2.start();
    CTF_SCALAPACK::porgqr<dtype>(m,std::min(m,n),std::min(m,n),dQ,1,1,desca,tau,(dtype*)&dlwork,-1,&info);
    lwork = CTF_SCALAPACK::get_int_fromreal<dtype>(dlwork);
    work = (dtype*)malloc(((int64_t)lwork)*sizeof(dtype));
    CTF_SCALAPACK::porgqr<dtype>(m,std::min(m,n),std::min(m,n),dQ,1,1,desca,tau,work,lwork,&info);
    __t2.stop();
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
  void Matrix<dtype>::svd(Matrix<dtype> & U, Vector<dtype> & S, Matrix<dtype> & VT, int rank, double threshold){

    Timer t_svd("SVD");
    t_svd.start();
    if (this->sym[0] != NS){
      Matrix<dtype> A(this->nrow, this->ncol, *this->wrld);
      A["ij"] = this->operator[]("ij");
      return A.svd(U,S,VT,rank,threshold);
    }

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
    
    int _pr, _pc, _ipr, _ipc;
    CTF_SCALAPACK::cblacs_gridinfo(ictxt, &_pr, &_pc, &_ipr, &_ipc);

    CTF_SCALAPACK::cdescinit(descu, m, k, 1, 1, 0, 0, ictxt, mpr, &info);
    CTF_SCALAPACK::cdescinit(descvt, k, n, 1, 1, 0, 0, ictxt, kpr, &info);

    dtype * A = (dtype*)CTF_int::alloc(mpr*npc*sizeof(dtype));


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
    //  lwork = CTF_SCALAPACK::get_int_fromreal<dtype>(dlwork);
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
    //  lwork = CTF_SCALAPACK::get_int_fromreal<dtype>(dlwork);
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

    CTF_int::add_estimated_flops(CTF_int::estimate_svd_flops(m, n));

    dtype * s = (dtype*)CTF_int::alloc(sizeof(dtype)*k);
    Timer __t("SCALAPACK_PGESVD");
    __t.start();
    CTF_SCALAPACK::pgesvd<dtype>('V', 'V', m, n, NULL, 1, 1, desca, NULL, NULL, 1, 1, descu, vt, 1, 1, descvt, &dlwork, -1, &info);
    lwork = CTF_SCALAPACK::get_int_fromreal<dtype>(dlwork);
    dtype * work = (dtype*)CTF_int::alloc(sizeof(dtype)*((int64_t)lwork));
    CTF_SCALAPACK::pgesvd<dtype>('V', 'V', m, n, A, 1, 1, desca, s, u, 1, 1, descu, vt, 1, 1, descvt, work, lwork, &info);
    __t.stop();
    if (threshold > 0.0){
      int rankt = std::upper_bound(s, s+k, (dtype)threshold, [](const dtype a, const dtype b){ return std::abs(a) > std::abs(b); }) - s;
      if (rank > 0){
        rank = std::min(rankt, rank);
      } else {
        rank = rankt;
      }
      //printf("truncated value ");
      //this->sr->print((char*)(s+rank));
      //printf(", threshold was %lf, rank is %d rankt is %d\n",threshold, rank, rankt);
    }
    bool is_zero_layer_S;
    if (S.edge_map[0].type == CTF_int::PHYSICAL_MAP)
      is_zero_layer_S = (S.wrld->rank == S.topo->lda[S.edge_map[0].cdt]*S.edge_map[0].calc_phys_rank(S.topo));
    else
      is_zero_layer_S = (S.wrld->rank == 0);
    int phase = S.edge_map[0].calc_phase();
    //if ((int)(this->wrld->rank) < phase){
    if (is_zero_layer_S){
      for (int i = S.edge_map[0].calc_phys_rank(S.topo); i < k; i += phase) {
        s_data[i/phase] = s[i];
      }
    }
    //}
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
    if (this->sym[0] != NS){
      Matrix<dtype> A(this->nrow, this->ncol, *this->wrld);
      A["ij"] = this->operator[]("ij");
      return A.svd_rand(U,S,VT,rank,iter,oversamp,U_guess);
    }
    int max_rank = std::min(std::min(nrow,ncol), (int64_t)rank+oversamp);
    IASSERT(rank+oversamp <= std::min(nrow,ncol) || U_guess==NULL);
    bool del_U_guess = false;
    if (U_guess == NULL){
      del_U_guess = true;
      U_guess = new Matrix<dtype>(this->nrow, max_rank, *this->wrld);
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
    Matrix<dtype> B(rank, this->ncol, *this->wrld);
    B["ij"] = U["ki"]*this->operator[]("kj");
    Matrix<dtype> U1;
    B.svd(U1,S,VT,rank);
    U["ij"] = U["ik"] * U1["kj"];
    t_svd.stop();
  }


  template<typename dtype>
  void Matrix<dtype>::eigh(Matrix<dtype> & U, Vector<dtype> & D){
    Timer t_eigh("EIGH");
    t_eigh.start();
    if (this->sym[0] != NS){
      Matrix<dtype> A(this->nrow, this->ncol, *this->wrld);
      A["ij"] = this->operator[]("ij");
      return A.eigh(U,D);
    }
    int info;

    int64_t m = this->nrow;
    int64_t n = this->ncol;

    IASSERT(m==n);

    int pr, pc;
    pr = this->edge_map[0].calc_phase();
    pc = this->edge_map[1].calc_phase();

    //ScaLAPACK currently only supports square grids for symeigsolve
    if (pr != pc){
      IASSERT(this->wrld->comm == MPI_COMM_WORLD);
      int ctxt;
      CTF_SCALAPACK::cblacs_get(-1, 0, &ctxt);
      int sqrtp = (int)std::sqrt((double)(this->wrld->np));
      while ((sqrtp+1) * (sqrtp+1) <= this->wrld->np) sqrtp++;
      if (sqrtp * sqrtp == this->wrld->np){
        int dims[2];
        dims[0] = sqrtp;
        dims[1] = sqrtp;
        Matrix<dtype> A(this->nrow, this->ncol, "ij", Partition(2,dims)["ij"], Idx_Partition(), 0, *this->wrld);
        A["ij"] = this->operator[]("ij");
        A.eigh(U,D);
      } else {
        MPI_Comm subcomm;
        MPI_Comm_split(this->wrld->comm, ((int)this->wrld->rank) < sqrtp*sqrtp, this->wrld->rank, &subcomm);
        World sworld(subcomm);
        //cblacs_gridinit needs to be called by all processors in bigger comm, so create the grid with all processes and add it (FIXME: create general infrastructure for this) FIXME: also, this won't extend to calling eigh on subcomms.
        CTF_int::grid_wrapper gw;
        gw.pr = sqrtp;
        gw.pc = sqrtp;
        gw.layout = 'C';
        std::set<CTF_int::grid_wrapper>::iterator s = CTF_int::scalapack_grids.find(gw);
        if (s == CTF_int::scalapack_grids.end()){
          CTF_SCALAPACK::cblacs_gridinit(&ctxt, &gw.layout, gw.pr, gw.pc);
          gw.ctxt = ctxt;
          CTF_int::scalapack_grids.insert(gw);
        }
        int dims[2];
        dims[0] = sqrtp;
        dims[1] = sqrtp;
        if (((int)this->wrld->rank) < sqrtp*sqrtp){
          Matrix<dtype> A(this->nrow, this->ncol, "ij", Partition(2,dims)["ij"], Idx_Partition(), 0, sworld);
          this->add_to_subworld(&A);
          Matrix<dtype> sU;
          Vector<dtype> sD;
          A.eigh(sU,sD);
          U = Matrix<dtype>(n, n, *this->wrld);
          D = Vector<dtype>(n, *this->wrld);
          U.add_from_subworld(&sU);
          D.add_from_subworld(&sD);
        } else {
          Tensor<dtype> dummy;//0,0,(int64_t*)NULL,NULL,sworld);
          this->add_to_subworld(&dummy);
          U = Matrix<dtype>(n, n, *this->wrld);
          D = Vector<dtype>(n, *this->wrld);
          U.add_from_subworld(&dummy);
          D.add_from_subworld(&dummy);
        }
        MPI_Comm_free(&subcomm);
      }
      t_eigh.stop();
      return;
    }

    int * desca;// = (int*)malloc(9*sizeof(int));
    int * descu = (int*)malloc(9*sizeof(int));

    int ictxt;
    char layout_order;
    this->get_desc(ictxt, desca, layout_order);

    //CTF_SCALAPACK::cdescinit(desca, m, n, 1, 1, 0, 0, ictxt, m/(*(this->wrld)).np, &info);
    int64_t npr = n/pr + (n % pr != 0);
    int64_t npc = n/pc + (n % pc != 0);

    CTF_SCALAPACK::cdescinit(descu, n, n, 1, 1, 0, 0, ictxt, npr, &info);

    dtype * A = (dtype*)CTF_int::alloc(npr*npc*sizeof(dtype));
    dtype * u = (dtype*)CTF_int::alloc(sizeof(dtype)*npr*npc);
    this->read_mat(desca, A, layout_order);


    D = Vector<dtype>(n, (*(this->wrld)));
    int64_t sc;
    dtype * d_data = D.get_raw_data(&sc);

    dtype * d = (dtype*)CTF_int::alloc(sizeof(dtype)*n);
    Timer __t("SCALAPACK_PGEIGH");
    __t.start();
    CTF_SCALAPACK::pgeigh('U', n, npr, npc, A, desca, d, u, desca);
    __t.stop();

    int phase = D.edge_map[0].calc_phase();
    if ((int)(this->wrld->rank) < phase){
      for (int i = D.edge_map[0].calc_phys_rank(D.topo); i < n; i += phase) {
        d_data[i/phase] = d[i];
      }
    }
    CTF_int::cdealloc(d);

    U = Matrix<dtype>(descu, u, layout_order, (*(this->wrld)));

    CTF_int::cdealloc(A);
    CTF_int::cdealloc(u);
    free(desca);
    free(descu);
    t_eigh.stop();
  }

  template<typename dtype>
  std::vector<CTF::Vector<dtype>*> Matrix<dtype>::to_vector_batch(){
    std::vector<CTF_int::tensor*> subtsrs = this->partition_last_mode_implicit();
    std::vector<CTF::Vector<dtype>*> subvecs;
    for (int64_t i=0; i<(int64_t)subtsrs.size(); i++){
      subvecs.push_back(new CTF::Vector<dtype>(*subtsrs[i]));
      delete subtsrs[i];
    }
    return subvecs;
  }

}

