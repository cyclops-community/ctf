/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "common.h"
#include "../shared/blas_symbs.h"
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
  Matrix<dtype>::Matrix() : Tensor<dtype>() { }

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
    ASSERT(A.order == 2);
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
          this->sr->print((char*)&(data[i*ncol+j]));
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
//    printf("nrow = %d ncol = %d nmyr = %ld, nmyc = %ld mb = %d nb = %d pr = %d pc = %d\n",nrow,ncol,nmyr,nmyc,mb,nb,pr,pc);
    pairs = (Pair<dtype>*)CTF_int::alloc(sizeof(Pair<dtype>)*nmyr*nmyc);
    int cblk = (rank/pr+pc-csrc)%pc;
    for (int64_t i=0; i<nmyc;  i++){
      int rblk = (rank+pr-rsrc)%pr;
      for (int64_t j=0; j<nmyr;  j++){
        pairs[i*nmyr+j].k = (cblk*nb+(i%nb))*nrow+rblk*mb+(j%mb);
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
            memcpy(this->data+i*nrow*sizeof(dtype)/pr,(char*)(data_+i*nrow/pr), nrow*sizeof(dtype)/pr);
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
      CTF_int::cdealloc(pairs);
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
    if (mb==1 && nb==1 && nrow%pr==0 && ncol%pc==0 && rsrc==0 && csrc==0){
      if (this->edge_map[0].np == pr && this->edge_map[1].np == pc){
        if (lda == nrow/pc){
          memcpy((char*)data_, this->data, sizeof(dtype)*this->size);
        } else {
          for (int i=0; i<ncol/pc; i++){
            memcpy((char*)(data_+i*nrow/pr), this->data+i*nrow*sizeof(dtype)/pr, nrow*sizeof(dtype)/pr);
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
        }
      } else {
        for (int64_t i=0; i<nmyc; i++){
          for (int64_t j=0; j<nmyr; j++){
            data_[i*lda+j] = pairs[i*nmyr+j].d;
          }
        }
      }
      CTF_int::cdealloc(pairs);
    }
  }

  template<typename dtype>
  void Matrix<dtype>::read_mat(int const * desc,
                               dtype *     data_){
    int ictxt = desc[1];
    int pr, pc, ipr, ipc;
    CTF_BLAS::BLACS_GRIDINFO(&ictxt, &pr, &pc, &ipr, &ipc);
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

  

  template<typename dtype>
  Matrix<dtype>::Matrix(int const *               desc,
                        dtype const *             data_,
                        World &                   wrld_,
                        CTF_int::algstrct const & sr_,
                        char const *              name_,
                        int                       profile_)
    : Tensor<dtype>(2, false, CTF_int::int2(desc[2], desc[3]),  CTF_int::int2(NS, NS),
                           wrld_, sr_, name_, profile_) {
    nrow = desc[2];
    ncol = desc[3];
    symm = NS;
    int ictxt = desc[1];
    int pr, pc, ipr, ipc;
    CTF_BLAS::BLACS_GRIDINFO(&ictxt, &pr, &pc, &ipr, &ipc);
    IASSERT(ipr == wrld_.rank%pr);
    IASSERT(ipc == wrld_.rank/pr);
    IASSERT(pr*pc == wrld_.np);
    this->set_distribution("ij", Partition(2,CTF_int::int2(pr, pc))["ij"], Idx_Partition());
    write_mat(desc[4],desc[5],pr,pc,desc[6],desc[7],desc[8],data_);
  }


}
