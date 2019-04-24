#include "spr_seq_sum.h"
#include "../shared/iter_tsr.h"
#include "../shared/util.h"

namespace CTF_int{
  template<int idim>
  void spA_dnB_seq_sum_loop(char const *            alpha,
                            ConstPairIterator &     A,
                            int64_t &               size_A,
                            algstrct const *        sr_A,
                            char const *            beta,
                            char *&                 B,
                            algstrct const *        sr_B,
                            int                     order_B,
                            int64_t                 idx_B,
                            int64_t const *         edge_len_B,
                            int64_t const *         lda_B,
                            int const *             sym_B,
                            univar_function const * func){
    int64_t imax = edge_len_B[idim];
    if (sym_B[idim] != NS) imax = ((idx_B/lda_B[idim+1])%edge_len_B[idim+1])+1;

    for (int64_t i=0; i<imax; i++){
      //int nidx_B[order_B];
      //memcpy(nidx_B, idx_B, order_B*sizeof(int));
      spA_dnB_seq_sum_loop<idim-1>(alpha,A,size_A,sr_A,beta,B,sr_B,order_B,
                                   idx_B+i*lda_B[idim],
                                   edge_len_B, lda_B, sym_B, func);
    } 
  }

  template<>
  void spA_dnB_seq_sum_loop<0>(char const *            alpha,
                               ConstPairIterator &     A,
                               int64_t &               size_A,
                               algstrct const *        sr_A,
                               char const *            beta,
                               char *&                 B,
                               algstrct const *        sr_B,
                               int                     order_B,
                               int64_t                 idx_B,
                               int64_t const *         edge_len_B,
                               int64_t const *         lda_B,
                               int const *             sym_B,
                               univar_function const * func){
    
    int64_t imax = edge_len_B[0];
    if (sym_B[0] != NS) imax = ((idx_B/lda_B[0+1])%edge_len_B[0+1])+1;

    for (int64_t i=0; i<imax; i++){
      while (size_A > 0 && idx_B == A.k()){
        if (func == NULL){
          if (sr_A->isequal(alpha, sr_A->mulid()) || alpha == NULL){
            sr_B->add(A.d(), B, B);
          } else {
            char tmp[sr_A->el_size];
            sr_A->mul(A.d(), alpha, tmp);
            sr_B->add(tmp, B, B);
          }
        } else {
          if (sr_A->isequal(alpha, sr_A->mulid()) || alpha == NULL){
            func->acc_f(A.d(), B, sr_B);
          } else {
            char tmp[sr_A->el_size];
            sr_A->mul(A.d(), alpha, tmp);
            func->acc_f(tmp, B, sr_B);
          }
        }
        A = A[1];
        size_A--;
      }
      B += sr_B->el_size;
      idx_B++;
    }
  }

  template
  void spA_dnB_seq_sum_loop< MAX_ORD >
                                 (char const *            alpha,
                                  ConstPairIterator &     A,
                                  int64_t &               size_A,
                                  algstrct const *        sr_A,
                                  char const *            beta,
                                  char *&                 B,
                                  algstrct const *        sr_B,
                                  int                     order_B,
                                  int64_t                 idx_B,
                                  int64_t const *         edge_len_B,
                                  int64_t const *         lda_B,
                                  int const *             sym_B,
                                  univar_function const * func);
    


  void spA_dnB_seq_sum(char const *            alpha,
                       char const *            A,
                       int64_t                 size_A,
                       algstrct const *        sr_A,
                       char const *            beta,
                       char *                  B,
                       algstrct const *        sr_B,
                       int                     order_B,
                       int64_t const *         edge_len_B,
                       int const *             sym_B,
                       univar_function const * func){
    TAU_FSTART(spA_dnB_seq_sum);
    if (order_B == 0){
      if (!sr_B->isequal(beta, sr_B->mulid())){
        sr_B->mul(beta, B, B);
      }
      ConstPairIterator pi(sr_A, A);
      for (int64_t i=0; i<size_A; i++){
        char tmp_buf[sr_A->el_size];
        char const * tmp_ptr;
        if (alpha != NULL){
          sr_A->mul(alpha, pi[i].d(), tmp_buf);
          tmp_ptr = tmp_buf;
        } else tmp_ptr = pi[i].d();
        if (func != NULL){
          func->acc_f(tmp_ptr, B, sr_B); 
        } else {
          sr_B->add(tmp_ptr, B, B);
        }
      }
    } else {
      int64_t sz_B = sy_packed_size(order_B, edge_len_B, sym_B);
      if (!sr_B->isequal(beta, sr_B->mulid())){
        if (sr_B->isequal(beta, sr_B->addid()) || sr_B->isequal(beta, NULL))
          sr_B->set(B, sr_B->addid(), sz_B);
        else
          sr_B->scal(sz_B, beta, B, 1);
      }

      int64_t lda_B[order_B];
      for (int i=0; i<order_B; i++){
        if (i==0) lda_B[i] = 1;
        else      lda_B[i] = lda_B[i-1]*edge_len_B[i-1];
      }

      ASSERT(order_B<=MAX_ORD);

      ConstPairIterator pA(sr_A, A);
      int64_t idx = 0;
      SWITCH_ORD_CALL(spA_dnB_seq_sum_loop, order_B-1, alpha, pA, size_A, sr_A, beta, B, sr_B, order_B, idx, edge_len_B, lda_B, sym_B, func);
    }
    TAU_FSTOP(spA_dnB_seq_sum);
  }

  void dnA_spB_seq_sum(char const *            alpha,
                       char const *            A,
                       algstrct const *        sr_A,
                       int                     order_A,
                       int64_t const *         edge_len_A,
                       int const *             sym_A,
                       char const *            beta,
                       char const *            B,
                       int64_t                 size_B,
                       char *&                 new_B,
                       int64_t &               new_size_B,
                       algstrct const *        sr_B,
                       univar_function const * func){
    assert(0);
  }

  /**
   * \brief As pairs in a sparse A set to the 
   *         sparse set of elements defining the tensor,
   *         resulting in a set of size between nB and nB+nA
   * \param[in] sr_A algstrct defining data type of array
   * \param[in] nA number of elements in sparse tensor
   * \param[in] prs_A pairs of the sparse tensor
   * \param[in] beta scaling factor for data of the sparse tensor
   * \param[in] sr_B algstrct defining data type of array
   * \param[in] nB number of elements in the A set
   * \param[in] prs_B pairs of the A set
   * \param[in] alpha scaling factor for data of the A set
   * \param[out] nnew number of elements in resulting set
   * \param[out] pprs_new char array containing the pairs of the resulting set
   * \param[in] func NULL or pointer to a function to apply elementwise
   * \param[in] map_pfx how many times each element of A should be replicated
   */

  void spspsum(algstrct const *        sr_A,
               int64_t                 nA,
               ConstPairIterator       prs_A,
               char const *            beta,
               algstrct const *        sr_B,
               int64_t                 nB,
               ConstPairIterator       prs_B,
               char const *            alpha,
               int64_t &               nnew,
               char *&                 pprs_new,
               univar_function const * func,
               int64_t                 map_pfx){

    TAU_FSTART(spA_spB_seq_sum);
    // determine how many unique keys there are in prs_tsr and prs_Write
    nnew = nB;
    bool is_acc = (func != NULL && func->is_accumulator());
    TAU_FSTART(spA_spB_seq_sum_pre);
    for (int64_t t=0,ww=0; ww<nA*map_pfx; ww++){
      while (ww<nA*map_pfx){
        int64_t w = ww/map_pfx;
        int64_t mw = ww%map_pfx;
        if (t<nB && prs_B[t].k() < prs_A[w].k()*map_pfx+mw)
          t++;
        else if (t<nB && prs_B[t].k() == prs_A[w].k()*map_pfx+mw){
          t++;
          ww++;
        } else {
          //ASSERT(map_pfx == 1);
          if (!is_acc && (map_pfx != 1 || ww==0 || prs_A[ww-1].k() != prs_A[ww].k()))
            nnew++;
          ww++; w=ww;
        }
      }
    }
    TAU_FSTOP(spA_spB_seq_sum_pre);
//    printf("nB = %ld nA = %ld nnew = %ld\n",nB,nA,nnew); 
    pprs_new = sr_B->pair_alloc(nnew);
    PairIterator prs_new(sr_B, pprs_new);
    // each for loop computes one new value of prs_new 
    //    (multiple writes may contribute to it), 
    //    t, w, and n are incremented within
    // only incrementing r allows multiple writes of the same val
    int64_t n=0;
    for (int64_t t=0,ww=0; n<nnew; n++){
      /*if (n>0){ 
        printf("n=%ld\n",n-1);
        sr_A->print(prs_new[n-1].d());
      }*/
      int64_t w = ww/map_pfx;
      int64_t mw = ww%map_pfx;
      bool skip = 0;
      if (t<nB && (w==nA || prs_B[t].k() < prs_A[w].k()*map_pfx+mw)){
        sr_B->copy_pair(prs_new[n].ptr, prs_B[t].ptr);
        if (beta != NULL)
          sr_B->mul(prs_B[t].d(), beta, prs_new[n].d());
        t++;
      } else {
        /*if (t<nB)
          printf("%ld %ld\n",prs_B[t].k(), prs_A[w].k()*map_pfx+mw);*/
        if (t>=nB || prs_B[t].k() > prs_A[w].k()*map_pfx+mw){
          if (func == NULL){
            if (map_pfx == 1){
              sr_A->copy_pair(prs_new[n].ptr, prs_A[w].ptr);
            } else {
              ((int64_t*)prs_new[n].ptr)[0] = prs_A[w].k()*map_pfx+mw; 
              prs_new[n].write_val(prs_A[w].d());
            }
            if (alpha != NULL)
              sr_A->mul(prs_new[n].d(), alpha, prs_new[n].d());
          } else {
            //((int64_t*)prs_new[n].ptr)[0] = prs_A[w].k();
            if (!is_acc){
              ((int64_t*)prs_new[n].ptr)[0] = prs_A[w].k()*map_pfx+mw; 
              if (alpha != NULL){
                char a[sr_A->el_size];
                sr_A->mul(prs_A[w].d(), alpha, a);
                //if (sr_B->addid() != NULL){
                 // prs_new[n].write_val(sr_B->addid());
                  func->apply_f(a, prs_new[n].d());
               // } else {

                //}
              } else {
             //   prs_new[n].write_val(sr_B->addid());
                func->apply_f(prs_A[w].d(), prs_new[n].d());
              }
            } else { n--; skip=1; }
          }
          ww++;
        } else {
          char a[sr_A->el_size];
          char b[sr_B->el_size];
          if (alpha != NULL){
            sr_A->mul(prs_A[w].d(), alpha, a);
          } else {
            prs_A[w].read_val(a);
          }
          if (beta != NULL){
            sr_B->mul(prs_B[t].d(), beta, b);
          } else {
            prs_B[t].read_val(b);
          }
          if (func == NULL){ 
            sr_B->add(a, b, b);
          } else {
            func->acc_f(a, b, sr_B);
          }
          prs_new[n].write_val(b);
          ((int64_t*)(prs_new[n].ptr))[0] = prs_B[t].k();
          t++;
          ww++;
        }
        // accumulate any repeated key writes
        while (map_pfx == 1 && ww > 0 && ww<nA && prs_A[ww].k() == prs_A[ww-1].k()){
          if (!skip){
            if (alpha != NULL){
              char a[sr_A->el_size];
              sr_A->mul(prs_A[ww].d(), alpha, a);
              if (func == NULL)
                sr_B->add(prs_new[n].d(), a, prs_new[n].d());
              else
                func->acc_f(a, prs_new[n].d(), sr_B);
            } else {
              if (func == NULL)
                sr_B->add(prs_new[n].d(), prs_A[ww].d(), prs_new[n].d());
              else
                func->acc_f(prs_A[ww].d(), prs_new[n].d(), sr_B);
            }
          }
          ww++; w=ww;
        }
      }
      /*if (n>=0){
        printf("%ldth value is ", n);
        sr_B->print(prs_new[n].d());
        printf(" with key %ld\n",prs_new[n].k());
      }*/
    }
    ASSERT(n==nnew);
    TAU_FSTOP(spA_spB_seq_sum);
  }


  void spA_spB_seq_sum(char const *            alpha,
                       char const *            A,
                       int64_t                 size_A,
                       algstrct const *        sr_A,
                       char const *            beta,
                       char *                  B,
                       int64_t                 size_B,
                       char *&                 new_B,
                       int64_t &               new_size_B,
                       algstrct const *        sr_B,
                       univar_function const * func,
                       int64_t                 map_pfx){

/*      if (!sr_B->isequal(beta, sr_B->mulid())){
        printf("scaling B by 0\n");
        sr_B->scal(size_B, beta, B, 1);
      }*/
      spspsum(sr_A, size_A, ConstPairIterator(sr_A, A), beta,
              sr_B, size_B, ConstPairIterator(sr_B, B),alpha,
              new_size_B, new_B, func, map_pfx);
  }

}
