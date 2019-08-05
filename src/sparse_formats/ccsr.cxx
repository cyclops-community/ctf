#include "ccsr.h"
#include "csr.h"
#include "../contraction/ctr_comm.h"
#include "../shared/util.h"

#define ALIGN 256

namespace CTF_int {
  int64_t get_ccsr_size(int64_t nnz, int64_t nnz_row, int val_size){
    int64_t offset = 5*sizeof(int64_t);
    if (offset % ALIGN != 0) offset += ALIGN-(offset%ALIGN);
    offset += nnz_row*sizeof(int64_t);
    if (offset % ALIGN != 0) offset += ALIGN-(offset%ALIGN);
    offset += nnz*val_size;
    if (offset % ALIGN != 0) offset += ALIGN-(offset%ALIGN);
    offset += (nnz_row+1)*sizeof(int);
    if (offset % ALIGN != 0) offset += ALIGN-(offset%ALIGN);
    offset += sizeof(int)*nnz;
    if (offset % ALIGN != 0) offset += ALIGN-(offset%ALIGN);
    return offset;
  }

  CCSR_Matrix::CCSR_Matrix(int64_t nnz, int64_t nnz_row, int64_t nrow_, int64_t ncol, accumulatable const * sr){
    ASSERT(ALIGN >= 16);
    int64_t size = get_ccsr_size(nnz, nnz_row, sr->el_size);
    all_data = (char*)alloc(size);
    ((int64_t*)all_data)[0] = nnz;
    ((int64_t*)all_data)[1] = sr->el_size;
    ((int64_t*)all_data)[2] = nrow_;
    ((int64_t*)all_data)[3] = ncol;
    ((int64_t*)all_data)[4] = nnz_row;
    sr->init_shell(nnz,this->vals());
  }

  CCSR_Matrix::CCSR_Matrix(char * all_data_){
    ASSERT(ALIGN >= 16);
    all_data = all_data_;
  }

  CCSR_Matrix::CCSR_Matrix(tCOO_Matrix<int64_t> const & coom, int64_t nrow_, int64_t ncol, algstrct const * sr, char * data, bool init_data){
    ASSERT(ALIGN >= 16);
    TAU_FSTART(ccsr_conv);
    int64_t nz = coom.nnz(); 
    int64_t v_sz = coom.val_size(); 
    int64_t const * coo_rs = coom.rows();
    int64_t const * coo_cs = coom.cols();
    char const * vs = coom.vals();
    /*if (nz >= 2){
      printf("hereccsr\n");
      sr->print(vs);
      sr->print(vs+v_sz);
    }*/
   
    int64_t * coo_rs_copy = (int64_t*)alloc(nz*sizeof(int64_t));
    memcpy(coo_rs_copy, coo_rs, nz*sizeof(int64_t)); 
    std::sort(coo_rs_copy, coo_rs_copy+nz);
    int64_t nnz_row = 0;
    if (nz > 0){
#ifdef USE_OMP
      #pragma omp parallel for reduction(+: nnz_row)
#endif
      for (int64_t i=1; i<nz; i++){
        nnz_row += (coo_rs_copy[i-1] != coo_rs_copy[i]);
      }
      nnz_row+=1;
    }

    int64_t size = get_ccsr_size(nz, nnz_row, v_sz);
    if (data == NULL)
      all_data = (char*)alloc(size);
    else
      all_data = data;
    
    ((int64_t*)all_data)[0] = nz;
    ((int64_t*)all_data)[1] = v_sz;
    ((int64_t*)all_data)[2] = (int64_t)nrow_;
    ((int64_t*)all_data)[3] = ncol;
    ((int64_t*)all_data)[4] = nnz_row;

    int64_t * row_enc = nnz_row_encoding();
    int64_t nnz_row_ctr = 0;
    if (nz > 0){
      row_enc[0] = coo_rs_copy[0];
      nnz_row_ctr++;
    }
    //FIXME add openmp
    for (int64_t i=1; i<nz; i++){
      if (coo_rs_copy[i-1] != coo_rs_copy[i]){
        row_enc[nnz_row_ctr] = coo_rs_copy[i];
        nnz_row_ctr++;
        //printf("row_enc[%d] = %d\n");
      }
    }
    cdealloc(coo_rs_copy);

    char * ccsr_vs = vals();
    int * ccsr_ja = JA();
    int * ccsr_ia = IA();

    if (init_data){
      sr->init_shell(nz, ccsr_vs);
    }
    //memcpy(ccsr_vs, vs, nz*v_sz);
    //memset(ccsr_ja

    TAU_FSTART(coo_to_ccsr);
    sr->coo_to_ccsr(nz, nnz_row, ccsr_vs, ccsr_ja, ccsr_ia, vs, coo_rs, coo_cs);
    TAU_FSTOP(coo_to_ccsr);
/*    for (int i=0; i<nrow_; i++){
      printf("ccsr_ja[%d] = %d\n",i,ccsr_ja[i]);
    }
    for (int i=0; i<nz; i++){
      printf("ccsr_ia[%d] = %d\n",i,ccsr_ia[i]);
    }*/
    TAU_FSTOP(ccsr_conv);
    
  }

  int64_t CCSR_Matrix::nnz() const {
    return ((int64_t*)all_data)[0];
  }

  int CCSR_Matrix::val_size() const {
    return ((int64_t*)all_data)[1];
  }

  int64_t CCSR_Matrix::size() const {
    return get_ccsr_size(nnz(),nnz_row(),val_size());
  }
  
  int64_t CCSR_Matrix::nrow() const {
    return ((int64_t*)all_data)[2];
  }
  
  int64_t CCSR_Matrix::ncol() const {
    return ((int64_t*)all_data)[3];
  }
   
  int64_t CCSR_Matrix::nnz_row() const {
    return ((int64_t*)all_data)[4];
  }

  int64_t * CCSR_Matrix::nnz_row_encoding() const {
    int64_t offset = 5*sizeof(int64_t);
    if (offset % ALIGN != 0) offset += ALIGN-(offset%ALIGN);
    return (int64_t*)(all_data + offset);
  }

  char * CCSR_Matrix::vals() const {
    char * ptr = (char*)this->nnz_row_encoding();
    int64_t offset = ptr-all_data; 
    offset += this->nnz_row()*sizeof(int64_t);
    if (offset % ALIGN != 0) offset += ALIGN-(offset%ALIGN);
    return all_data + offset;
  }

  int * CCSR_Matrix::IA() const {
    int64_t n = this->nnz();
    int v_sz = this->val_size();
    char * ptr = this->vals();
    int64_t offset = ptr-all_data; 
    offset += n*v_sz;
    if (offset % ALIGN != 0) offset += ALIGN-(offset%ALIGN);

    return (int*)(all_data + offset);
  } 

  int * CCSR_Matrix::JA() const {
    int64_t nr = this->nnz_row();
    char * ptr = (char*)this->IA();
    int64_t offset = ptr-all_data; 
    offset += (nr+1)*sizeof(int);
    if (offset % ALIGN != 0) offset += ALIGN-(offset%ALIGN);
    return (int*)(all_data + offset);
  } 

  void CCSR_Matrix::ccsrmm(char const * A, algstrct const * sr_A, int64_t m, int64_t n, int64_t k, char const * alpha, char const * B, algstrct const * sr_B, char const * beta, char *& C, algstrct const * sr_C, bivar_function const * func, bool do_offload){
    if (func != NULL && func->has_off_gemm && do_offload){
      assert(sr_C->isequal(beta, sr_C->mulid()));
      assert(alpha == NULL || sr_C->isequal(alpha, sr_C->mulid()));
      ASSERT(0);
    } else {
      CCSR_Matrix cA((char*)A);
      int64_t nz = cA.nnz(); 
      int64_t nnz_row = cA.nnz_row(); 
      int64_t const * row_enc = cA.nnz_row_encoding();
      int const * ja = cA.JA();
      int const * ia = cA.IA();
      char const * vs = cA.vals();
      if (func != NULL){
        assert(sr_C->isequal(beta, sr_C->mulid()));
        assert(alpha == NULL || sr_C->isequal(alpha, sr_C->mulid()));
        assert(0); // CCSR functionality with functions is not yet available
//        func->fccsrmm(m,n,k,vs,ja,ia,nz,B,C,sr_C);
      } else {
        ASSERT(sr_B->el_size == sr_A->el_size);
        ASSERT(sr_C->el_size == sr_A->el_size);
        assert(!do_offload);
        sr_C->ccsrmm(m,n,k,nnz_row,alpha,vs,ja,ia,row_enc,nz,B,beta,C,func);
      }
    }
  }

 /* void CCSR_Matrix::ccsrmultd(char const * A, algstrct const * sr_A, int m, int n, int k, char const * alpha, char const * B, algstrct const * sr_B, char const * beta, char * C, algstrct const * sr_C, bivar_function const * func, bool do_offload){
    if (func != NULL && func->has_off_gemm && do_offload){
      assert(0);
      assert(sr_C->isequal(beta, sr_C->mulid()));
      assert(alpha == NULL || sr_C->isequal(alpha, sr_C->mulid()));
    } else {
      CCSR_Matrix cA((char*)A);
      int64_t nzA = cA.nnz(); 
      int const * jA = cA.JA();
      int const * iA = cA.IA();
      char const * vsA = cA.vals();
      CCSR_Matrix cB((char*)B);
      int64_t nzB = cB.nnz(); 
      int const * jB = cB.JA();
      int const * iB = cB.IA();
      char const * vsB = cB.vals();
      if (func != NULL){
        assert(sr_C->isequal(beta, sr_C->mulid()));
        assert(alpha == NULL || sr_C->isequal(alpha, sr_C->mulid()));
        func->fccsrmultd(m,n,k,vsA,jA,iA,nzA,vsB,jB,iB,nzB,C,sr_C);
      } else {
        ASSERT(sr_B->el_size == sr_A->el_size);
        ASSERT(sr_C->el_size == sr_A->el_size);
        assert(!do_offload);
        sr_C->ccsrmultd(m,n,k,alpha,vsA,jA,iA,nzA,vsB,jB,iB,nzB,beta,C);
      }
    }

  }

  void CCSR_Matrix::ccsrmultccsr(char const * A, algstrct const * sr_A, int m, int n, int k, char const * alpha, char const * B, algstrct const * sr_B, char const * beta, char *& C, algstrct const * sr_C, bivar_function const * func, bool do_offload){
    if (func != NULL && func->has_off_gemm && do_offload){
      assert(0);
      assert(sr_C->isequal(beta, sr_C->mulid()));
      assert(alpha == NULL || sr_C->isequal(alpha, sr_C->mulid()));
    } else {
      CCSR_Matrix cA((char*)A);
      int64_t nzA = cA.nnz(); 
      int const * jA = cA.JA();
      int const * iA = cA.IA();
      char const * vsA = cA.vals();
      CCSR_Matrix cB((char*)B);
      int64_t nzB = cB.nnz(); 
      int const * jB = cB.JA();
      int const * iB = cB.IA();
      char const * vsB = cB.vals();
      if (func != NULL){
        assert(sr_C->isequal(beta, sr_C->mulid()));
        assert(alpha == NULL || sr_C->isequal(alpha, sr_C->mulid()));
        func->fccsrmultccsr(m,n,k,vsA,jA,iA,nzA,vsB,jB,iB,nzB,C,sr_C);
      } else {
        ASSERT(sr_B->el_size == sr_A->el_size);
        ASSERT(sr_C->el_size == sr_A->el_size);
        assert(!do_offload);
        sr_C->ccsrmultccsr(m,n,k,alpha,vsA,jA,iA,nzA,vsB,jB,iB,nzB,beta,C);
      }
    }


  }*/

  void CCSR_Matrix::partition(int s, char ** parts_buffer, sparse_matrix ** parts){
    TAU_FSTART(ccsr_partition);
    int part_nnz[s], part_nrows[s];
    int64_t nnz_r = nnz_row();
    int64_t nr = nrow();
    int v_sz = val_size();
    char * org_vals = vals();
    int64_t const * row_enc = nnz_row_encoding();
    int const * org_ia = IA();
    int const * org_ja = JA();
    for (int i=0; i<s; i++){
      part_nnz[i] = 0;
      part_nrows[i] = 0;
    }
    for (int64_t i=0; i<nnz_r; i++){
      int is = (row_enc[i]-1) % s;
      part_nrows[is]++;
      part_nnz[is]+=org_ia[i+1]-org_ia[i];
    }
    int64_t tot_sz = 0;
    for (int i=0; i<s; i++){
      tot_sz += get_ccsr_size(part_nnz[i], part_nrows[i], v_sz);
    }
    alloc_ptr(tot_sz, (void**)parts_buffer);
    char * part_data = *parts_buffer;
    for (int i=0; i<s; i++){
      ((int64_t*)part_data)[0] = part_nnz[i];
      ((int64_t*)part_data)[1] = v_sz;
      ((int64_t*)part_data)[2] = nr / s + (nr%s > i);
      ((int64_t*)part_data)[3] = ncol();
      ((int64_t*)part_data)[4] = part_nrows[i];
      CCSR_Matrix * mat = new CCSR_Matrix(part_data);
      parts[i] = mat;
      char * pvals = mat->vals();
      int64_t * prow_enc = mat->nnz_row_encoding();
      int * pja = mat->JA();
      int * pia = mat->IA();
      pia[0] = 1;
      int64_t k = 0;
      for (int64_t j=0; j<nnz_r; j++){
        if ((row_enc[j]-1) % s == i){
          prow_enc[k] = (row_enc[j]-1) / s + 1;
          memcpy(pvals+(pia[k]-1)*v_sz, org_vals+(org_ia[j]-1)*v_sz, (org_ia[j+1]-org_ia[j])*v_sz);
          memcpy(pja+(pia[k]-1), org_ja+(org_ia[j]-1), (org_ia[j+1]-org_ia[j])*sizeof(int));
          pia[k+1] = pia[k]+org_ia[j+1]-org_ia[j];
          k++;
        }
      }
      part_data += get_ccsr_size(part_nnz[i], part_nrows[i], v_sz);
    }
    TAU_FSTOP(ccsr_partition);
  }
      
  void CCSR_Matrix::assemble(char * const * smnds, int s){
    TAU_FSTART(ccsr_assemble);
    CCSR_Matrix * ccsrs = new CCSR_Matrix[s];
    int const ** pja = (int const **)malloc(sizeof(int*)*s);
    int const ** pia = (int const **)malloc(sizeof(int*)*s);
    int64_t const ** prow_enc = (int64_t const **)malloc(sizeof(int64_t*)*s);
    int64_t * pnnz_row = (int64_t*)malloc(sizeof(int64_t)*s);
    int64_t tot_nnz=0, tot_nnz_row=0, tot_nrow=0;
    for (int i=0; i<s; i++){
      ccsrs[i] = CCSR_Matrix(smnds[i]);
      tot_nnz += ccsrs[i].nnz();
      tot_nrow += ccsrs[i].nrow();
      tot_nnz_row += ccsrs[i].nnz_row();
      pnnz_row[i] = ccsrs[i].nnz_row();
      pja[i] = ccsrs[i].JA();
      pia[i] = ccsrs[i].IA();
      prow_enc[i] = ccsrs[i].nnz_row_encoding();
    }
    int64_t v_sz = ccsrs[0].val_size();
    int64_t tot_ncol = ccsrs[0].ncol();
    all_data = (char*)alloc(get_ccsr_size(tot_nnz, tot_nnz_row, v_sz));
    ((int64_t*)all_data)[0] = tot_nnz;
    ((int64_t*)all_data)[1] = v_sz;
    ((int64_t*)all_data)[2] = tot_nrow;
    ((int64_t*)all_data)[3] = tot_ncol;
    ((int64_t*)all_data)[4] = tot_nnz_row;
    
    char * ccsr_vs = vals();
    int64_t * row_enc = nnz_row_encoding();
    int * ccsr_ja = JA();
    int * ccsr_ia = IA();

    ccsr_ia[0] = 1;

    int * row_inds = (int*)malloc(sizeof(int)*s);
    std::fill(row_inds,row_inds+s,0);
    for (int64_t i=0; i<tot_nnz_row; i++){
      int64_t min_row = INT64_MAX;
      int ipart = -1;
      for (int j=0; j<s; j++){
        if (row_inds[j] < pnnz_row[j]){
          if (prow_enc[j][row_inds[j]] < min_row){
            min_row = prow_enc[j][row_inds[j]];
            ipart = j;
          }
        }
      }
      int ri = row_inds[ipart];
      row_inds[ipart]++;
      ASSERT(ipart != -1);
      int64_t i_nnz = pia[ipart][ri+1]-pia[ipart][ri];
      memcpy(ccsr_vs+(ccsr_ia[i]-1)*v_sz,
             ccsrs[ipart].vals()+(pia[ipart][ri]-1)*v_sz,
             i_nnz*v_sz);
      memcpy(ccsr_ja+(ccsr_ia[i]-1),
             pja[ipart]+(pia[ipart][ri]-1),
             i_nnz*sizeof(int));
      ccsr_ia[i+1] = ccsr_ia[i]+i_nnz;
      row_enc[i] = s*(prow_enc[ipart][ri]-1) + ipart + 1;
    }
    free(pja);
    free(pia);
    free(pnnz_row);
    free(prow_enc);
    free(row_inds);
    delete [] ccsrs;
    TAU_FSTOP(ccsr_assemble);
  }

  void CCSR_Matrix::print(algstrct const * sr){
    char * ccsr_vs = vals();
    int * ccsr_ja = JA();
    int * ccsr_ia = IA();
    int irow= 0;
    int v_sz = val_size();
    int64_t nz = nnz();
    int64_t nzr = nnz_row();
    int64_t * row_enc = nnz_row_encoding();
    printf("CCSR Matrix has %ld nonzeros %ld rows (%ld of them nonzero) %ld cols\n", nz, nrow(), nzr, ncol());
    for (int64_t i=0; i<nzr; i++){
      printf("row_enc[%ld] = %ld\n", i,row_enc[i]);
    }
    for (int64_t i=0; i<nz; i++){
      while (i>=ccsr_ia[irow+1]-1) irow++;
      printf("[%ld,%d] ",row_enc[irow],ccsr_ja[i]);
      sr->print(ccsr_vs+v_sz*i);
      printf("\n");
      assert(row_enc[irow] <= nrow());
      assert(ccsr_ja[i] <= ncol());
    }

  }

  //void CCSR_Matrix::compute_has_col(
  //                    int const * JA,
  //                    int const * IA,
  //                    int const * JB,
  //                    int const * IB,
  //                    int         i,
  //                    int *       has_col){
  //  for (int j=0; j<IA[i+1]-IA[i]; j++){
  //    int row_B = JA[IA[i]+j-1]-1;
  //    for (int k=0; k<IB[row_B+1]-IB[row_B]; k++){
  //      int idx_B = IB[row_B]+k-1;
  //      has_col[JB[idx_B]-1] = 1;
  //    }
  //  }
  //}

  char * CCSR_Matrix::ccsr_add(char * cA, char * cB, accumulatable const * adder){
    TAU_FSTART(ccsr_add);
    CCSR_Matrix A(cA);
    CCSR_Matrix B(cB);
    //A.print((algstrct*)adder);
    //B.print((algstrct*)adder);

    int el_size = A.val_size();

    char const * vA = A.vals();
    int const * JA = A.JA();
    int const * IA = A.IA();
    int64_t const * row_enc_A = A.nnz_row_encoding();
    int64_t nrow = A.nrow();
    int64_t nnz_row_A = A.nnz_row();
    char const * vB = B.vals();
    int const * JB = B.JA();
    int const * IB = B.IA();
    int64_t const * row_enc_B = B.nnz_row_encoding();
    int64_t nnz_row_B = B.nnz_row();
    ASSERT(nrow == B.nrow());
    int64_t ncol = std::max(A.ncol(),B.ncol());
    int64_t nnz_row = 0;
    int64_t innz_row_A = 0;
    int64_t innz_row_B = 0;
    while (innz_row_A<nnz_row_A && innz_row_B<nnz_row_B){
      if (row_enc_A[innz_row_A] == row_enc_B[innz_row_B]){
        innz_row_A++;
        innz_row_B++;
      } else if (row_enc_A[innz_row_A] < row_enc_B[innz_row_B]){
        innz_row_A++;
      } else {
        innz_row_B++;
      }
      nnz_row++;
    }
    nnz_row += (nnz_row_A-innz_row_A) + (nnz_row_B-innz_row_B);
    int64_t * row_enc = (int64_t*)alloc(sizeof(int64_t)*nnz_row);
    int * IC = (int*)alloc(sizeof(int)*(nnz_row+1));
    int * has_col = (int*)alloc(sizeof(int)*ncol);
    IC[0] = 1;
    innz_row_A = 0;
    innz_row_B = 0;
    for (int64_t i=0; i<nnz_row; i++){
      memset(has_col, 0, sizeof(int)*ncol);
      IC[i+1] = IC[i];
      //printf("i = %d nnz_row = %d, innz_row_A = %d nnz_row_A = %d, innz_row_B = %d nnz_row_B = %d\n",i,nnz_row,innz_row_A,nnz_row_A,innz_row_B,nnz_row_B);
      if (innz_row_A<nnz_row_A && innz_row_B<nnz_row_B &&  row_enc_A[innz_row_A] == row_enc_B[innz_row_B]){
        row_enc[i] = row_enc_A[innz_row_A];
        for (int j=0; j<IA[innz_row_A+1]-IA[innz_row_A]; j++){
          has_col[JA[IA[innz_row_A]+j-1]-1] = 1;
        }
        for (int j=0; j<IB[innz_row_B+1]-IB[innz_row_B]; j++){
          has_col[JB[IB[innz_row_B]+j-1]-1] = 1;
        }
        for (int j=0; j<ncol; j++){
          IC[i+1] += has_col[j];
        }
        innz_row_A++;
        innz_row_B++;
      } else if (innz_row_B>=nnz_row_B || (innz_row_A < nnz_row_A && row_enc_A[innz_row_A] < row_enc_B[innz_row_B])){
        row_enc[i] = row_enc_A[innz_row_A];
        IC[i+1] += IA[innz_row_A+1] - IA[innz_row_A];
        innz_row_A++;
      } else {
        row_enc[i] = row_enc_B[innz_row_B];
        IC[i+1] += IB[innz_row_B+1] - IB[innz_row_B];
        innz_row_B++;
      }
      //printf("i=%d,nnz_row =%d\n",i,nnz_row);
    }
    CCSR_Matrix C(IC[nnz_row]-1, nnz_row, nrow, ncol, adder);
    char * vC = C.vals();
    int * JC = C.JA();
    int64_t * row_enc_C = C.nnz_row_encoding();
    memcpy(C.IA(), IC, sizeof(int)*(nnz_row+1));
    cdealloc(IC);
    memcpy(row_enc_C, row_enc, sizeof(int64_t)*nnz_row);
    cdealloc(row_enc);
    IC = C.IA();
    int64_t * rev_col = (int64_t*)alloc(sizeof(int64_t)*ncol);
    innz_row_A = 0;
    innz_row_B = 0;
    for (int64_t i=0; i<nnz_row; i++){
      if (innz_row_A < nnz_row_A && innz_row_B < nnz_row_B && row_enc_A[innz_row_A] == row_enc_B[innz_row_B]){
        memset(has_col, 0, sizeof(int)*ncol);
        for (int j=0; j<IA[innz_row_A+1]-IA[innz_row_A]; j++){
          //assert(JA[IA[innz_row_A]+j-1]-1 < ncol);
          has_col[JA[IA[innz_row_A]+j-1]-1] = 1;
        }
        for (int j=0; j<IB[innz_row_B+1]-IB[innz_row_B]; j++){
          //assert(JB[IB[innz_row_B]+j-1]-1 < ncol);
          has_col[JB[IB[innz_row_B]+j-1]-1] = 1;
        }
        int vs = 0;
        for (int j=0; j<ncol; j++){
          if (has_col[j]){
            JC[IC[i]+vs-1] = j+1;
            //FIXME:: overflow?
            rev_col[j] = (IC[i]+vs-1)*el_size;
            vs++;
          }
        }
        memset(has_col, 0, sizeof(int)*ncol);
        for (int j=0; j<IA[innz_row_A+1]-IA[innz_row_A]; j++){
          int idx_A = IA[innz_row_A]+j-1;
          //assert(JA[idx_A]-1 < ncol);
          memcpy(vC+rev_col[JA[idx_A]-1],vA+idx_A*el_size,el_size);
          has_col[JA[idx_A]-1] = 1;
        }
        for (int j=0; j<IB[innz_row_B+1]-IB[innz_row_B]; j++){
          int idx_B = IB[innz_row_B]+j-1;
          //assert(JB[idx_B]-1 < ncol);
          if (has_col[JB[idx_B]-1])
            adder->accum(vB+idx_B*el_size,vC+rev_col[JB[idx_B]-1]);
          else
            memcpy(vC+rev_col[JB[idx_B]-1],vB+idx_B*el_size,el_size);
        }
        innz_row_A++;
        innz_row_B++;
      } else if (innz_row_B>=nnz_row_B || (innz_row_A < nnz_row_A && row_enc_A[innz_row_A] < row_enc_B[innz_row_B])){
        //assert(IC[i]-1+IA[innz_row_A+1] - IA[innz_row_A] <= C.nnz());
        memcpy(JC+IC[i]-1, JA+IA[innz_row_A]-1, sizeof(int)*(IA[innz_row_A+1] - IA[innz_row_A]));
        memcpy(vC+(IC[i]-1)*el_size, vA+(IA[innz_row_A]-1)*el_size, el_size*(IA[innz_row_A+1] - IA[innz_row_A]));
        innz_row_A++;
      } else {
        //assert(IC[i]-1+IB[innz_row_B+1] - IB[innz_row_B] <= C.nnz());
        memcpy(JC+IC[i]-1, JB+IB[innz_row_B]-1, sizeof(int)*(IB[innz_row_B+1] - IB[innz_row_B]));
        memcpy(vC+(IC[i]-1)*el_size, vB+(IB[innz_row_B]-1)*el_size, el_size*(IB[innz_row_B+1] - IB[innz_row_B]));
        innz_row_B++;
      }
    }
    cdealloc(has_col);
    cdealloc(rev_col);
    //printf("%d %d %d\n",C.IA()[0],C.IA()[1],C.IA()[2]);
    //printf("%d %d\n",C.JA()[0],C.JA()[1]);
    //printf("%lf %lf\n",((double*)C.vals())[0],((double*)C.vals())[1]);
    TAU_FSTOP(ccsr_add);
    return C.all_data;
  }

}
