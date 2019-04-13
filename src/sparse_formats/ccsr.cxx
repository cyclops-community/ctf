#include "ccsr.h"
#include "csr.h"
#include "../contraction/ctr_comm.h"
#include "../shared/util.h"

#define ALIGN 256

namespace CTF_int {
  int64_t get_ccsr_size(int64_t nnz, int64_t nnz_row, int val_size){
    int64_t offset = 5*sizeof(int64_t);
    if (offset % ALIGN != 0) offset += ALIGN-(offset%ALIGN);
    offset += nnz_row*sizeof(int);
    if (offset % ALIGN != 0) offset += ALIGN-(offset%ALIGN);
    offset += nnz*val_size;
    if (offset % ALIGN != 0) offset += ALIGN-(offset%ALIGN);
    offset += (nnz_row+1)*sizeof(int);
    if (offset % ALIGN != 0) offset += ALIGN-(offset%ALIGN);
    offset += sizeof(int)*nnz;
    if (offset % ALIGN != 0) offset += ALIGN-(offset%ALIGN);
    return offset;

    return get_csr_size(nnz_row,nnz_row,val_size)+get_csr_size(nnz_row,nnz,val_size);
  }

  CCSR_Matrix::CCSR_Matrix(int64_t nnz, int64_t nnz_row, int64_t nrow_, int64_t ncol, accumulatable const * sr){
    ASSERT(ALIGN >= 16);
    int64_t size = get_ccsr_size(nnz, nrow_, sr->el_size);
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

  CCSR_Matrix::CCSR_Matrix(COO_Matrix const & coom, int64_t nrow_, int64_t ncol, algstrct const * sr, char * data, bool init_data){
    ASSERT(ALIGN >= 16);
    int64_t nz = coom.nnz(); 
    int64_t v_sz = coom.val_size(); 
    int const * coo_rs = coom.rows();
    int const * coo_cs = coom.cols();
    char const * vs = coom.vals();
    /*if (nz >= 2){
      printf("hereccsr\n");
      sr->print(vs);
      sr->print(vs+v_sz);
    }*/
   
    int * coo_rs_copy = (int*)alloc(nz*sizeof(int));
    memcpy(coo_rs_copy, coo_rs, nz*sizeof(int)); 
    std::sort(coo_rs_copy, coo_rs_copy+nz);
    int64_t nnz_row = 0;
    if (nz > 0){
#ifdef USE_OMP
      #pragma omp parallel for shared(nnz_row) reduction(+: nnz_row)
#endif
      for (int i=1; i<nz; i++){
        nnz_row += (coo_rs_copy[i-1] != coo_rs_copy[i]);
      }
      int64_t nnz_row+=1;
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

    int * row_enc = nnz_row_encoding();
    int nnz_row_ctr = 0;
    if (nz > 0){
      row_enc[0] = coo_rs_copy[0];
      nnz_row_ctr++;
    }
    //FIXME add openmp
    for (int i=1; i<nz; i++){
      if (coo_rs_copy[i-1] != coo_rs_copy[i]){
        row_enc[nnz_row_ctr] = coo_rs_copy[i];
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

    sr->coo_to_ccsr(nz, nnz_row, ccsr_vs, ccsr_ja, ccsr_ia, vs, coo_rs, coo_cs);
/*    for (int i=0; i<nrow_; i++){
      printf("ccsr_ja[%d] = %d\n",i,ccsr_ja[i]);
    }
    for (int i=0; i<nz; i++){
      printf("ccsr_ia[%d] = %d\n",i,ccsr_ia[i]);
    }*/
    
  }

  int64_t CCSR_Matrix::nnz() const {
    return ((int64_t*)all_data)[0];
  }

  int CCSR_Matrix::val_size() const {
    return ((int64_t*)all_data)[1];
  }

  int64_t CCSR_Matrix::size() const {
    return get_ccsr_size(nnz(),nrow(),val_size());
  }
  
  int CCSR_Matrix::nrow() const {
    return ((int64_t*)all_data)[2];
  }
  
  int CCSR_Matrix::ncol() const {
    return ((int64_t*)all_data)[3];
  }
   
  int CCSR_Matrix::nnz_row() const {
    return ((int64_t*)all_data)[4];
  }

  int * CCSR_Matrix::nnz_row_encoding() const {
    int offset = 5*sizeof(int64_t);
    if (offset % ALIGN != 0) offset += ALIGN-(offset%ALIGN);
    return all_data + offset;
  }

  char * CCSR_Matrix::vals() const {
    char * ptr = this->nnz_row_encoding();
    int64_t offset = ptr-all_data; 
    offset += this->nnz_row()*sizeof(int);
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
    int64_t nr = this->nrow();
    ptr = (char*)this->IA();
    int64_t offset = ptr-all_data; 
    offset += (nr+1)*sizeof(int);
    if (offset % ALIGN != 0) offset += ALIGN-(offset%ALIGN);
    return (int*)(all_data + offset);
  } 

  void CCSR_Matrix::ccsrmm(char const * A, algstrct const * sr_A, int m, int n, int k, char const * alpha, char const * B, algstrct const * sr_B, char const * beta, char * C, algstrct const * sr_C, bivar_function const * func, bool do_offload){
    if (func != NULL && func->has_off_gemm && do_offload){
      assert(sr_C->isequal(beta, sr_C->mulid()));
      assert(alpha == NULL || sr_C->isequal(alpha, sr_C->mulid()));
      func->coffload_ccsrmm(m,n,k,A,B,C);
    } else {
      CCSR_Matrix cA((char*)A);
      int64_t nz = cA.nnz(); 
      int const * row_enc = cA.nnz_row_encoding();
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
        sr_C->ccsrmm(m,n,k,alpha,vs,row_enc,ja,ia,nz,B,beta,C,func);
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

  void CCSR_Matrix::partition(int s, char ** parts_buffer, CCSR_Matrix ** parts){
    int part_nnz[s], part_nrows[s];
    int nnz_r = nnz_row();
    int nr = nrow();
    int v_sz = val_size();
    char * org_vals = vals();
    int const * row_enc = cA.nnz_row_encoding();
    int const * org_ia = IA();
    int const * org_ja = JA();
    for (int i=0; i<s; i++){
      part_nnz[i] = 0;
      part_nrows[i] = 0;
    }
    for (int i=0; i<nnz_r; i++){
      int is = row_enc[i] % s;
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
      ((int64_t*)part_data)[2] = nr / s + (nr%s < s); //FIXME: check this
      ((int64_t*)part_data)[3] = ncol();
      ((int64_t*)part_data)[4] = part_nrows[i];
      parts[i] = new CCSR_Matrix(part_data);
      char * pvals = parts[i]->vals();
      int * prow_enc = parts[i]->nnz_row_encoding();
      int * pja = parts[i]->JA();
      int * pia = parts[i]->IA();
      pia[0] = 1;
      for (int j=i, k=0; j<m; j++, k++){
        if (row_enc[j] % s == i){
          prow_enc[k] = row_enc[j] / s;
          memcpy(pvals+(pia[k]-1)*v_sz, org_vals+(org_ia[j]-1)*v_sz, (org_ia[j+1]-org_ia[j])*v_sz);
          memcpy(pja+(pia[k]-1), org_ja+(org_ia[j]-1), (org_ia[j+1]-org_ia[j])*sizeof(int));
          pia[k+1] = pia[k]+org_ia[j+1]-org_ia[j];
        }
      }
      part_data += get_ccsr_size(part_nnz[i], part_nrows[i], v_sz);
    }
  }
      
  CCSR_Matrix::CCSR_Matrix(char * const * smnds, int s){
    CCSR_Matrix * ccsrs[s];
    int64_t tot_nnz=0, tot_nnz_row=0, tot_nrow=0;
    for (int i=0; i<s; i++){
      ccsrs[i] = new CCSR_Matrix(smnds[i]);
      tot_nnz += ccsrs[i]->nnz();
      tot_nrow += ccsrs[i]->nrow();
      tot_nnz_row += ccsrs[i]->nnz_row();
    }
    int64_t v_sz = ccsrs[0]->val_size();
    int64_t tot_ncol = ccsrs[0]->ncol();
    all_data = (char*)alloc(get_ccsr_size(tot_nnz, tot_nrow, v_sz));
    ((int64_t*)all_data)[0] = tot_nnz;
    ((int64_t*)all_data)[1] = v_sz;
    ((int64_t*)all_data)[2] = tot_nrow;
    ((int64_t*)all_data)[3] = tot_ncol;
    ((int64_t*)all_data)[4] = tot_nnz_row;
    
    char * ccsr_vs = vals();
    int * row_enc = nnz_row_encoding();
    int * ccsr_ja = JA();
    int * ccsr_ia = IA();

    ccsr_ia[0] = 1;

    for (int i=0; i<tot_nrow; i++){
      int ipart = i%s;
      int const * pja = ccsrs[ipart]->JA();
      int const * pia = ccsrs[ipart]->IA();
      int const * prow_enc = ccsrs[ipart]->nnz_row_encoding();
      int i_nnz = pia[i/s+1]-pia[i/s];
      memcpy(ccsr_vs+(ccsr_ia[i]-1)*v_sz,
             ccsrs[ipart]->vals()+(pia[i/s]-1)*v_sz,
             i_nnz*v_sz);
      memcpy(ccsr_ja+(ccsr_ia[i]-1),
             pja+(pia[i/s]-1),
             i_nnz*sizeof(int));
      ccsr_ia[i+1] = ccsr_ia[i]+i_nnz;
      row_enc[i] = prow_enc[...//FIXME: need to be smarter about merging nonzero rows, maybe out of order]
    }
    for (int i=0; i<s; i++){
      delete ccsrs[i];
    }
  }

  void CCSR_Matrix::print(algstrct const * sr){
    char * ccsr_vs = vals();
    int * ccsr_ja = JA();
    int * ccsr_ia = IA();
    int irow= 0;
    int v_sz = val_size();
    int64_t nz = nnz();
    printf("CCSR Matrix has %ld nonzeros %d rows %d cols\n", nz, nrow(), ncol());
    for (int64_t i=0; i<nz; i++){
      while (i>=ccsr_ia[irow+1]-1) irow++;
      printf("[%d,%d] ",irow,ccsr_ja[i]);
      sr->print(ccsr_vs+v_sz*i);
      printf("\n");
    }

  }

  void CCSR_Matrix::compute_has_col(
                      int const * JA,
                      int const * IA,
                      int const * JB,
                      int const * IB,
                      int         i,
                      int *       has_col){
    for (int j=0; j<IA[i+1]-IA[i]; j++){
      int row_B = JA[IA[i]+j-1]-1;
      for (int k=0; k<IB[row_B+1]-IB[row_B]; k++){
        int idx_B = IB[row_B]+k-1;
        has_col[JB[idx_B]-1] = 1;
      }
    }
  }

  char * CCSR_Matrix::ccsr_add(char * cA, char * cB, accumulatable const * adder){
    TAU_FSTART(ccsr_add);
    CCSR_Matrix A(cA);
    CCSR_Matrix B(cB);

    int el_size = A.val_size();

    char const * vA = A.vals();
    int const * JA = A.JA();
    int const * IA = A.IA();
    int nrow = A.nrow();
    char const * vB = B.vals();
    int const * JB = B.JA();
    int const * IB = B.IA();
    ASSERT(nrow == B.nrow());
    int ncol = std::max(A.ncol(),B.ncol());
    int * IC = (int*)alloc(sizeof(int)*(nrow+1));
    int * has_col = (int*)alloc(sizeof(int)*ncol);
    IC[0] = 1;
    for (int i=0; i<nrow; i++){
      memset(has_col, 0, sizeof(int)*ncol);
      IC[i+1] = IC[i];
      for (int j=0; j<IA[i+1]-IA[i]; j++){
        has_col[JA[IA[i]+j-1]-1] = 1;
      }
      for (int j=0; j<IB[i+1]-IB[i]; j++){
        has_col[JB[IB[i]+j-1]-1] = 1;
      }
      for (int j=0; j<ncol; j++){
        IC[i+1] += has_col[j];
      }
    }
    CCSR_Matrix C(IC[nrow]-1, nrow, ncol, adder);
    char * vC = C.vals();
    int * JC = C.JA();
    memcpy(C.IA(), IC, sizeof(int)*(nrow+1));
    cdealloc(IC);
    IC = C.IA();
    int64_t * rev_col = (int64_t*)alloc(sizeof(int64_t)*ncol);
    for (int i=0; i<nrow; i++){
      memset(has_col, 0, sizeof(int)*ncol);
      for (int j=0; j<IA[i+1]-IA[i]; j++){
        has_col[JA[IA[i]+j-1]-1] = 1;
      }
      for (int j=0; j<IB[i+1]-IB[i]; j++){
        has_col[JB[IB[i]+j-1]-1] = 1;
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
      for (int j=0; j<IA[i+1]-IA[i]; j++){
        int idx_A = IA[i]+j-1;
        memcpy(vC+rev_col[JA[idx_A]-1],vA+idx_A*el_size,el_size);
        has_col[JA[idx_A]-1] = 1;
      }
      for (int j=0; j<IB[i+1]-IB[i]; j++){
        int idx_B = IB[i]+j-1;
        if (has_col[JB[idx_B]-1])
          adder->accum(vB+idx_B*el_size,vC+rev_col[JB[idx_B]-1]);
        else
          memcpy(vC+rev_col[JB[idx_B]-1],vB+idx_B*el_size,el_size);
      }
    }
    cdealloc(has_col);
    cdealloc(rev_col);
    /*printf("nnz C is %ld\n", C.nnz());
    printf("%d %d %d\n",C.IA()[0],C.IA()[1],C.IA()[2]);
    printf("%d %d\n",C.JA()[0],C.JA()[1]);
    printf("%lf %lf\n",((double*)C.vals())[0],((double*)C.vals())[1]);*/
    TAU_FSTOP(ccsr_add);
    
    return C.all_data;
  }

}
