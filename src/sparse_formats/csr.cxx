#include "csr.h"
#include "../contraction/ctr_comm.h"
#include "../shared/util.h"

#define ALIGN 256

namespace CTF_int {
  int64_t get_csr_size(int64_t nnz, int nrow_, int val_size){
    int64_t offset = 4*sizeof(int64_t);
    if (offset % ALIGN != 0) offset += ALIGN-(offset%ALIGN);
    offset += nnz*val_size;
    if (offset % ALIGN != 0) offset += ALIGN-(offset%ALIGN);
    offset += (nrow_+1)*sizeof(int);
    if (offset % ALIGN != 0) offset += ALIGN-(offset%ALIGN);
    offset += sizeof(int)*nnz;
    if (offset % ALIGN != 0) offset += ALIGN-(offset%ALIGN);
    if (offset % 1024 != 0) offset += 1024-(offset%1024);
    return offset;
  }

  CSR_Matrix::CSR_Matrix(int64_t nnz, int nrow_, int ncol, accumulatable const * sr){
    ASSERT(ALIGN >= 16);
    int64_t size = get_csr_size(nnz, nrow_, sr->el_size);
    all_data = (char*)alloc(size);
    ((int64_t*)all_data)[0] = nnz;
    ((int64_t*)all_data)[1] = sr->el_size;
    ((int64_t*)all_data)[2] = (int64_t)nrow_;
    ((int64_t*)all_data)[3] = ncol;
    sr->init_shell(nnz,this->vals());
  }

  CSR_Matrix::CSR_Matrix(char * all_data_){
    ASSERT(ALIGN >= 16);
    all_data = all_data_;
  }

  CSR_Matrix::CSR_Matrix(COO_Matrix const & coom, int nrow_, int ncol, algstrct const * sr, char * data, bool init_data){
    ASSERT(ALIGN >= 16);
    int64_t nz = coom.nnz(); 
    int64_t v_sz = coom.val_size(); 
    int const * coo_rs = coom.rows();
    int const * coo_cs = coom.cols();
    char const * vs = coom.vals();
    /*if (nz >= 2){
      printf("herecsr\n");
      sr->print(vs);
      sr->print(vs+v_sz);
    }*/

    int64_t size = get_csr_size(nz, nrow_, v_sz);
    if (data == NULL)
      all_data = (char*)alloc(size);
    else
      all_data = data;
    
    ((int64_t*)all_data)[0] = nz;
    ((int64_t*)all_data)[1] = v_sz;
    ((int64_t*)all_data)[2] = (int64_t)nrow_;
    ((int64_t*)all_data)[3] = ncol;

    char * csr_vs = vals();
    int * csr_ja = JA();
    int * csr_ia = IA();

    if (init_data){
      sr->init_shell(nz, csr_vs);
    }
    //memcpy(csr_vs, vs, nz*v_sz);
    //memset(csr_ja

    sr->coo_to_csr(nz, nrow_, csr_vs, csr_ja, csr_ia, vs, coo_rs, coo_cs);
/*    for (int i=0; i<nrow_; i++){
      printf("csr_ja[%d] = %d\n",i,csr_ja[i]);
    }
    for (int i=0; i<nz; i++){
      printf("csr_ia[%d] = %d\n",i,csr_ia[i]);
    }*/
    
  }

  int64_t CSR_Matrix::nnz() const {
    return ((int64_t*)all_data)[0];
  }

  int CSR_Matrix::val_size() const {
    return ((int64_t*)all_data)[1];
  }


  int64_t CSR_Matrix::size() const {
    return get_csr_size(nnz(),nrow(),val_size());
  }
  
  int CSR_Matrix::nrow() const {
    return ((int64_t*)all_data)[2];
  }
  
  int CSR_Matrix::ncol() const {
    return ((int64_t*)all_data)[3];
  }
  
  char * CSR_Matrix::vals() const {
    int offset = 4*sizeof(int64_t);
    if (offset % ALIGN != 0) offset += ALIGN-(offset%ALIGN);
    return all_data + offset;
  }

  int * CSR_Matrix::IA() const {
    int64_t n = this->nnz();
    int v_sz = this->val_size();

    int offset = 4*sizeof(int64_t);
    if (offset % ALIGN != 0) offset += ALIGN-(offset%ALIGN);
    offset += n*v_sz;
    if (offset % ALIGN != 0) offset += ALIGN-(offset%ALIGN);

    return (int*)(all_data + offset);
  } 

  int * CSR_Matrix::JA() const {
    int64_t n = this->nnz();
    int64_t nr = this->nrow();
    int v_sz = this->val_size();

    int offset = 4*sizeof(int64_t);
    if (offset % ALIGN != 0) offset += ALIGN-(offset%ALIGN);
    offset += n*v_sz;
    if (offset % ALIGN != 0) offset += ALIGN-(offset%ALIGN);
    offset += (nr+1)*sizeof(int);
    if (offset % ALIGN != 0) offset += ALIGN-(offset%ALIGN);
    //return (int*)(all_data + n*v_sz+(nr+1)*sizeof(int)+3*sizeof(int64_t));
    return (int*)(all_data + offset);
  } 

  void CSR_Matrix::csrmm(char const * A, algstrct const * sr_A, int m, int n, int k, char const * alpha, char const * B, algstrct const * sr_B, char const * beta, char * C, algstrct const * sr_C, bivar_function const * func, bool do_offload){
    if (func != NULL && func->has_off_gemm && do_offload){
      assert(sr_C->isequal(beta, sr_C->mulid()));
      assert(alpha == NULL || sr_C->isequal(alpha, sr_C->mulid()));
      func->coffload_csrmm(m,n,k,A,B,C);
    } else {
      CSR_Matrix cA((char*)A);
      int64_t nz = cA.nnz(); 
      int const * ja = cA.JA();
      int const * ia = cA.IA();
      char const * vs = cA.vals();
      if (func != NULL){
        assert(sr_C->isequal(beta, sr_C->mulid()));
        assert(alpha == NULL || sr_C->isequal(alpha, sr_C->mulid()));
        func->fcsrmm(m,n,k,vs,ja,ia,nz,B,C,sr_C);
      } else {
        ASSERT(sr_B->el_size == sr_A->el_size);
        ASSERT(sr_C->el_size == sr_A->el_size);
        assert(!do_offload);
        sr_C->csrmm(m,n,k,alpha,vs,ja,ia,nz,B,beta,C,func);
      }
    }
  }

  void CSR_Matrix::csrmultd(char const * A, algstrct const * sr_A, int m, int n, int k, char const * alpha, char const * B, algstrct const * sr_B, char const * beta, char * C, algstrct const * sr_C, bivar_function const * func, bool do_offload){
    if (func != NULL && func->has_off_gemm && do_offload){
      assert(0);
      assert(sr_C->isequal(beta, sr_C->mulid()));
      assert(alpha == NULL || sr_C->isequal(alpha, sr_C->mulid()));
    } else {
      CSR_Matrix cA((char*)A);
      int64_t nzA = cA.nnz(); 
      int const * jA = cA.JA();
      int const * iA = cA.IA();
      char const * vsA = cA.vals();
      CSR_Matrix cB((char*)B);
      int64_t nzB = cB.nnz(); 
      int const * jB = cB.JA();
      int const * iB = cB.IA();
      char const * vsB = cB.vals();
      if (func != NULL){
        assert(sr_C->isequal(beta, sr_C->mulid()));
        assert(alpha == NULL || sr_C->isequal(alpha, sr_C->mulid()));
        func->fcsrmultd(m,n,k,vsA,jA,iA,nzA,vsB,jB,iB,nzB,C,sr_C);
      } else {
        ASSERT(sr_B->el_size == sr_A->el_size);
        ASSERT(sr_C->el_size == sr_A->el_size);
        assert(!do_offload);
        sr_C->csrmultd(m,n,k,alpha,vsA,jA,iA,nzA,vsB,jB,iB,nzB,beta,C);
      }
    }

  }

  void CSR_Matrix::csrmultcsr(char const * A, algstrct const * sr_A, int m, int n, int k, char const * alpha, char const * B, algstrct const * sr_B, char const * beta, char *& C, algstrct const * sr_C, bivar_function const * func, bool do_offload){
    if (func != NULL && func->has_off_gemm && do_offload){
      assert(0);
      assert(sr_C->isequal(beta, sr_C->mulid()));
      assert(alpha == NULL || sr_C->isequal(alpha, sr_C->mulid()));
    } else {
      CSR_Matrix cA((char*)A);
      int64_t nzA = cA.nnz(); 
      int const * jA = cA.JA();
      int const * iA = cA.IA();
      char const * vsA = cA.vals();
      CSR_Matrix cB((char*)B);
      int64_t nzB = cB.nnz(); 
      int const * jB = cB.JA();
      int const * iB = cB.IA();
      char const * vsB = cB.vals();
      if (func != NULL){
        assert(sr_C->isequal(beta, sr_C->mulid()));
        assert(alpha == NULL || sr_C->isequal(alpha, sr_C->mulid()));
        func->fcsrmultcsr(m,n,k,vsA,jA,iA,nzA,vsB,jB,iB,nzB,C,sr_C);
      } else {
        ASSERT(sr_B->el_size == sr_A->el_size);
        ASSERT(sr_C->el_size == sr_A->el_size);
        assert(!do_offload);
        sr_C->csrmultcsr(m,n,k,alpha,vsA,jA,iA,nzA,vsB,jB,iB,nzB,beta,C);
      }
    }


  }

  void CSR_Matrix::partition(int s, char ** parts_buffer, sparse_matrix ** parts){
    int part_nnz[s], part_nrows[s];
    int m = nrow();
    int v_sz = val_size();
    char const * org_vals = vals();
    int const * org_ia = IA();
    int const * org_ja = JA();
    for (int i=0; i<s; i++){
      part_nnz[i] = 0;
      part_nrows[i] = 0;
    }
    for (int i=0; i<m; i++){
      part_nrows[i%s]++;
      part_nnz[i%s]+=org_ia[i+1]-org_ia[i];
    }
    int64_t tot_sz = 0;
    for (int i=0; i<s; i++){
      tot_sz += get_csr_size(part_nnz[i], part_nrows[i], v_sz);
    }
    alloc_ptr(tot_sz, (void**)parts_buffer);
    char * part_data = *parts_buffer;
    for (int i=0; i<s; i++){
      ((int64_t*)part_data)[0] = part_nnz[i];
      ((int64_t*)part_data)[1] = v_sz;
      ((int64_t*)part_data)[2] = part_nrows[i];
      ((int64_t*)part_data)[3] = ncol();
      CSR_Matrix * mat = new CSR_Matrix(part_data);
      parts[i] = mat;
      char * pvals = mat->vals();
      int * pja = mat->JA();
      int * pia = mat->IA();
      pia[0] = 1;
      for (int j=i, k=0; j<m; j+=s, k++){
        memcpy(pvals+(pia[k]-1)*v_sz, org_vals+(org_ia[j]-1)*v_sz, (org_ia[j+1]-org_ia[j])*v_sz);
        memcpy(pja+(pia[k]-1), org_ja+(org_ia[j]-1), (org_ia[j+1]-org_ia[j])*sizeof(int));
        pia[k+1] = pia[k]+org_ia[j+1]-org_ia[j];
      }
      part_data += get_csr_size(part_nnz[i], part_nrows[i], v_sz);
    }
  }
      
  void CSR_Matrix::assemble(char * const * smnds, int s){
    CSR_Matrix * csrs[s];
    int64_t tot_nnz=0, tot_nrow=0;
    for (int i=0; i<s; i++){
      csrs[i] = new CSR_Matrix(smnds[i]);
      tot_nnz += csrs[i]->nnz();
      tot_nrow += csrs[i]->nrow();
    }
    int64_t v_sz = csrs[0]->val_size();
    int64_t tot_ncol = csrs[0]->ncol();
    all_data = (char*)alloc(get_csr_size(tot_nnz, tot_nrow, v_sz));
    ((int64_t*)all_data)[0] = tot_nnz;
    ((int64_t*)all_data)[1] = v_sz;
    ((int64_t*)all_data)[2] = tot_nrow;
    ((int64_t*)all_data)[3] = tot_ncol;
    
    char * csr_vs = vals();
    int * csr_ja = JA();
    int * csr_ia = IA();

    csr_ia[0] = 1;

    for (int i=0; i<tot_nrow; i++){
      int ipart = i%s;
      int const * pja = csrs[ipart]->JA();
      int const * pia = csrs[ipart]->IA();
      int i_nnz = pia[i/s+1]-pia[i/s];
      memcpy(csr_vs+(csr_ia[i]-1)*v_sz,
             csrs[ipart]->vals()+(pia[i/s]-1)*v_sz,
             i_nnz*v_sz);
      memcpy(csr_ja+(csr_ia[i]-1),
             pja+(pia[i/s]-1),
             i_nnz*sizeof(int));
      csr_ia[i+1] = csr_ia[i]+i_nnz;
    }
    for (int i=0; i<s; i++){
      delete csrs[i];
    }
  }

  void CSR_Matrix::print(algstrct const * sr){
    char * csr_vs = vals();
    int * csr_ja = JA();
    int * csr_ia = IA();
    int irow= 0;
    int v_sz = val_size();
    int64_t nz = nnz();
    printf("CSR Matrix has %ld nonzeros %d rows %d cols\n", nz, nrow(), ncol());
    for (int64_t i=0; i<nz; i++){
      while (i>=csr_ia[irow+1]-1) irow++;
      printf("[%d,%d] ",irow,csr_ja[i]);
      sr->print(csr_vs+v_sz*i);
      printf("\n");
    }

  }

  void CSR_Matrix::compute_has_col(
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

  char * CSR_Matrix::csr_add(char * cA, char * cB, accumulatable const * adder){
    TAU_FSTART(csr_add);
    CSR_Matrix A(cA);
    CSR_Matrix B(cB);

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
    CSR_Matrix C(IC[nrow]-1, nrow, ncol, adder);
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
    TAU_FSTOP(csr_add);
    
    return C.all_data;
  }

}
