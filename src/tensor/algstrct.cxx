/*Copyright (c) 2014, Edgar Solomonik, all rights reserved.*/
#include "../shared/util.h"
#include "../shared/blas_symbs.h"
#include "untyped_tensor.h"
#include "algstrct.h"
#include "../sparse_formats/csr.h"

namespace CTF_int {
  LinModel<3> csrred_mdl(csrred_mdl_init,"csrred_mdl");
  LinModel<3> csrred_mdl_cst(csrred_mdl_cst_init,"csrred_mdl_cst");

  void sgemm(char           tA,
             char           tB,
             int            m,
             int            n,
             int            k,
             float          alpha,
             float  const * A,
             float  const * B,
             float          beta,
             float  *       C){
    int lda, lda_B, lda_C;
    lda_C = m;
    if (tA == 'n' || tA == 'N'){
      lda = m;
    } else {
      lda = k;
    }
    if (tB == 'n' || tB == 'N'){
      lda_B = k;
    } else {
      lda_B = n;
    }
    CTF_BLAS::SGEMM(&tA,&tB,&m,&n,&k,&alpha,A,&lda,B,&lda_B,&beta,C,&lda_C);
  }


  void cidgemm(char           tA,
               char           tB,
               int            m,
               int            n,
               int            k,
               double         alpha,
               double const * A,
               double const * B,
               double         beta,
               double *       C){
    int lda, lda_B, lda_C;
    lda_C = m;
    if (tA == 'n' || tA == 'N'){
      lda = m;
    } else {
      lda = k;
    }
    if (tB == 'n' || tB == 'N'){
      lda_B = k;
    } else {
      lda_B = n;
    }
    CTF_BLAS::DGEMM(&tA,&tB,&m,&n,&k,&alpha,A,&lda,B,&lda_B,&beta,C,&lda_C);
  }

  void cgemm(char                        tA,
             char                        tB,
             int                         m,
             int                         n,
             int                         k,
             std::complex<float>         alpha,
             std::complex<float> const * A,
             std::complex<float> const * B,
             std::complex<float>         beta,
             std::complex<float> *       C){
    int lda, lda_B, lda_C;
    lda_C = m;
    if (tA == 'n' || tA == 'N'){
      lda = m;
    } else {
      lda = k;
    }
    if (tB == 'n' || tB == 'N'){
      lda_B = k;
    } else {
      lda_B = n;
    }
    CTF_BLAS::CGEMM(&tA,&tB,&m,&n,&k,&alpha,A,&lda,B,&lda_B,&beta,C,&lda_C);
  }


  void zgemm(char                         tA,
             char                         tB,
             int                          m,
             int                          n,
             int                          k,
             std::complex<double>         alpha,
             std::complex<double> const * A,
             std::complex<double> const * B,
             std::complex<double>         beta,
             std::complex<double> *       C){
    int lda, lda_B, lda_C;
    lda_C = m;
    if (tA == 'n' || tA == 'N'){
      lda = m;
    } else {
      lda = k;
    }
    if (tB == 'n' || tB == 'N'){
      lda_B = k;
    } else {
      lda_B = n;
    }
    CTF_BLAS::ZGEMM(&tA,&tB,&m,&n,&k,&alpha,A,&lda,B,&lda_B,&beta,C,&lda_C);
  }
  algstrct::algstrct(int el_size_){
    el_size = el_size_;
    has_coo_ker = false;
  }


  MPI_Op algstrct::addmop() const {
    printf("CTF ERROR: no addition MPI_Op present for this algebraic structure\n");
    ASSERT(0);
    assert(0);
    return MPI_SUM;
  }

  MPI_Datatype algstrct::mdtype() const {
    printf("CTF ERROR: no MPI_Datatype present for this algebraic structure\n");
    ASSERT(0);
    assert(0);
    return MPI_CHAR;
  }



  char const * algstrct::addid() const {
    return NULL;
  }

  char const * algstrct::mulid() const {
    return NULL;
  }

  void algstrct::safeaddinv(char const * a, char *& b) const {
    printf("CTF ERROR: no additive inverse present for this algebraic structure\n");
    ASSERT(0);
    assert(0);
  }

  void algstrct::addinv(char const * a, char * b) const {
    printf("CTF ERROR: no additive inverse present for this algebraic structure\n");
    ASSERT(0);
    assert(0);
  }

  void algstrct::add(char const * a, char const * b, char * c) const {
    printf("CTF ERROR: no addition operation present for this algebraic structure\n");
    ASSERT(0);
    assert(0);
  }

  void algstrct::accum(char const * a, char * b) const {
    this->add(a, b, b);
  }


  void algstrct::mul(char const * a, char const * b, char * c) const {
    printf("CTF ERROR: no multiplication operation present for this algebraic structure\n");
    ASSERT(0);
    assert(0);
  }

  void algstrct::safemul(char const * a, char const * b, char *& c) const {
    printf("CTF ERROR: no multiplication operation present for this algebraic structure\n");
    ASSERT(0);
    assert(0);
  }

  void algstrct::min(char const * a, char const * b, char * c) const {
    printf("CTF ERROR: no min operation present for this algebraic structure\n");
    ASSERT(0);
    assert(0);
  }

  void algstrct::max(char const * a, char const * b, char * c) const {
    printf("CTF ERROR: no max operation present for this algebraic structure\n");
    ASSERT(0);
    assert(0);
  }

  void algstrct::cast_int(int64_t i, char * c) const {
    printf("CTF ERROR: integer scaling not possible for this algebraic structure\n");
    ASSERT(0);
    assert(0);
  }

  void algstrct::cast_double(double d, char * c) const {
    printf("CTF ERROR: double scaling not possible for this algebraic structure\n");
    ASSERT(0);
    assert(0);
  }

  double algstrct::cast_to_double(char const * c) const {
    printf("CTF ERROR: double cast not possible for this algebraic structure\n");
    ASSERT(0);
    assert(0);
    return 0.0;
  }

  int64_t algstrct::cast_to_int(char const * c) const {
    printf("CTF ERROR: int cast not possible for this algebraic structure\n");
    ASSERT(0);
    assert(0);
    return 0;
  }

  void algstrct::print(char const * a, FILE * fp) const {
    for (int i=0; i<el_size; i++){
      fprintf(fp,"%x",a[i]);
    }
  }

  void algstrct::min(char * c) const {
    printf("CTF ERROR: min limit not present for this algebraic structure\n");
    ASSERT(0);
    assert(0);
  }

  void algstrct::max(char * c) const {
    printf("CTF ERROR: max limit not present for this algebraic structure\n");
    ASSERT(0);
    assert(0);
  }

  void algstrct::scal(int          n,
                      char const * alpha,
                      char       * X,
                      int          incX)  const {
    if (isequal(alpha, addid())){
      if (incX == 1) set(X, addid(), n);
      else {
        for (int i=0; i<n; i++){
          copy(X+i*el_size, addid());
        }
      }
    } else {
      printf("CTF ERROR: scal not present for this algebraic structure\n");
      ASSERT(0);
      assert(0);
    }
  }

  void algstrct::axpy(int          n,
                      char const * alpha,
                      char const * X,
                      int          incX,
                      char       * Y,
                      int          incY)  const {
    printf("CTF ERROR: axpy not present for this algebraic structure\n");
    ASSERT(0);
    assert(0);
  }

   void algstrct::gemm(char         tA,
                       char         tB,
                       int          m,
                       int          n,
                       int          k,
                       char const * alpha,
                       char const * A,
                       char const * B,
                       char const * beta,
                       char *       C)  const {
    printf("CTF ERROR: gemm not present for this algebraic structure\n");
    ASSERT(0);
  }


   void algstrct::offload_gemm(char         tA,
                               char         tB,
                               int          m,
                               int          n,
                               int          k,
                               char const * alpha,
                               char const * A,
                               char const * B,
                               char const * beta,
                               char *       C) const {
    printf("CTF ERROR: offload gemm not present for this algebraic structure\n");
    ASSERT(0);
  }

  bool algstrct::is_offloadable() const {
    return false;
  }

  bool algstrct::isequal(char const * a, char const * b) const {
    if (a == NULL && b == NULL) return true;
    if (a == NULL || b == NULL) return false;
    bool iseq = true;
    for (int i=0; i<el_size; i++) {
      if (a[i] != b[i]) iseq = false;
    }
    return iseq;
  }

  void algstrct::coo_to_csr(int64_t nz, int nrow, char * csr_vs, int * csr_cs, int * csr_rs, char const * coo_vs, int const * coo_rs, int const * coo_cs) const {
    printf("CTF ERROR: cannot convert elements of this algebraic structure to CSR\n");
    ASSERT(0);
  }

  void algstrct::csr_to_coo(int64_t nz, int nrow, char const * csr_vs, int const * csr_ja, int const * csr_ia, char * coo_vs, int * coo_rs, int * coo_cs) const {
    printf("CTF ERROR: cannot convert elements of this algebraic structure to CSR\n");
    ASSERT(0);
  }


//  void algstrct::csr_add(int64_t m, int64_t n, char const * a, int const * ja, int const * ia, char const * b, int const * jb, int const * ib, char *& c, int *& jc, int *& ic){
  char * algstrct::csr_add(char * cA, char * cB) const {

    return CTF_int::CSR_Matrix::csr_add(cA, cB, this);
  }

  char * algstrct::csr_reduce(char * cA, int root, MPI_Comm cm) const {
    int r, p;
    MPI_Comm_rank(cm, &r);
    MPI_Comm_size(cm, &p);
    if (p==1) return cA;
    TAU_FSTART(csr_reduce);
    int s = 2;
    double t_st = MPI_Wtime();
    while (p%s != 0) s++;
    int sr = r%s;
    MPI_Comm scm;
    MPI_Comm rcm;
    MPI_Comm_split(cm, r/s, sr, &scm);
    MPI_Comm_split(cm, sr, r/s, &rcm);

    CSR_Matrix A(cA);
    int64_t sz_A = A.size();
    char * parts_buffer;
    CSR_Matrix ** parts = (CSR_Matrix**)alloc(sizeof(CSR_Matrix*)*s);
    A.partition(s, &parts_buffer, parts);
    //MPI_Request reqs[2*(s-1)];
    int rcv_szs[s];
    int snd_szs[s];
    int64_t tot_buf_size = 0;
    for (int i=0; i<s; i++){
      if (i==sr) snd_szs[i] = 0;
      else snd_szs[i] = parts[i]->size();
      tot_buf_size += snd_szs[i];
    }

    MPI_Alltoall(snd_szs, 1, MPI_INT, rcv_szs, 1, MPI_INT, scm);
    int64_t tot_rcv_sz = 0;
    for (int i=0; i<s; i++){
      //printf("i=%d/%d,rcv_szs[i]=%d\n",i,s,rcv_szs[i]);
      tot_rcv_sz += rcv_szs[i];
    }
    char * rcv_buf = (char*)alloc(tot_rcv_sz);

    char * smnds[s];
    int rcv_displs[s];
    int snd_displs[s];
    rcv_displs[0] = 0;
    for (int i=0; i<s; i++){
      if (i>0) rcv_displs[i] = rcv_szs[i-1]+rcv_displs[i-1];
      snd_displs[i] = parts[i]->all_data - parts[0]->all_data;
      if (i==sr) smnds[i] = parts[i]->all_data;
      else smnds[i] = rcv_buf + rcv_displs[i];
//      printf("parts[%d].all_data = %p\n",i,parts[i]->all_data);
  //    printf("snd_dipls[%d] = %d\n", i, snd_displs[i]);
//      printf("rcv_dipls[%d] = %d\n", i, rcv_displs[i]);
    }
    MPI_Alltoallv(parts[0]->all_data, snd_szs, snd_displs, MPI_CHAR, rcv_buf, rcv_szs, rcv_displs, MPI_CHAR, scm);
    for (int i=0; i<s; i++){
      delete parts[i]; //does not actually free buffer space
    }
    cdealloc(parts);
    /*  smnds[i] = (char*)alloc(rcv_szs[i]);
      int sbw = (r/phase - i + s-1)%s;
      int rbw = sbw + (r/(phase*s))*s + (r%phase);
      int rfw = sfw + (r/(phase*s))*s + (r%phase);
      char * rcv_data = (char*)alloc(rcv_szs[i]);
      smnds[i] = rcv_data;
      MPI_Isend(parts[sfw], snd_szs[i], MPI_CHAR, rfw, s+i, cm, reqs+i);
      MPI_Irecv(rcv_data, rcv_szs[i], MPI_CHAR, rbw, s+i, cm, reqs+s-1+i);
    }
    MPI_Status stats[2*(s-1)];
    MPI_Waitall(2*(s-1), reqs, stats);
    for (int i=1; i<s; i++){
      int sfw = (r/phase + i + s-1)%s;
      cdealloc(parts[sfw]);
    }
    cdealloc(parts);*/
    for (int z=1; z<s; z<<=1){
      for (int i=0; i<s-z; i+=2*z){
        char * csr_new = csr_add(smnds[i], smnds[i+z]);
        if ((smnds[i] < parts_buffer ||
             smnds[i] > parts_buffer+tot_buf_size) &&
            (smnds[i] < rcv_buf ||
             smnds[i] > rcv_buf+tot_rcv_sz))
          cdealloc(smnds[i]);
        if ((smnds[i+z] < parts_buffer ||
             smnds[i+z] > parts_buffer+tot_buf_size) &&
            (smnds[i+z] < rcv_buf ||
             smnds[i+z] > rcv_buf+tot_rcv_sz))
          cdealloc(smnds[i+z]);
        smnds[i] = csr_new;
      }
    }
    cdealloc(parts_buffer); //dealloc all parts
    cdealloc(rcv_buf);
    TAU_FSTOP(csr_reduce);
    char * red_sum = csr_reduce(smnds[0], root/s, rcm);
    TAU_FSTART(csr_reduce);
    if (smnds[0] != red_sum) cdealloc(smnds[0]);
    if (r/s == root/s){
      CSR_Matrix cf(red_sum);
      int sz = cf.size();
      int sroot = root%s;
      int cb_sizes[s];
      if (sroot == sr) sz = 0;
      MPI_Gather(&sz, 1, MPI_INT, cb_sizes, 1, MPI_INT, sroot, scm);
      int64_t tot_cb_size = 0;
      int cb_displs[s];
      if (sr == sroot){
        for (int i=0; i<s; i++){
          cb_displs[i] = tot_cb_size;
          tot_cb_size += cb_sizes[i];
        }
      }
      char * cb_bufs = (char*)alloc(tot_cb_size);
      MPI_Gatherv(red_sum, sz, MPI_CHAR, cb_bufs, cb_sizes, cb_displs, MPI_CHAR, sroot, scm);
      MPI_Comm_free(&scm);
      MPI_Comm_free(&rcm);
      if (sr == sroot){
        for (int i=0; i<s; i++){
          smnds[i] = cb_bufs + cb_displs[i];
          if (i==sr) smnds[i] = red_sum;
        }
        CSR_Matrix out(smnds, s);
        cdealloc(red_sum);
        cdealloc(cb_bufs);
        double t_end = MPI_Wtime() - t_st;
        double tps[] = {t_end, 1.0, log2((double)p), (double)sz_A};

        // note-quite-sure
        csrred_mdl.observe(tps);
        TAU_FSTOP(csr_reduce);
        return out.all_data;
      } else {
        cdealloc(red_sum);
        cdealloc(cb_bufs);
        TAU_FSTOP(csr_reduce);
        return NULL;
      }
    } else {
      MPI_Comm_free(&scm);
      MPI_Comm_free(&rcm);
      TAU_FSTOP(csr_reduce);
      return NULL;
    }
  }

  double algstrct::estimate_csr_red_time(int64_t msg_sz, CommData const * cdt) const {

    double ps[] = {1.0, log2((double)cdt->np), (double)msg_sz};
    return csrred_mdl.est_time(ps);
  }

  void algstrct::acc(char * b, char const * beta, char const * a, char const * alpha) const {
    char tmp[el_size];
    mul(b, beta, tmp);
    mul(a, alpha, b);
    add(b, tmp, b);
  }

  void algstrct::accmul(char * c, char const * a, char const * b, char const * alpha) const {
    char tmp[el_size];
    mul(a, b, tmp);
    mul(tmp, alpha, tmp);
    add(c, tmp, c);
  }


  void algstrct::safecopy(char *& a, char const * b) const {
    if (b == NULL){
      if (a != NULL) cdealloc(a);
      a = NULL;
    } else {
      if (a == NULL) a = (char*)alloc(el_size);
      memcpy(a, b, el_size);
    }
  }
  void algstrct::copy(char * a, char const * b) const {
    memcpy(a, b, el_size);
  }

  void algstrct::copy_pair(char * a, char const * b) const {
    memcpy(a, b, pair_size());
  }

  void algstrct::copy(char * a, char const * b, int64_t n) const {
    memcpy(a, b, el_size*n);
  }

  void algstrct::copy_pairs(char * a, char const * b, int64_t n) const {
    memcpy(a, b, pair_size()*n);
  }


  void algstrct::copy(int n, char const * a, int inc_a, char * b, int inc_b) const {
    switch (el_size) {
      case 4:
        CTF_BLAS::SCOPY(&n, (float const*)a, &inc_a, (float*)b, &inc_b);
        break;
      case 8:
        CTF_BLAS::DCOPY(&n, (double const*)a, &inc_a, (double*)b, &inc_b);
        break;
      case 16:
        CTF_BLAS::ZCOPY(&n, (std::complex<double> const*)a, &inc_a, (std::complex<double>*)b, &inc_b);
        break;
      default:
#ifdef USE_OMP
        #pragma omp parallel for
#endif
        for (int i=0; i<n; i++){
          copy(b+el_size*inc_b*i, a+el_size*inc_a*i);
        }
        break;
    }
  }

  void algstrct::copy(int64_t      m,
                      int64_t      n,
                      char const * a,
                      int64_t      lda_a,
                      char *       b,
                      int64_t      lda_b) const {
    if (lda_a == m && lda_b == n){
      memcpy(b,a,el_size*m*n);
    } else {
      for (int i=0; i<n; i++){
        memcpy(b+el_size*lda_b*i,a+el_size*lda_a*i,m*el_size);
      }
    }
  }

  void algstrct::copy(int64_t      m,
                      int64_t      n,
                      char const * a,
                      int64_t      lda_a,
                      char const * alpha,
                      char *       b,
                      int64_t      lda_b,
                      char const * beta) const {
    if (!isequal(beta, mulid())){
      if (isequal(beta, addid())){
        if (lda_b == 1)
          set(b, addid(), m*n);
        else {
          for (int i=0; i<n; i++){
            set(b+i*lda_b*el_size, addid(), m);
          }
        }
      } else {
        if (lda_b == m)
          scal(m*n, beta, b, 1);
        else {
          for (int i=0; i<n; i++){
            scal(m, beta, b+i*lda_b*el_size, 1);
          }
        }
      }
    }
    if (lda_a == m && lda_b == m){
      axpy(m*n, alpha, a, 1, b, 1);
    } else {
      for (int i=0; i<n; i++){
        axpy(m, alpha, a+el_size*lda_a*i, 1, b+el_size*lda_b*i, 1);
      }
    }
  }

  void algstrct::set(char * a, char const * b, int64_t n) const {
    switch (el_size) {
      case 4: {
          float * ia = (float*)a;
          float ib = *((float*)b);
          std::fill(ia, ia+n, ib);
        }
        break;
      case 8: {
          double * ia = (double*)a;
          double ib = *((double*)b);
          std::fill(ia, ia+n, ib);
        }
        break;
      case 16: {
          std::complex<double> * ia = (std::complex<double>*)a;
          std::complex<double> ib = *((std::complex<double>*)b);
          std::fill(ia, ia+n, ib);
        }
        break;
      default: {
          for (int i=0; i<n; i++) {
            memcpy(a+i*el_size, b, el_size);
          }
        }
        break;
    }
  }

  void algstrct::set_pair(char * a, int64_t key, char const * vb) const {
    memcpy(a, &key, sizeof(int64_t));
    memcpy(a+sizeof(int64_t), vb, el_size);
  }

  void algstrct::set_pairs(char * a, char const * b, int64_t n) const {
    for (int i=0; i<n; i++) {
      memcpy(a + i*(sizeof(int64_t)+el_size), b, (sizeof(int64_t)+el_size));
    }
  }

  int64_t algstrct::get_key(char const * a) const {
    return (int64_t)*a;
  }

  char const * algstrct::get_value(char const * a) const {
    return a+sizeof(int64_t);
  }


  void algstrct::coomm(int m, int n, int k, char const * alpha, char const * A, int const * rows_A, int const * cols_A, int64_t nnz_A, char const * B, char const * beta, char * C, bivar_function const * func) const {
    printf("CTF ERROR: coomm not present for this algebraic structure\n");
    ASSERT(0);
  }

  void algstrct::csrmm(int m, int n, int k, char const * alpha, char const * A, int const * JA, int const * IA, int64_t nnz_A, char const * B, char const * beta, char * C, bivar_function const * func) const {
    printf("CTF ERROR: csrmm not present for this algebraic structure\n");
    ASSERT(0);
  }

  void algstrct::csrmultd
                (int          m,
                 int          n,
                 int          k,
                 char const * alpha,
                 char const * A,
                 int const *  JA,
                 int const *  IA,
                 int64_t      nnz_A,
                 char const * B,
                 int const *  JB,
                 int const *  IB,
                 int64_t      nnz_B,
                 char const * beta,
                 char *       C) const {
    printf("CTF ERROR: csrmultd not present for this algebraic structure\n");
    ASSERT(0);
  }

  void algstrct::csrmultcsr
                (int          m,
                 int          n,
                 int          k,
                 char const * alpha,
                 char const * A,
                 int const *  JA,
                 int const *  IA,
                 int64_t      nnz_A,
                 char const * B,
                 int const *  JB,
                 int const *  IB,
                 int64_t      nnz_B,
                 char const * beta,
                 char *&      C_CSR) const {

    printf("CTF ERROR: csrmultcsr not present for this algebraic structure\n");
    ASSERT(0);
  }

  ConstPairIterator::ConstPairIterator(PairIterator const & pi){
    sr=pi.sr; ptr=pi.ptr;
  }

  ConstPairIterator::ConstPairIterator(algstrct const * sr_, char const * ptr_){
    sr=sr_; ptr=ptr_;
  }

  ConstPairIterator ConstPairIterator::operator[](int n) const {
    return ConstPairIterator(sr,ptr+(sr->el_size+sizeof(int64_t))*n);
  }

  int64_t ConstPairIterator::k() const {
    return ((int64_t*)ptr)[0];
  }

  char const * ConstPairIterator::d() const {
    return ptr+sizeof(int64_t);
  }

  void ConstPairIterator::read(char * buf, int64_t n) const {
    memcpy(buf, ptr, (sizeof(int64_t)+sr->el_size)*n);
  }

  void ConstPairIterator::read_val(char * buf) const {
    memcpy(buf, ptr+sizeof(int64_t), sr->el_size);
  }

  PairIterator::PairIterator(algstrct const * sr_, char * ptr_){
    sr=sr_;
    ptr=ptr_;
  }

  PairIterator PairIterator::operator[](int n) const {
    return PairIterator(sr,ptr+(sr->el_size+sizeof(int64_t))*n);
  }

  int64_t PairIterator::k() const {
    return ((int64_t*)ptr)[0];
  }

  char * PairIterator::d() const {
    return ptr+sizeof(int64_t);
  }

  void PairIterator::read(char * buf, int64_t n) const {
    memcpy(buf, ptr, (sizeof(int64_t)+sr->el_size)*n);
  }

  void PairIterator::read_val(char * buf) const {
    memcpy(buf, ptr+sizeof(int64_t), sr->el_size);
  }

  void PairIterator::write(char const * buf, int64_t n){
    memcpy(ptr, buf, (sizeof(int64_t)+sr->el_size)*n);
  }

  void PairIterator::write(PairIterator const iter, int64_t n){
    memcpy(ptr, iter.ptr, (sizeof(int64_t)+sr->el_size)*n);
  }

  void PairIterator::write(ConstPairIterator const iter, int64_t n){
    memcpy(ptr, iter.ptr, (sizeof(int64_t)+sr->el_size)*n);
  }

  void PairIterator::write_val(char const * buf){
    memcpy(ptr+sizeof(int64_t), buf, sr->el_size);
  }

  void PairIterator::write_key(int64_t key){
    ((int64_t*)ptr)[0] = key;
  }

  template<int l>
  struct CompPair{
    int64_t key;
    char data[l];
    bool operator < (const CompPair& other) const {
      return (key < other.key);
    }
  } __attribute__((packed));

  struct IntPair{
    int64_t key;
    int data;
    bool operator < (const IntPair& other) const {
      return (key < other.key);
    }
  } __attribute__((packed));

  struct ShortPair{
    int64_t key;
    short data;
    bool operator < (const ShortPair& other) const {
      return (key < other.key);
    }
  } __attribute__((packed));

  struct BoolPair{
    int64_t key;
    bool data;
    bool operator < (const BoolPair& other) const {
      return (key < other.key);
    }
  } __attribute__((packed));

/*  template struct CompPair<1>;
  template struct CompPair<2>;
  template struct CompPair<4>;*/
  template struct CompPair<8>;
  template struct CompPair<12>;
  template struct CompPair<16>;
  template struct CompPair<20>;
  template struct CompPair<24>;
  template struct CompPair<28>;
  template struct CompPair<32>;

  struct CompPtrPair{
    int64_t key;
    int64_t idx;
    bool operator < (const CompPtrPair& other) const {
      return (key < other.key);
    }
  };

  void PairIterator::sort(int64_t n){
    switch (sr->el_size){
      case 1:
        ASSERT(sizeof(BoolPair)==sr->pair_size());
        std::sort((BoolPair*)ptr,((BoolPair*)ptr)+n);
        break;
      case 2:
        ASSERT(sizeof(ShortPair)==sr->pair_size());
        std::sort((ShortPair*)ptr,((ShortPair*)ptr)+n);
        break;
      case 4:
        ASSERT(sizeof(IntPair)==sr->pair_size());
        std::sort((IntPair*)ptr,((IntPair*)ptr)+n);
        break;
      case 8:
        ASSERT(sizeof(CompPair<8>)==sr->pair_size());
        std::sort((CompPair<8>*)ptr,((CompPair<8>*)ptr)+n);
        break;
      case 12:
        ASSERT(sizeof(CompPair<12>)==sr->pair_size());
        std::sort((CompPair<12>*)ptr,((CompPair<12>*)ptr)+n);
        break;
      case 16:
        ASSERT(sizeof(CompPair<16>)==sr->pair_size());
        std::sort((CompPair<16>*)ptr,((CompPair<16>*)ptr)+n);
        break;
      case 20:
        ASSERT(sizeof(CompPair<20>)==sr->pair_size());
        std::sort((CompPair<20>*)ptr,((CompPair<20>*)ptr)+n);
        break;
      case 24:
        ASSERT(sizeof(CompPair<24>)==sr->pair_size());
        std::sort((CompPair<24>*)ptr,((CompPair<24>*)ptr)+n);
        break;
      case 28:
        ASSERT(sizeof(CompPair<28>)==sr->pair_size());
        std::sort((CompPair<28>*)ptr,((CompPair<28>*)ptr)+n);
        break;
      case 32:
        ASSERT(sizeof(CompPair<32>)==sr->pair_size());
        std::sort((CompPair<32>*)ptr,((CompPair<32>*)ptr)+n);
        break;
      default:
        //Causes a bogus uninitialized variable warning with GNU
        CompPtrPair ptr_pairs[n];
#ifdef USE_OMP
        #pragma omp parallel for
#endif
        for (int64_t i=0; i<n; i++){
          ptr_pairs[i].key = *(int64_t*)(ptr+i*(sizeof(int64_t)+sr->el_size));
          ptr_pairs[i].idx = i;
        }
        //FIXME :(
        char swap_buffer[(sizeof(int64_t)+sr->el_size)*n];

        memcpy(swap_buffer, ptr, (sizeof(int64_t)+sr->el_size)*n);

        std::sort(ptr_pairs, ptr_pairs+n);

#ifdef USE_OMP
        #pragma omp parallel for
#endif
        for (int64_t i=0; i<n; i++){
          memcpy(ptr+i*(sizeof(int64_t)+sr->el_size),
                 swap_buffer+ptr_pairs[i].idx*(sizeof(int64_t)+sr->el_size),
                 sizeof(int64_t)+sr->el_size);
        }
        break; //compiler warning here seems to be gcc bug
    }
  }

  void ConstPairIterator::permute(int64_t n, int order, int const * old_lens, int64_t const * new_lda, PairIterator wA){
    ConstPairIterator rA = * this;
#ifdef USE_OMP
    #pragma omp parallel for
#endif
    for (int64_t i=0; i<n; i++){
      int64_t k = rA[i].k();
      int64_t k_new = 0;
      for (int j=0; j<order; j++){
        k_new += (k%old_lens[j])*new_lda[j];
        k = k/old_lens[j];
      }
      ((int64_t*)wA[i].ptr)[0] = k_new;
      memcpy(wA[i].d(), rA[i].d(), sr->el_size);
      //printf("value %lf old key %ld new key %ld\n",((double*)wA[i].d())[0], rA[i].k(), wA[i].k());
    }


  }

  void ConstPairIterator::pin(int64_t n, int order, int const * lens, int const * divisor, PairIterator pi_new){
    TAU_FSTART(pin);
    ConstPairIterator pi = *this;
    int * div_lens;
    alloc_ptr(order*sizeof(int), (void**)&div_lens);
    for (int j=0; j<order; j++){
      div_lens[j] = (lens[j]/divisor[j] + (lens[j]%divisor[j] > 0));
//      printf("lens[%d] = %d divisor[%d] = %d div_lens[%d] = %d\n",j,lens[j],j,divisor[j],j,div_lens[j]);
    }
#ifdef USE_OMP
    #pragma omp parallel for
#endif
    for (int64_t i=0; i<n; i++){
      int64_t key = pi[i].k();
      int64_t new_key = 0;
      int64_t lda = 1;
//      printf("rank = %d, in key = %ld,  val = %lf\n",  phys_rank[0], save_key,  ((double*)pi_new[i].d())[0]);
      for (int j=0; j<order; j++){
//        printf("%d %ld %d\n",j,(key%lens[j])%divisor[j],phys_rank[j]);
        //ASSERT(((key%lens[j])%(divisor[j]/virt_dim[j])) == phys_rank[j]);
        new_key += ((key%lens[j])/divisor[j])*lda;
        lda *= div_lens[j];
        key = key/lens[j];
      }
      ((int64_t*)pi_new[i].ptr)[0] = new_key;
/*      if (i>0 && pi[i].k() > pi[i-1].k()){
        assert(pi_new[i].k() > pi_new[i-1].k());
      }*/
    }
    cdealloc(div_lens);
    TAU_FSTOP(pin);

  }

  void depin(algstrct const * sr, int order, int const * lens, int const * divisor, int nvirt, int const * virt_dim, int const * phys_rank, char * X, int64_t & new_nnz_B, int64_t * nnz_blk, char *& new_B, bool check_padding){

    TAU_FSTART(depin);

    int * div_lens;
    alloc_ptr(order*sizeof(int), (void**)&div_lens);
    for (int j=0; j<order; j++){
      div_lens[j] = (lens[j]/divisor[j] + (lens[j]%divisor[j] > 0));
//      printf("lens[%d] = %d divisor[%d] = %d div_lens[%d] = %d\n",j,lens[j],j,divisor[j],j,div_lens[j]);
    }
    if (check_padding){
      check_padding = false;
      for (int v=0; v<nvirt; v++){
        int vv = v;
        for (int j=0; j<order; j++){
          int vo = (vv%virt_dim[j])*(divisor[j]/virt_dim[j])+phys_rank[j];
          if (lens[j]%divisor[j] != 0 && vo >= lens[j]%divisor[j]){
            check_padding = true;
          }
          vv=vv/virt_dim[j];
        }
      }
    }
    int64_t * old_nnz_blk_B = nnz_blk;
    if (check_padding){
      //FIXME: uses a bit more memory then we will probably need, but probably worth not doing another round to count first
      new_B = (char*)alloc(sr->pair_size()*new_nnz_B);
      old_nnz_blk_B = (int64_t*)alloc(sizeof(int64_t)*nvirt);
      memcpy(old_nnz_blk_B, nnz_blk, sizeof(int64_t)*nvirt);
      memset(nnz_blk, 0, sizeof(int64_t)*nvirt);
    }

    int * virt_offset;
    alloc_ptr(order*sizeof(int), (void**)&virt_offset);
    int64_t nnz_off = 0;
    if (check_padding)
      new_nnz_B = 0;
    for (int v=0; v<nvirt; v++){
      //printf("%d %p new_B %p pin %p new_blk_nnz_B[%d] = %ld\n",A_or_B,this,new_B,nnz_blk,v,nnz_blk[v]);
      int vv=v;
      for (int j=0; j<order; j++){
        virt_offset[j] = (vv%virt_dim[j])*(divisor[j]/virt_dim[j])+phys_rank[j];
        vv=vv/virt_dim[j];
      }

      if (check_padding){
        int64_t new_nnz_blk = 0;
        ConstPairIterator vpi(sr, X+nnz_off*sr->pair_size());
        PairIterator vpi_new(sr, new_B+new_nnz_B*sr->pair_size());
        for (int64_t i=0; i<old_nnz_blk_B[v]; i++){
          int64_t key = vpi[i].k();
          int64_t new_key = 0;
          int64_t lda = 1;
          bool is_outside = false;
          for (int j=0; j<order; j++){
            //printf("%d %ld %ld %d\n",j,vpi[i].k(),((key%div_lens[j])*divisor[j]+virt_offset[j]),lens[j]);
            if (((key%div_lens[j])*divisor[j]+virt_offset[j])>=lens[j]){
              //printf("element is outside\n");
              is_outside = true;
            }
            new_key += ((key%div_lens[j])*divisor[j]+virt_offset[j])*lda;
            lda *= lens[j];
            key = key/div_lens[j];
          }
          if (!is_outside){
            //printf("key = %ld, new_key = %ld, val = %lf\n", vpi[i].k(), new_key, ((double*)vpi[i].d())[0]);
            ((int64_t*)vpi_new[new_nnz_blk].ptr)[0] = new_key;
            vpi_new[new_nnz_blk].write_val(vpi[i].d());
            new_nnz_blk++;
          }
        }
        nnz_blk[v] = new_nnz_blk;
        new_nnz_B += nnz_blk[v];
        nnz_off += old_nnz_blk_B[v];

      } else {
        ConstPairIterator vpi(sr, X+nnz_off*sr->pair_size());
        PairIterator vpi_new(sr, X+nnz_off*sr->pair_size());
  #ifdef USE_OMP
        #pragma omp parallel for
  #endif
        for (int64_t i=0; i<nnz_blk[v]; i++){
          int64_t key = vpi[i].k();
          //int64_t save_key = vpi[i].k();
          int64_t new_key = 0;
          int64_t lda = 1;
          for (int64_t j=0; j<order; j++){
            new_key += ((key%div_lens[j])*divisor[j]+virt_offset[j])*lda;
            lda *= lens[j];
            key = key/div_lens[j];
          }
          ((int64_t*)vpi_new[i].ptr)[0] = new_key;
          //printf(",,key = %ld, new_key = %ld, val = %lf\n",  save_key, new_key, ((double*)vpi_new[i].d())[0]);
        }
        nnz_off += nnz_blk[v];
      }
    }
    if (check_padding){
     cdealloc(old_nnz_blk_B);
    }
    cdealloc(virt_offset);
    cdealloc(div_lens);

    TAU_FSTOP(depin);
  }



  int64_t PairIterator::lower_bound(int64_t n, ConstPairIterator op){
    switch (sr->el_size){
      case 1:
        return std::lower_bound((CompPair<1>*)ptr,((CompPair<1>*)ptr)+n, ((CompPair<1>*)op.ptr)[0]) - (CompPair<1>*)ptr;
        break;
      case 2:
        return std::lower_bound((CompPair<2>*)ptr,((CompPair<2>*)ptr)+n, ((CompPair<2>*)op.ptr)[0]) - (CompPair<2>*)ptr;
        break;
      case 4:
        return std::lower_bound((CompPair<4>*)ptr,((CompPair<4>*)ptr)+n, ((CompPair<4>*)op.ptr)[0]) - (CompPair<4>*)ptr;
        break;
      case 8:
        return std::lower_bound((CompPair<8>*)ptr,((CompPair<8>*)ptr)+n, ((CompPair<8>*)op.ptr)[0]) - (CompPair<8>*)ptr;
        break;
      case 12:
        return std::lower_bound((CompPair<12>*)ptr,((CompPair<12>*)ptr)+n, ((CompPair<12>*)op.ptr)[0]) - (CompPair<12>*)ptr;
        break;
      case 16:
        return std::lower_bound((CompPair<16>*)ptr,((CompPair<16>*)ptr)+n, ((CompPair<16>*)op.ptr)[0]) - (CompPair<16>*)ptr;
        break;
      case 20:
        return std::lower_bound((CompPair<20>*)ptr,((CompPair<20>*)ptr)+n, ((CompPair<20>*)op.ptr)[0]) - (CompPair<20>*)ptr;
        break;
      case 24:
        return std::lower_bound((CompPair<24>*)ptr,((CompPair<24>*)ptr)+n, ((CompPair<24>*)op.ptr)[0]) - (CompPair<24>*)ptr;
        break;
      case 28:
        return std::lower_bound((CompPair<28>*)ptr,((CompPair<28>*)ptr)+n, ((CompPair<28>*)op.ptr)[0]) - (CompPair<28>*)ptr;
        break;
      case 32:
        return std::lower_bound((CompPair<32>*)ptr,((CompPair<32>*)ptr)+n, ((CompPair<32>*)op.ptr)[0]) - (CompPair<32>*)ptr;
        break;
      default: {
        int64_t keys[n];
#ifdef USE_OMP
        #pragma omp parallel for
#endif
        for (int64_t i=0; i<n; i++){
          keys[i] = (*this)[i].k();
        }
        return std::lower_bound(keys, keys+n, op.k())-keys;
        } break;
    }
  }

}
