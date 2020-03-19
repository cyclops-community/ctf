namespace CTF {
 
  template<typename dtype> 
  Contract<dtype>::Contract() {
    pctr = nullptr;
  }
  
  template<typename dtype>
  Contract<dtype>::Contract(dtype                 alpha,
                            CTF_int::tensor&      A,
                            const char *          idx_A,
                            CTF_int::tensor&      B,
                            const char *          idx_B,
                            dtype                 beta,
                            CTF_int::tensor&      C,
                            const char *          idx_C) {
    if (A.wrld->cdt.cm != C.wrld->cdt.cm || B.wrld->cdt.cm != C.wrld->cdt.cm){
      printf("CTF ERROR: worlds of contracted tensors must match\n");
      IASSERT(0);
    }
    pctr = new CTF_int::contraction(&A, idx_A, &B, idx_B, (char const *)&alpha, &C, idx_C, (char const *)&beta, nullptr);
    pctr->prepare();
  }
  
  template<typename dtype>
  Contract<dtype>::Contract(dtype                 alpha,
                            CTF_int::tensor&      A,
                            const char *          idx_A,
                            CTF_int::tensor&      B,
                            const char *          idx_B,
                            dtype                 beta,
                            CTF_int::tensor&      C,
                            const char *          idx_C,
                            Bivar_Function<dtype> func) {
    if (A.wrld->cdt.cm != C.wrld->cdt.cm || B.wrld->cdt.cm != C.wrld->cdt.cm){
      printf("CTF ERROR: worlds of contracted tensors must match\n");
      IASSERT(0);
    }
    pctr = new CTF_int::contraction(&A, idx_A, &B, idx_B, (char const *)&alpha, &C, idx_C, (char const *)&beta, &func);
    pctr->prepare();
  }

  template<typename dtype>
  void Contract<dtype>::prepareA(CTF_int::tensor& A,
                                 const char *     idx_A) {
    IASSERT(pctr != nullptr);
    pctr->prepareT(&A, pctr->A, pctr->cm_topo_A, pctr->cm_edge_map_A, idx_A);
  }

  template<typename dtype>
  void Contract<dtype>::prepareB(CTF_int::tensor& B,
                                 const char *     idx_B) {
    IASSERT(pctr != nullptr);
    pctr->prepareT(&B, pctr->B, pctr->cm_topo_B, pctr->cm_edge_map_B, idx_B);
  }
  
  template<typename dtype>
  void Contract<dtype>::prepareC(CTF_int::tensor& C,
                                 const char *     idx_C) {
    IASSERT(pctr != nullptr);
    pctr->prepareT(&C, pctr->C, pctr->cm_topo_C, pctr->cm_edge_map_C, idx_C);
  }
  
  template<typename dtype>
  void Contract<dtype>::execute() {
    IASSERT(pctr != nullptr);
    pctr->execute_persistent();
  }
  
  template<typename dtype>
  void Contract<dtype>::releaseA() {
    IASSERT(pctr != nullptr);
    pctr->releaseT(pctr->A);
  }
  
  template<typename dtype>
  void Contract<dtype>::releaseB() {
    IASSERT(pctr != nullptr);
    IASSERT(pctr->A != pctr->B);
    pctr->releaseT(pctr->B);
  }

  template<typename dtype>
  void Contract<dtype>::releaseC() {
    IASSERT(pctr != nullptr);
    pctr->releaseT(pctr->C);
  }

  template<typename dtype> 
  Contract<dtype>::~Contract() {
    IASSERT(pctr != nullptr);
    pctr->release();
    // TODO: should we cleanup here?
    delete [] pctr->cm_edge_map_A;
    delete [] pctr->cm_edge_map_B;
    delete [] pctr->cm_edge_map_C;
    delete pctr;
  }
}
