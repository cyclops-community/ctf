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
    pctr = new CTF_int::contraction(&A, idx_A, &B, idx_B, (char const *)&alpha, &C, idx_C, (char const *)&beta, nullptr, true);
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
    pctr = new CTF_int::contraction(&A, idx_A, &B, idx_B, (char const *)&alpha, &C, idx_C, (char const *)&beta, &func, true);
    pctr->prepare();
  }

  // TODO: can do away with this function
  template<typename dtype>
  void Contract<dtype>::prepare() {
    IASSERT(pctr != nullptr);
    pctr->prepare();
  }
 
  template<typename dtype>
  void Contract<dtype>::prepareA(CTF_int::tensor& A,
                                 const char *     idx_A) {
    IASSERT(pctr != nullptr);
    pctr->prepareA(&A, idx_A);
  }

  template<typename dtype>
  void Contract<dtype>::prepareB(CTF_int::tensor& B,
                                 const char *     idx_B) {
    IASSERT(pctr != nullptr);
    pctr->prepareB(&B, idx_B);
  }
  
  template<typename dtype>
  void Contract<dtype>::prepareC(CTF_int::tensor& C,
                                 const char *     idx_C) {
    IASSERT(pctr != nullptr);
    pctr->prepareC(&C, idx_C);
  }
  
  template<typename dtype>
  void Contract<dtype>::execute() {
    IASSERT(pctr != nullptr);
    pctr->execute_persistent();
  }
  
  // TODO: can do away with this function
  template<typename dtype>
  void Contract<dtype>::release() {
    IASSERT(pctr != nullptr);
    pctr->release();
  }
  
  template<typename dtype>
  void Contract<dtype>::releaseA() {
    IASSERT(pctr != nullptr);
    pctr->releaseA();
  }
  
  template<typename dtype>
  void Contract<dtype>::releaseB() {
    IASSERT(pctr != nullptr);
    pctr->releaseB();
  }

  template<typename dtype>
  void Contract<dtype>::releaseC() {
    IASSERT(pctr != nullptr);
    pctr->releaseC();
  }

  template<typename dtype> 
  Contract<dtype>::~Contract() {
    IASSERT(pctr != nullptr);
    pctr->release();
    delete pctr;
  }
}
