namespace CTF_int {
  /**
   * \brief return format string for templated type
   */
  template <typename dtype>
  const char * get_fmt(){
    printf("CTF ERROR: Format of tensor unsupported for sparse I/O\n");
    IASSERT(0);
    return NULL;
  }

  template <>
  inline const char * get_fmt<float>(){
    return " %f";
  }

  template <>
  inline const char * get_fmt<double>(){
    return " %lf";
  }

  template <>
  inline const char * get_fmt<int>(){
    return " %d";
  }

  template <>
  inline const char * get_fmt<int64_t>(){
    return " %ld";
  }

  /**
   * \brief parse string containing sparse tensor into data
   * \param[in] lvals array of string, one per line/entry,
   *   formatted as i1, ..., i_order v  or
   *                i1, ..., i_order    if with_vals=false
   * \param[in] order num modes in tensor
   * \param[in] pmulid pointer to multiplicative identity, used only if with_vals=false
   * \param[in] lens dimensions of tensor
   * \param[in] nvals number of entries in lvals
   * \param[in] pairs array of tensor index/value pairs to  fill
   * \param[in] with_vals whether values are included in file
   * \param[in] rev_order whether index order should be reversed
   */
  template <typename dtype>
  void parse_sparse_tensor_data(char **lvals, int order, dtype const * pmulid, int64_t const * lens, int64_t nvals, CTF::Pair<dtype> * pairs, bool with_vals, bool rev_order){
    int64_t i;
    dtype mulid;
    if (!with_vals) mulid = *pmulid;

    int64_t * ind = (int64_t *)malloc(order*sizeof(int64_t));
    for (i=0; i<nvals; i++) {
      double v;
      int ptr = 0;
      int aptr;
      sscanf(lvals[i]+ptr, "%ld%n", ind+0, &aptr);
      ptr += aptr;
      for (int j=1; j<order; j++){
        sscanf(lvals[i]+ptr, " %ld%n", ind+j, &aptr);
        ptr += aptr;
      }
      if (with_vals)
        sscanf(lvals[i]+ptr, get_fmt<dtype>(), &v);
      int64_t lda = 1;
      pairs[i].k = 0;
      if (rev_order){
        for (int j=0; j<order; j++){
          pairs[i].k += ind[order-j-1]*lda;
          lda *= lens[j];
        }
      } else {
        for (int j=0; j<order; j++){
          pairs[i].k += ind[j]*lda;
          lda *= lens[j];
        }
      }
      if (with_vals)
        pairs[i].d = v;
      else
        pairs[i].d = mulid;
    }
    free(ind);
  }

  /**
   * \brief serialize sparse tensor data to create string
   * \return lvals array of string, one per line/entry,
   *   formatted as i1, ..., i_order v  or
   *                i1, ..., i_order    if with_vals=false or
   *                i_order, ..., i1 v  if rev_order=true
   * \param[in] order num modes in tensor
   * \param[in] pmulid pointer to multiplicative identity, used only if with_vals=false
   * \param[in] lens dimensions of tensor
   * \param[in] nvals number of entries in lvals
   * \param[in] pairs array of tensor index/value pairs to  fill
   * \param[in] with_vals whether values are included in file
   * \param[in] rev_order whether index order should be reversed
   * \param[out] length of string output
   */
  template <typename dtype>
  char * serialize_sparse_tensor_data(int order, int64_t const * lens, int64_t nvals, CTF::Pair<dtype> * pairs, bool with_vals, bool rev_order, int64_t & str_len){
    int64_t i;

    int64_t * ind = (int64_t *)malloc(order*sizeof(int64_t));
    str_len = 0;
    for (i=0; i<nvals; i++){
      int64_t key = pairs[i].k;
      for (int j=0; j<order; j++){
        ind[j] = key % lens[j];
        key = key / lens[j];
      }

      int astr_len = 0;
      astr_len += snprintf(NULL, 0, "%ld", ind[0]);
      for (int j=1; j<order; j++){
        astr_len += snprintf(NULL, 0, " %ld", ind[j]);
      }
      if (with_vals)
        astr_len += snprintf(NULL, 0, get_fmt<dtype>(), pairs[i].d);
      astr_len += snprintf(NULL, 0, "\n");

      str_len += astr_len;
    }
    char * datastr = (char*)CTF_int::alloc(sizeof(char)*(str_len+1));
    int64_t str_ptr = 0;
    for (i=0; i<nvals; i++){
      int64_t key = pairs[i].k;
      if (rev_order){
        for (int j=0; j<order; j++){
          ind[order-j-1] = key % lens[j];
          key = key / lens[j];
        }
      } else {
        for (int j=0; j<order; j++){
          ind[j] = key % lens[j];
          key = key / lens[j];
        }
      }

      str_ptr += sprintf(datastr+str_ptr, "%ld", ind[0]);
      for (int j=1; j<order; j++){
        str_ptr += sprintf(datastr+str_ptr, " %ld", ind[j]);
      }
      if (with_vals)
        str_ptr += sprintf(datastr+str_ptr, get_fmt<dtype>(), pairs[i].d);
      str_ptr += sprintf(datastr+str_ptr, "\n");
    }
    free(ind);
    return datastr;
  }

  /**
   * \brief read sparse tensor data from file using MPI-I/O, creating string with one entry per line (different entries on each process)
   * \param[in] dw MPI world/comm
   * \param[in] fpath file name
   * \param[in] datastr array of strings to create and read from file
   */
  template <typename dtype>
  int64_t read_data_mpiio(CTF::World const * dw, char const *fpath, char ***datastr){
    MPI_File fh;
    MPI_Offset filesize;
    MPI_Offset localsize;
    MPI_Offset start,end;
    MPI_Status status;
    char *chunk = NULL;
    int overlap = 300; // define
    int64_t ned = 0;
    int64_t i = 0;

    MPI_File_open(MPI_COMM_WORLD,fpath, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

    /* Get the size of file */
    MPI_File_get_size(fh, &filesize); //return in bytes

    localsize = filesize/dw->np;
    start = dw->rank * localsize;
    end = start + localsize;
    end += overlap;

    if (dw->rank  == dw->np-1) end = filesize;
    localsize = end - start; //OK

    chunk = (char*)malloc( (localsize + 1)*sizeof(char));
    MPI_File_read_at_all(fh, start, chunk, localsize, MPI_CHAR, &status);
    chunk[localsize] = '\0';

    int64_t locstart=0, locend=localsize;
    if (dw->rank != 0) {
      while(chunk[locstart] != '\n') locstart++;
      locstart++;
    }
    if (dw->rank != dw->np-1) {
      locend-=overlap;
      while(chunk[locend] != '\n') locend++;
      locend++;
    }
    localsize = locend-locstart; //OK

    char *data = (char *)CTF_int::alloc((localsize+1)*sizeof(char));
    memcpy(data, &(chunk[locstart]), localsize);
    data[localsize] = '\0';
    free(chunk);

    //printf("[%d] local chunk = [%ld,%ld) / %ld\n", myid, start+locstart, start+locstart+localsize, filesize);
    for ( i=0; i<localsize; i++){
      if (data[i] == '\n') ned++;
    }
    //printf("[%d] ned= %ld\n",myid, ned);

    (*datastr) = (char **)CTF_int::alloc(std::max(ned,(int64_t)1)*sizeof(char *));
    (*datastr)[0] = strtok(data,"\n");

    for ( i=1; i < ned; i++)
      (*datastr)[i] = strtok(NULL, "\n");
    if ((*datastr)[0] == NULL)
      CTF_int::cdealloc(data); 
    MPI_File_close(&fh);

    return ned;
  }


  /**
   * \brief write sparse tensor data to file using MPI-I/O, from string with one entry per line (different entries on each process)
   * \param[in] dw world (comm)
   * \param[in] fpath file name
   * \param[in] datastr array of strings to write to file
   * \param[in] str_len num chars in string
   */
  template <typename dtype>
  void write_data_mpiio(CTF::World const * dw, char const *fpath, char * datastr, int64_t str_len){
    MPI_File fh;
    MPI_Offset offset;
    MPI_Status status;

    // clear contents of file if exists, then reopen
    MPI_File_open(MPI_COMM_WORLD, fpath, MPI_MODE_WRONLY | MPI_MODE_CREATE | MPI_MODE_DELETE_ON_CLOSE, MPI_INFO_NULL, &fh);
    MPI_File_close(&fh);
    MPI_File_open(MPI_COMM_WORLD, fpath, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fh);

    int64_t ioffset;
    MPI_Scan(&str_len, &ioffset, 1, MPI_INT64_T, MPI_SUM, dw->comm);

    offset = ioffset - str_len;

    MPI_File_write_at_all(fh, offset, datastr, str_len, MPI_CHAR, &status);
    MPI_File_close(&fh);
  }

}

