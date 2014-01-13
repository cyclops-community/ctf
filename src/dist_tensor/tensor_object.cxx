


template<typename dtype>
void tensor<dtype>::print_map(FILE * stream) const{

    printf("CTF: sym  len  tphs  pphs  vphs\n");
    for (int dim=0; dim<ndim; dim++){
      int tp = calc_phase(edge_map+dim);
      int pp = calc_phys_phase(edge_map+dim);
      int vp = tp/pp;
      printf("CTF: %2s %5d %5d %5d %5d\n", SY_strings[sym[dim]], edge_len[dim], tp, pp, vp);
    }
}
