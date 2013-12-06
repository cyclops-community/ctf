


template<typename dtype>
void tensor<dtype>::print_map(FILE * stream) const{

    printf("CTF: sym  len  tphs  pphs  vphs\n");
    for (int dim=0; dim<ndim; dim++){
      int tp = calc_phase(edge_map+dim);
      int pp = calc_phys_phase(edge_map+dim);
      int vp = tp/pp;
      printf("CTF: %2s %5d %5d %5d %5d\n", SY_strings[sym[dim]], edge_len[dim], tp, pp, vp);
    }
/*    printf("Tensor %d of dimension %d is mapped to a ", tid, tsr->ndim);
    if (is_inner){
      for (i=0; i<inner_topovec[tsr->itopo].ndim-1; i++){
              printf("%d-by-", inner_topovec[tsr->itopo].dim_comm[i]->np);
      }
      if (inner_topovec[tsr->itopo].ndim > 0)
              printf("%d inner topology.\n", inner_topovec[tsr->itopo].dim_comm[i]->np);
    } else {
      for (i=0; i<topovec[tsr->itopo].ndim-1; i++){
              printf("%d-by-", topovec[tsr->itopo].dim_comm[i]->np);
      }
      if (topovec[tsr->itopo].ndim > 0)
              printf("%d topology.\n", topovec[tsr->itopo].dim_comm[i]->np);
    }
    for (i=0; i<tsr->ndim; i++){
      switch (tsr->edge_map[i].type){
        case NOT_MAPPED:
          printf("Dimension %d of length %d and symmetry %d is not mapped\n",i,tsr->edge_len[i],tsr->sym[i]);
          break;

        case PHYSICAL_MAP:
          printf("Dimension %d of length %d and symmetry %d is mapped to physical dimension %d with phase %d\n",
            i,tsr->edge_len[i],tsr->sym[i],tsr->edge_map[i].cdt,tsr->edge_map[i].np);
          map = &tsr->edge_map[i];
          while (map->has_child){
            map = map->child;
            if (map->type == VIRTUAL_MAP)
              printf("\tDimension %d also has a virtualized child of phase %d\n", i, map->np);
            else
              printf("\tDimension %d also has a physical child mapped to physical dimension %d with phase %d\n",
                      i, map->cdt, map->np);
          }
          break;

        case VIRTUAL_MAP:
          printf("Dimension %d of length %d and symmetry %d is mapped virtually with phase %d\n",
            i,tsr->edge_len[i],tsr->sym[i],tsr->edge_map[i].np);
          break;
      }
    }*/
}
