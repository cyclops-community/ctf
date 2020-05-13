#include "symmetrization.h"
#include "../tensor/untyped_tensor.h"
#include "../shared/util.h"
#include "../interface/timer.h"
#include "sym_indices.h"
#include "../scaling/scaling.h"

using namespace CTF;

namespace CTF_int {

  void desymmetrize(tensor * sym_tsr,
                    tensor * nonsym_tsr,
                    bool     is_C){
    int i, is, j, sym_dim, scal_diag, num_sy, num_sy_neg;
    int * idx_map_A, * idx_map_B;
    int rev_sign;

    if (sym_tsr == nonsym_tsr) return;

    TAU_FSTART(desymmetrize);

    sym_dim = -1;
    is = -1;
    rev_sign = 1;
    scal_diag = 0;
    num_sy=0;
    num_sy_neg=0;
    for (i=0; i<sym_tsr->order; i++){
      if (sym_tsr->sym[i] != nonsym_tsr->sym[i]){
        is = i;
        if (sym_tsr->sym[i] == AS) rev_sign = -1;
        if (sym_tsr->sym[i] == SY){
          scal_diag = 1;
        }
        if (i>0 && sym_tsr->sym[i-1] != NS) i++;
        sym_dim = i;
        num_sy = 0;
        j=i;
        while (j<sym_tsr->order && sym_tsr->sym[j] != NS){
          num_sy++;
          j++;
        }
        num_sy_neg = 0;
        j=i-1;
        while (j>=0 && sym_tsr->sym[j] != NS){
          num_sy_neg++;
          j--;
        }
        break;
      }
    }
    nonsym_tsr->clear_mapping();
    nonsym_tsr->set_padding();
    copy_mapping(sym_tsr->order, sym_tsr->edge_map, nonsym_tsr->edge_map);
    if (nonsym_tsr->is_sparse){
      nonsym_tsr->nnz_blk = (int64_t*)alloc(sizeof(int64_t)*nonsym_tsr->calc_nvirt());
      std::fill(nonsym_tsr->nnz_blk, nonsym_tsr->nnz_blk+nonsym_tsr->calc_nvirt(), 0);
    }
    nonsym_tsr->is_mapped = 1;
    nonsym_tsr->topo      = sym_tsr->topo;
    nonsym_tsr->set_padding();

    if (sym_dim == -1) {
      nonsym_tsr->size            = sym_tsr->size;
      nonsym_tsr->data            = sym_tsr->data;
      nonsym_tsr->home_buffer     = sym_tsr->home_buffer;
      nonsym_tsr->is_home         = sym_tsr->is_home;
      nonsym_tsr->has_home        = sym_tsr->has_home;
      nonsym_tsr->home_size       = sym_tsr->home_size;
      nonsym_tsr->is_data_aliased = 1;
      nonsym_tsr->set_new_nnz_glb(sym_tsr->nnz_blk);
      TAU_FSTOP(desymmetrize);
      return;
    }
    #ifdef PROF_SYM
    if (sym_tsr->wrld->rank == 0) 
      VPRINTF(2,"Desymmetrizing %s\n", sym_tsr->name);
    if (sym_tsr->profile) {
      char spf[80];
      strcpy(spf,"desymmetrize_");
      strcat(spf,sym_tsr->name);
      CTF::Timer t_pf(spf);
      if (sym_tsr->wrld->rank == 0) 
        VPRINTF(2,"Desymmetrizing %s\n", sym_tsr->name);
      t_pf.start();
    }
    #endif

    if (!nonsym_tsr->is_sparse)
      nonsym_tsr->data = nonsym_tsr->sr->alloc(nonsym_tsr->size);
      //CTF_int::alloc_ptr(nonsym_tsr->size*nonsym_tsr->sr->el_size, (void**)&nonsym_tsr->data);
    nonsym_tsr->set_zero();
    //nonsym_tsr->sr->set(nonsym_tsr->data, nonsym_tsr->sr->addid(), nonsym_tsr->size);

    CTF_int::alloc_ptr(sym_tsr->order*sizeof(int), (void**)&idx_map_A);
    CTF_int::alloc_ptr(sym_tsr->order*sizeof(int), (void**)&idx_map_B);

    for (i=0; i<sym_tsr->order; i++){
      idx_map_A[i] = i;
      idx_map_B[i] = i;
    }

    if (scal_diag){
      if (!is_C){
        tensor * ctsr = sym_tsr;
        if (scal_diag && num_sy+num_sy_neg==1){
          ctsr = new tensor(sym_tsr);
          if (ctsr->is_sparse){
            ctsr->zero_out_sparse_diagonal(is);
          } else {
            ctsr->sym[is] = SH;
            ctsr->zero_out_padding();
            ctsr->sym[is] = SY;
          }
        } 
        for (i=-num_sy_neg-1; i<num_sy; i++){
          if (i==-1) continue;
          idx_map_A[sym_dim] = sym_dim+i+1;
          idx_map_A[sym_dim+i+1] = sym_dim;
          char * mksign = NULL;
          char const * ksign;
          if (rev_sign == -1){
            mksign = (char*)alloc(nonsym_tsr->sr->el_size);
            nonsym_tsr->sr->addinv(nonsym_tsr->sr->mulid(), mksign);
            ksign = mksign;
           } else
            ksign = nonsym_tsr->sr->mulid();
          summation csum = summation(ctsr, idx_map_A, ksign,
                                     nonsym_tsr, idx_map_B, nonsym_tsr->sr->mulid());

          csum.sum_tensors(0);
          if (rev_sign == -1) cdealloc(mksign);
          idx_map_A[sym_dim] = sym_dim;
          idx_map_A[sym_dim+i+1] = sym_dim+i+1;
        }
        if (scal_diag && num_sy+num_sy_neg==1) delete ctsr;
        summation ssum = summation(sym_tsr, idx_map_A, nonsym_tsr->sr->mulid(), nonsym_tsr, idx_map_B, nonsym_tsr->sr->mulid());
        ssum.sum_tensors(0);
      }
      /*printf("DESYMMETRIZED:\n");
      sym_tsr->print();
      printf("TO:\n");
      nonsym_tsr->print();*/

      /*printf("SYM %s\n",sym_tsr->name);
      sym_tsr->print();
      printf("NONSYM %s\n",sym_tsr->name);
      nonsym_tsr->print();*/


      /* Do not diagonal rescaling since sum has beta=0 and overwrites diagonal */
      if (num_sy+num_sy_neg>1){
  //      assert(0); //FIXME: zero_out_padding instead, fractional rescaling factor seems problematic?
    //    print_tsr(stdout, nonsym_tid);
        if (!is_C && scal_diag){
          for (i=-num_sy_neg-1; i<num_sy; i++){
            if (i==-1) continue;
            for (j=0; j<sym_tsr->order; j++){
              if (j==sym_dim+i+1){
                if (j>sym_dim)
                  idx_map_A[j] = sym_dim;
                else
                  idx_map_A[j] = sym_dim-1;
              } else if (j<sym_dim+i+1) {
                idx_map_A[j] = j;
              } else {
                idx_map_A[j] = j-1;
              }
            }
    /*        idx_map_A[sym_dim+i+1] = sym_dim-num_sy_neg;
            idx_map_A[sym_dim] = sym_dim-num_sy_neg;
            for (j=MAX(sym_dim+i+2,sym_dim+1); j<sym_tsr->order; j++){
              idx_map_A[j] = j-1;
            }*/
    /*        printf("tid %d before scale\n", nonsym_tid);
            print_tsr(stdout, nonsym_tid);*/
            char * scalf = (char*)alloc(nonsym_tsr->sr->el_size);
            nonsym_tsr->sr->cast_double(((double)(num_sy+num_sy_neg-1.))/(num_sy+num_sy_neg), scalf);
            scaling sscl(nonsym_tsr, idx_map_A, scalf);
            sscl.execute();
            cdealloc(scalf);
    /*        printf("tid %d after scale\n", nonsym_tid);
            print_tsr(stdout, nonsym_tid);*/
  //          if (ret != CTF_SUCCESS) ABORT;
          }
        } 
    //    print_tsr(stdout, nonsym_tid);
      }
    } else if (!is_C) { 
      summation ssum = summation(sym_tsr, idx_map_A, nonsym_tsr->sr->mulid(), nonsym_tsr, idx_map_B, nonsym_tsr->sr->mulid());
      ssum.execute();
    }
    CTF_int::cdealloc(idx_map_A);
    CTF_int::cdealloc(idx_map_B);  

  /*  switch (sym_tsr->edge_map[sym_dim].type){
      case NOT_MAPPED:
        ASSERT(sym_tsr->edge_map[sym_dim+1].type == NOT_MAPPED);
        rw_smtr<dtype>(sym_tsr->order, sym_tsr->edge_len, 1.0, 0,
           sym_tsr->sym, nonsym_tsr->sym,
           sym_tsr->data, nonsym_tsr->data);
        rw_smtr<dtype>(sym_tsr->order, sym_tsr->edge_len, rev_sign, 1,
           sym_tsr->sym, nonsym_tsr->sym,
           sym_tsr->data, nonsym_tsr->data);
        break;

      case VIRTUAL_MAP:
        if (sym_tsr->edge_map[sym_dim+1].type == VIRTUAL_MAP){
    nvirt = 

        } else {
    ASSERT(sym_tsr->edge_map[sym_dim+1].type == PHYSICAL_MAP);

        }
        break;
      
      case PHYSICAL_MAP:
        if (sym_tsr->edge_map[sym_dim+1].type == VIRTUAL_MAP){

        } else {
    ASSERT(sym_tsr->edge_map[sym_dim+1].type == PHYSICAL_MAP);

        }
        break;
    }*/
    #ifdef PROF_SYM
    if (sym_tsr->profile) {
      char spf[80];
      strcpy(spf,"desymmetrize_");
      strcat(spf,sym_tsr->name);
      CTF::Timer t_pf(spf);
      t_pf.stop();
    }
    #endif

    TAU_FSTOP(desymmetrize);

  }

  void symmetrize(tensor * sym_tsr,
                  tensor * nonsym_tsr){
    int i, j, is, sym_dim, scal_diag, num_sy, num_sy_neg;
    int * idx_map_A, * idx_map_B;
    int rev_sign;

    TAU_FSTART(symmetrize);
    
    #ifdef PROF_SYM
    if (sym_tsr->profile) {
      char spf[80];
      strcpy(spf,"symmetrize_");
      strcat(spf,sym_tsr->name);
      CTF::Timer t_pf(spf);
      if (sym_tsr->wrld->rank == 0) 
        VPRINTF(2,"Symmetrizing %s\n", sym_tsr->name);
      t_pf.start();
    }
    #endif

    sym_dim = -1;
    is = -1;
    rev_sign = 1;
    scal_diag = 0;
    num_sy=0;
    num_sy_neg=0;
    for (i=0; i<sym_tsr->order; i++){
      if (sym_tsr->sym[i] != nonsym_tsr->sym[i]){
        is = i;
        if (sym_tsr->sym[i] == AS) rev_sign = -1;
        if (sym_tsr->sym[i] == SY){
          scal_diag = 1;
        }
        if (i>0 && sym_tsr->sym[i-1] != NS) i++;
        sym_dim = i;
        num_sy = 0;
        j=i;
        while (j<sym_tsr->order && sym_tsr->sym[j] != NS){
          num_sy++;
          j++;
        }
        num_sy_neg = 0;
        j=i-1;
        while (j>=0 && sym_tsr->sym[j] != NS){
          num_sy_neg++;
          j--;
        }
        break;
      }
    }
    if (sym_dim == -1) {
      sym_tsr->topo    = nonsym_tsr->topo;
      copy_mapping(nonsym_tsr->order, nonsym_tsr->edge_map, sym_tsr->edge_map);
      if (sym_tsr->is_sparse){
        sym_tsr->nnz_blk = (int64_t*)alloc(sizeof(int64_t)*sym_tsr->calc_nvirt());
        std::fill(sym_tsr->nnz_blk, sym_tsr->nnz_blk+sym_tsr->calc_nvirt(), 0);
      }
      sym_tsr->is_mapped    = 1;
      sym_tsr->set_padding();
      sym_tsr->size     = nonsym_tsr->size;
      sym_tsr->data     = nonsym_tsr->data;
      sym_tsr->is_home  = nonsym_tsr->is_home;
      sym_tsr->set_new_nnz_glb(nonsym_tsr->nnz_blk);
    } else {
  
      //sym_tsr->sr->set(sym_tsr->data, sym_tsr->sr->addid(), sym_tsr->size);
      CTF_int::alloc_ptr(sym_tsr->order*sizeof(int), (void**)&idx_map_A);
      CTF_int::alloc_ptr(sym_tsr->order*sizeof(int), (void**)&idx_map_B);
  
      for (i=0; i<sym_tsr->order; i++){
        idx_map_A[i] = i;
        idx_map_B[i] = i;
      }
      
      if (0){
        //FIXME: this is not robust when doing e.g. {SY, SY, SY, NS} -> {SY, NS, SY, NS}
        for (i=-num_sy_neg-1; i<num_sy; i++){
          if (i==-1) continue;
          idx_map_A[sym_dim] = sym_dim+i+1;
          idx_map_A[sym_dim+i+1] = sym_dim;
      //    printf("symmetrizing\n");
    /*      summation csum = summation(nonsym_tsr, idx_map_A, rev_sign,
                                        sym_tsr, idx_map_B, 1.0);*/
            char * ksign = (char*)alloc(nonsym_tsr->sr->el_size);
            if (rev_sign == -1)
              nonsym_tsr->sr->addinv(nonsym_tsr->sr->mulid(), ksign);
            else
              nonsym_tsr->sr->copy(ksign, nonsym_tsr->sr->mulid());
            summation csum(nonsym_tsr, idx_map_A, ksign,
                              sym_tsr, idx_map_B, nonsym_tsr->sr->mulid());
            csum.sum_tensors(0);
            cdealloc(ksign);
  
      //    print_tsr(stdout, sym_tid);
          idx_map_A[sym_dim] = sym_dim;
          idx_map_A[sym_dim+i+1] = sym_dim+i+1;
        }
        if (scal_diag && num_sy+num_sy_neg == 1) {
      /*    for (i=-num_sy_neg-1; i<num_sy-1; i++){
            tensors[sym_tid]->sym[sym_dim+i+1] = SH;
            zero_out_padding(sym_tid);
            tensors[sym_tid]->sym[sym_dim+i+1] = SY;
          }*/
          sym_tsr->sym[is] = SH;
          sym_tsr->zero_out_padding();
          sym_tsr->sym[is] = SY;
      /*    for (i=-num_sy_neg-1; i<num_sy-1; i++){
            tensors[sym_tid]->sym[sym_dim+i+1] = SY;
          }*/
        }
        
        summation ssum = summation(nonsym_tsr, idx_map_A, nonsym_tsr->sr->mulid(), sym_tsr, idx_map_B, nonsym_tsr->sr->mulid());
        ssum.sum_tensors(0);
          
  
        if (num_sy+num_sy_neg > 1){
          //assert(0); //FIXME: use zero_out_padding?
          if (scal_diag){
         //   printf("symmetrizing diagonal=%d\n",num_sy);
            for (i=-num_sy_neg-1; i<num_sy; i++){
              if (i==-1) continue;
              idx_map_B[sym_dim+i+1] = sym_dim-num_sy_neg;
              idx_map_B[sym_dim] = sym_dim-num_sy_neg;
              for (j=MAX(sym_dim+i+2,sym_dim+1); j<sym_tsr->order; j++){
                idx_map_B[j] = j-i-num_sy_neg-1;
              }
              /*printf("tid %d before scale\n", nonsym_tid);
              print_tsr(stdout, sym_tid);*/
              char * scalf = (char*)alloc(nonsym_tsr->sr->el_size);
              nonsym_tsr->sr->cast_double(((double)(num_sy-i-1.))/(num_sy-i), scalf);
              scaling sscl = scaling(sym_tsr, idx_map_B, scalf);
              sscl.execute();
      /*        printf("tid %d after scale\n", sym_tid);
              print_tsr(stdout, sym_tid);*/
              //if (ret != CTF_SUCCESS) ABORT;
            }
          }
        }
      } else {
        tensor sym_tsr2(sym_tsr,1,0);
        summation ssum = summation(nonsym_tsr, idx_map_A, nonsym_tsr->sr->mulid(), &sym_tsr2, idx_map_B, nonsym_tsr->sr->addid());
        ssum.execute();
        summation ssum2 = summation(&sym_tsr2, idx_map_A, nonsym_tsr->sr->mulid(), sym_tsr, idx_map_B, nonsym_tsr->sr->mulid());
        ssum2.execute();
      }
      CTF_int::cdealloc(idx_map_A);
      CTF_int::cdealloc(idx_map_B);
    }
    #ifdef PROF_SYM
    if (sym_tsr->profile) {
      char spf[80];
      strcpy(spf,"symmetrize_");
      strcat(spf,sym_tsr->name);
      CTF::Timer t_pf(spf);
      t_pf.stop();
    }
    #endif


    TAU_FSTOP(symmetrize);
  }


  void cmp_sym_perms(int         ndim,
                     int const * sym,
                     int *       nperm,
                     int **      perm,
                     double *    sign){
    int i, np;
    int * pm;
    double sgn;

    ASSERT(sym[0] != NS);
    CTF_int::alloc_ptr(sizeof(int)*ndim, (void**)&pm);

    np=0;
    sgn=1.0;
    for (i=0; i<ndim; i++){
      if (sym[i]==AS){
        sgn=-1.0;
      }
      if (sym[i]!=NS){
        np++;
      } else {
        np++;
        break;
      }    
    }
    /* a circular shift of n indices requires n-1 swaps */
    if (np % 2 == 1) sgn = 1.0;

    for (i=0; i<np; i++){
      pm[i] = (i+1)%np;
    }
    for (i=np; i<ndim; i++){
      pm[i] = i;
    }

    *nperm = np;
    *perm = pm;
    *sign = sgn;
  }

  void order_perm(tensor const * A,
                  tensor const * B,
                  int *          idx_arr,
                  int            off_A,
                  int            off_B,
                  int *          idx_A,
                  int *          idx_B,
                  int &          add_sign,
                  int &          mod){
    int  iA, jA, iB, jB, iiB, broken, tmp;

    //find all symmetries in A
    for (iA=0; iA<A->order; iA++){
      if (A->sym[iA] != NS){
        jA=iA;
        iB = idx_arr[2*idx_A[iA]+off_B];
        while (A->sym[jA] != NS){
          broken = 0;
          jA++;
          jB = idx_arr[2*idx_A[jA]+off_B];
          if ((iB == -1) ^ (jB == -1)) broken = 1;
         /* if (iB != -1 && jB != -1) {
            if (B->sym[iB] != A->sym[iA]) broken = 1;
          }*/
          if (iB != -1 && jB != -1) {
            /* Do this because iB,jB can be in reversed order */
            for (iiB=MIN(iB,jB); iiB<MAX(iB,jB); iiB++){
              ASSERT(iiB >= 0 && iiB <= B->order);
              if (B->sym[iiB] == NS) broken = 1;
            }
          }
          

          /*//if (iB == jB) broken = 1;
          } */
          //if the symmetry is preserved, make sure index map is ordered
          if (!broken){
            if (idx_A[iA] > idx_A[jA]){
              idx_arr[2*idx_A[iA]+off_A] = jA;
              idx_arr[2*idx_A[jA]+off_A] = iA;
              tmp                          = idx_A[iA];
              idx_A[iA] = idx_A[jA];
              idx_A[jA] = tmp;
              if (A->sym[iA] == AS) add_sign *= -1.0;
              mod = 1;
            } 
          }
        }
      }
    }
  }

  void order_perm(tensor const * A,
                  tensor const * B,
                  tensor const * C,
                  int *          idx_arr,
                  int            off_A,
                  int            off_B,
                  int            off_C,
                  int *          idx_A,
                  int *          idx_B,
                  int *          idx_C,
                  int &          add_sign,
                  int &          mod){

    int  iA, jA, iB, iC, jB, jC, iiB, iiC, broken, tmp;

    //find all symmetries in A
    for (iA=0; iA<A->order; iA++){
      if (A->sym[iA] != NS){
        jA=iA;
        iB = idx_arr[3*idx_A[iA]+off_B];
        iC = idx_arr[3*idx_A[iA]+off_C];
        while (A->sym[jA] != NS){
          broken = 0;
          jA++;
          jB = idx_arr[3*idx_A[jA]+off_B];
          //if (iB == jB) broken = 1;
          if (iB != -1 && jB != -1){
            for (iiB=MIN(iB,jB); iiB<MAX(iB,jB); iiB++){
              if (B->sym[iiB] ==  NS) broken = 1;
            }
          } 
          if ((iB == -1) ^ (jB == -1)) broken = 1;
          jC = idx_arr[3*idx_A[jA]+off_C];
          //if (iC == jC) broken = 1;
          if (iC != -1 && jC != -1){
            for (iiC=MIN(iC,jC); iiC<MAX(iC,jC); iiC++){
              if (C->sym[iiC] == NS) broken = 1;
            }
          } 
          if ((iC == -1) ^ (jC == -1)) broken = 1;
          //if the symmetry is preserved, make sure index map is ordered
          if (!broken){
            if (idx_A[iA] > idx_A[jA]){
              idx_arr[3*idx_A[iA]+off_A] = jA;
              idx_arr[3*idx_A[jA]+off_A] = iA;
              tmp                          = idx_A[iA];
              idx_A[iA] = idx_A[jA];
              idx_A[jA] = tmp;
              if (A->sym[iA] == AS) add_sign *= -1.0;
              mod = 1;
            } 
          }
        }
      }
    }
  }

  void add_sym_perm(std::vector<summation>& perms,
                    std::vector<int>&       signs,
                    summation const &       new_perm,
                    int                     new_sign){

    int mod, num_tot, i;
    int * idx_arr;
    int add_sign;
    tensor * tsr_A, * tsr_B;

    add_sign = new_sign;
    summation norm_ord_perm(new_perm);
    
    tsr_A = new_perm.A;
    tsr_B = new_perm.B;
    
    inv_idx(tsr_A->order, norm_ord_perm.idx_A,
            tsr_B->order, norm_ord_perm.idx_B,
            &num_tot, &idx_arr);
    //keep permuting until we get to normal order (no permutations left)
    do {
      mod = 0;
      order_perm(tsr_A, tsr_B, idx_arr, 0, 1,
                 norm_ord_perm.idx_A, norm_ord_perm.idx_B,
                 add_sign, mod);
      order_perm(tsr_B, tsr_A, idx_arr, 1, 0,
                 norm_ord_perm.idx_B, norm_ord_perm.idx_A,
                 add_sign, mod);
    } while (mod);
    add_sign = add_sign*align_symmetric_indices(tsr_A->order, norm_ord_perm.idx_A, tsr_A->sym,
                                                tsr_B->order, norm_ord_perm.idx_B, tsr_B->sym);

    // check if this summation is equivalent to one of the other permutations
    for (i=0; i<(int)perms.size(); i++){
      if (perms[i].is_equal(norm_ord_perm)){
        CTF_int::cdealloc(idx_arr);
        return;
      }
    }
    perms.push_back(norm_ord_perm);
    signs.push_back(add_sign);
    CTF_int::cdealloc(idx_arr);
  }

  void add_sym_perm(std::vector<contraction>& perms,
                    std::vector<int>&         signs,
                    contraction const &       new_perm,
                    int                       new_sign){
    int mod, num_tot, i;
    int * idx_arr;
    int add_sign;
    tensor * tsr_A, * tsr_B, * tsr_C;

    tsr_A = new_perm.A;
    tsr_B = new_perm.B;
    tsr_C = new_perm.C;

    add_sign = new_sign;
    contraction norm_ord_perm(new_perm);
    
    inv_idx(tsr_A->order, norm_ord_perm.idx_A,
            tsr_B->order, norm_ord_perm.idx_B,
            tsr_C->order, norm_ord_perm.idx_C,
            &num_tot, &idx_arr);
    //keep permuting until we get to normal order (no permutations left)
    do {
      mod = 0;
      order_perm(tsr_A, tsr_B, tsr_C, idx_arr, 0, 1, 2,
                 norm_ord_perm.idx_A, norm_ord_perm.idx_B, norm_ord_perm.idx_C,
                 add_sign, mod);
      order_perm(tsr_B, tsr_A, tsr_C, idx_arr, 1, 0, 2,
                 norm_ord_perm.idx_B, norm_ord_perm.idx_A, norm_ord_perm.idx_C,
                 add_sign, mod);
      order_perm(tsr_C, tsr_B, tsr_A, idx_arr, 2, 1, 0,
                 norm_ord_perm.idx_C, norm_ord_perm.idx_B, norm_ord_perm.idx_A,
                 add_sign, mod);
    } while (mod);
    add_sign *= align_symmetric_indices(tsr_A->order,
                                        norm_ord_perm.idx_A,
                                        tsr_A->sym,
                                        tsr_B->order,
                                        norm_ord_perm.idx_B,
                                        tsr_B->sym,
                                        tsr_C->order,
                                        norm_ord_perm.idx_C,
                                        tsr_C->sym);

    for (i=0; i<(int)perms.size(); i++){
      if (perms[i].is_equal(norm_ord_perm)){
        CTF_int::cdealloc(idx_arr);
        return;
      }
    }
    perms.push_back(norm_ord_perm);
    signs.push_back(add_sign);
    CTF_int::cdealloc(idx_arr);
  }

  void get_sym_perms(summation const &       sum,
                     std::vector<summation>& perms,
                     std::vector<int>&       signs){
    int i, j, k, tmp;
    int sign;
    tensor * tsr_A, * tsr_B;
    tsr_A = sum.A;
    tsr_B = sum.B;

    summation new_type(sum);
    add_sym_perm(perms, signs, new_type, 1);

    for (i=0; i<tsr_A->order; i++){
      j=i;
      while (tsr_A->sym[j] != NS){
        j++;
        for (k=0; k<(int)perms.size(); k++){
          summation new_type1(perms[k]);
          sign = signs[k];
          if (tsr_A->sym[j-1] == AS) sign *= -1;
          tmp                 = new_type1.idx_A[i];
          new_type1.idx_A[i]  = new_type1.idx_A[j];
          new_type1.idx_A[j]  = tmp;
          add_sym_perm(perms, signs, new_type1, sign);
        }
      }
    }
    for (i=0; i<tsr_B->order; i++){
      j=i;
      while (tsr_B->sym[j] != NS){
        j++;
        for (k=0; k<(int)perms.size(); k++){
          summation new_type2(perms[k]);
          sign = signs[k];
          if (tsr_B->sym[j-1] == AS) sign *= -1;
          tmp                = new_type2.idx_B[i];
          new_type2.idx_B[i] = new_type2.idx_B[j];
          new_type2.idx_B[j] = tmp;
          add_sym_perm(perms, signs, new_type2, sign);
        }
      }
    }
  }

  void get_sym_perms(contraction const &       ctr,
                     std::vector<contraction>& perms,
                     std::vector<int>&         signs){
  //  dtype * scl_alpha_C;
  //  int ** scl_idx_maps_C;
  //  nscl_C = 0;
  //  CTF_int::alloc_ptr(sizeof(dtype)*order_C, (void**)&scl_alpha_C);
  //  CTF_int::alloc_ptr(sizeof(int*)*order_C, (void**)&scl_idx_maps_C);

    int i, j, k, tmp;
    int sign;
    tensor * tsr_A, * tsr_B, * tsr_C;
    tsr_A = ctr.A;
    tsr_B = ctr.B;
    tsr_C = ctr.C;

    contraction new_type = contraction(ctr);
     
    add_sym_perm(perms, signs, new_type, 1);

    for (i=0; i<tsr_A->order; i++){
      j=i;
      while (tsr_A->sym[j] != NS){
        j++;
        for (k=0; k<(int)perms.size(); k++){
          contraction ntype(perms[k]);
          sign = signs[k];
          if (tsr_A->sym[j-1] == AS) sign *= -1;
          tmp             = ntype.idx_A[i];
          ntype.idx_A[i]  = ntype.idx_A[j];
          ntype.idx_A[j]  = tmp;
          add_sym_perm(perms, signs, ntype, sign);
        }
      }
    }
    for (i=0; i<tsr_B->order; i++){
      j=i;
      while (tsr_B->sym[j] != NS){
        j++;
        for (k=0; k<(int)perms.size(); k++){
          contraction ntype(perms[k]);
          sign = signs[k];
          if (tsr_B->sym[j-1] == AS) sign *= -1;
          tmp             = ntype.idx_B[i];
          ntype.idx_B[i]  = ntype.idx_B[j];
          ntype.idx_B[j]  = tmp;
          add_sym_perm(perms, signs, ntype, sign);
        }
      }
    }
    
    for (i=0; i<tsr_C->order; i++){
      j=i;
      while (tsr_C->sym[j] != NS){
        j++;
        for (k=0; k<(int)perms.size(); k++){
          contraction ntype(perms[k]);
          sign = signs[k];
          if (tsr_C->sym[j-1] == AS) sign *= -1;
          tmp             = ntype.idx_C[i];
          ntype.idx_C[i]  = ntype.idx_C[j];
          ntype.idx_C[j]  = tmp;
          add_sym_perm(perms, signs, ntype, sign);
        }
      }
    }
  }
}
