/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __INT_TENSOR_H__
#define __INT_TENSOR_H__

#include "../mapping/int_mapping.h"

namespace CTF_int {

class pair {
  public: 
    int64_t k;

    virtual char * v() { assert(0); };

    pair() {}
/*    pair(key k_, char const * d_, int len) {
      k = k_;
      memcpy(d, d_, len); 
    }*/ 

    bool operator< (const pair& other) const{
      return k < other.k;
    }
/*    bool operator==(const pair& other) const{
      return (k == other.k && d == other.d);
    }
    bool operator!=(const pair& other) const{
      return !(*this == other);
    }*/
};


class tensor {
  private:
  int init(semiring sr,
           int order,
           int const * edge_len,
           int const * sym,
           bool alloc_data,
           char const * name,
           bool profile);
  public:
  semiring sr;
  int order;
  //padded tensor edge lengths
  int * edge_len;
  //padding along each edge length
  int * padding;
  int is_scp_padded;
  int * scp_padding; /* to be used by scalapack wrapper */
  int * sym;
  int * sym_table; /* can be compressed into bitmap */
  bool is_mapped;
  bool is_alloced;
  int itopo;
  mapping * edge_map;
  int64_t size;
  bool is_folded;
  int * inner_ordering;
  int rec_tid;
  bool is_cyclic;
  bool is_matrix;
  bool is_data_aliased;
  bool slay;
  bool has_zero_edge_len;
  union {
    char * data;
    pair * pairs;
  };
  char * home_buffer;
  int64_t home_size;
  bool is_home;
  bool has_home;
  char const * name;
  bool profile;

  //FIXME: unfolds other
  //tensor(tensor const & other);
  tensor(tensor * other);

  tensor(semiring sr,
         int order,
         int const * edge_len,
         int const * sym,
         bool alloc_data = false,
         char const * name = NULL,
         bool profile = 1);

  int set_padding();

  void set_zero();

  void print_map(FILE * stream) const;


  int save_mapping(int **     old_phase,
                   int **     old_rank,
                   int **     old_virt_dim,
                   int **     old_pe_lda,
                   int64_t *   old_size,
                   int *      was_cyclic,
                   int **     old_padding,
                   int **     old_edge_len,
                   topology const * topo);


};

}

#endif// __INT_TENSOR_H__

