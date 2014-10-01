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

    bool operator< (const pair& other) const{
      return k < other.k;
    }
/*  bool operator==(const pair& other) const{
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
    world wrld;
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
    tensor * rec_tsr;
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

    /**
     * \brief creates tensor copy
     * \param[in] other tensor to copy
     */
    //FIXME: unfolds other
    //tensor(tensor const & other);
    tensor(tensor * other);

    /**
     * \brief defines a tensor object with some mapping (if alloc_data)
     * \param[in] sr semiring 
     */
    tensor(semiring sr,
           int order,
           int const * edge_len,
           int const * sym,
           bool alloc_data = false,
           char const * name = NULL,
           bool profile = 1);

    /**
     * \brief sets padding and local size of a tensor
     */
    int set_padding();

    void set_zero();

    void print_map(FILE * stream) const;

    /**
     * \brief compute the cyclic phase of each tensor dimension
     * \return int * of cyclic phases
     */
    int * calc_phase();

    /**
     * \brief calculate the total number of blocks of the tensor
     * \return int total phase factor
     */
    int calc_tot_phase();

    /**
     * \brief calculate virtualization factor of tensor
     * return virtualization factor
     */
    int64_t calc_nvirt();

    int save_mapping(int **     old_phase,
                     int **     old_rank,
                     int **     old_virt_dim,
                     int **     old_pe_lda,
                     int64_t *   old_size,
                     int *      was_cyclic,
                     int **     old_padding,
                     int **     old_edge_len,
                     topology const * topo);

    int info_tensor(int *     order,
                    int **    edge_len,
                    int **    sym) const;

    /* set the tensor name */
    int set_name(int tensor_id, char const * name);

    /* get the tensor name */
    int get_name(int tensor_id, char const ** name);

    /* turn on profiling */
    int profile_on(int tensor_id);

    /* turn off profiling */
    int profile_off(int tensor_id);

    /* get dimension of a tensor */
    int get_dimension(int tensor_id, int *order) const;

    /* get lengths of a tensor */
    int get_lengths(int tensor_id, int **edge_len) const;

    /* get symmetry of a tensor */
    int get_symmetry(int tensor_id, int **sym) const;

    /* get raw data pointer WARNING: includes padding */
    int get_raw_data(int tensor_id, char ** data, int64_t * size);

};

}

#endif// __INT_TENSOR_H__

