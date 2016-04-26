#ifndef __GROUP_H__
#define __GROUP_H__

#include "../tensor/algstrct.h"

namespace CTF {
  /**
   * \addtogroup algstrct 
   * @{
   **/
  /**
   * \brief Group is a Monoid with operator '-' defined
   *   special case (parent) of a ring
   */
  template <typename dtype=double, bool is_ord=CTF_int::get_default_is_ord<dtype>()> 
  class Group : public Monoid<dtype, is_ord> {
    public:
      Group(Group const & other) : Monoid<dtype, is_ord>(other) { }

      virtual CTF_int::algstrct * clone() const {
        return new Group<dtype, is_ord>(*this);
      }

      Group() : Monoid<dtype, is_ord>() { 
        this->abs = &CTF_int::char_abs< dtype, CTF_int::default_abs<dtype, is_ord> >;
      } 

      Group(dtype taddid_,
            dtype (*fadd_)(dtype a, dtype b),
            MPI_Op addmop_)
              : Monoid<dtype, is_ord>(taddid_, fadd_, addmop_) { 
        this->abs = &CTF_int::char_abs< dtype, CTF_int::default_abs<dtype, is_ord> >;
      }

      //treat NULL as mulid
      void safeaddinv(char const * a, char *& b) const {
        if (a == NULL){
          printf("CTF ERROR: unfortunately additive inverse functionality for groups is currently limited, as it is done for rings via scaling by the inverse of the multiplicative identity, which groups don't have. Use the tensor addinv function rather than an indexed expression.\n");
          double * ptr = NULL;
          ptr[0]=3.;
          assert(0);
        } else {
          if (b==NULL) b = (char*)malloc(this->el_size);
          ((dtype*)b)[0] = -((dtype*)a)[0];
        }
      }

      void addinv(char const * a, char * b) const {
        ((dtype*)b)[0] = -((dtype*)a)[0];
      }
  };

  /**
   * @}
   */
}
#include "semiring.h"
#endif
