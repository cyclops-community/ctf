#ifndef __PARTITION_H__
#define __PARTITION_H__

/**
 * \defgroup CTF CTF: C++ Partition/Decomposition interface
 * @{
 */

namespace CTF {

  class Idx_Partition;

  class Partition {
    public:
      int order;
      int * lens;

      Partition(int order, int const * lens);
      ~Partition();
      Partition(Partition const & other);
      Partition();

      Idx_Partition operator[](char const * idx);
      void operator=(Partition const & other);
  };

  class Idx_Partition {
    public:
      Partition part;
      char const * idx;
      Idx_Partition();
      Idx_Partition(Partition const & part, char const * idx);
  };

}

/**
 * @}
 */
#endif
