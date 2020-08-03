#ifndef __PARTITION_H__
#define __PARTITION_H__


namespace CTF {

  /**
   * \defgroup CTF_part Partition/Decomposition interface
   * \addtogroup CTF_part
   * @{
   */
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
      char * idx;
      Idx_Partition();
      ~Idx_Partition();
      Idx_Partition(Partition const & part, char const * idx);

      void operator=(Idx_Partition const & other);

      /**
       * \brief extracts non-trivial part of partition by ommitting unit dimensions
       * \return new partition with all dimensions non-unit
       */
      Idx_Partition reduce_order() const;
  };

/**
 * @}
 */
}

#endif
