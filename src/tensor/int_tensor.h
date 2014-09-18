/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __INT_TENSOR_H__
#define __INT_TENSOR_H__


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

};

#endif// __INT_TENSOR_H__

