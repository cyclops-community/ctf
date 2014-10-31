/** Written by Devin Matthews */

#ifndef __INT_SYM_INDICES_H__
#define __INT_SYM_INDICES_H__

#include "assert.h"

template<typename RAIterator>
int relativeSign(RAIterator s1b, RAIterator s1e, RAIterator s2b, RAIterator s2e);

template<typename T>
int relativeSign(const T& s1, const T& s2);

template <typename T>
int align_symmetric_indices(int order_A, T& idx_A, const int* sym_A,
                               int order_B, T& idx_B, const int* sym_B);

template <typename T>
int align_symmetric_indices(int order_A, T& idx_A, const int* sym_A,
                               int order_B, T& idx_B, const int* sym_B,
                               int order_C, T& idx_C, const int* sym_C);

template <typename T>
int overcounting_factor(int order_A, const T& idx_A, const int* sym_A,
                           int order_B, const T& idx_B, const int* sym_B,
                           int order_C, const T& idx_C, const int* sym_C);

template <typename T>
int overcounting_factor(int order_A, const T& idx_A, const int* sym_A,
                           int order_B, const T& idx_B, const int* sym_B);

#endif

