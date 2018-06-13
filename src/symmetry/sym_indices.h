/** Written by Devin Matthews */

#ifndef __INT_SYM_INDICES_H__
#define __INT_SYM_INDICES_H__

#include <assert.h>

template<typename RAIterator>
int relativeSign(RAIterator s1b, RAIterator s1e, RAIterator s2b, RAIterator s2e)
{
    int sz = s1e-s1b;
    assert(sz == (int)(s2e-s2b));
    int i, k;
    int sign = 1;
    std::vector<bool> seen(sz);

    for (i = 0;i < sz;i++) seen[i] = false;

    for (i = 0;i < sz;i++)
    {
        if (seen[i]) continue;
        int j = i;
        while (true)
        {
            for (k = 0;k < sz && (!(*(s1b+k) == *(s2b+j)) || seen[k]);k++);
            assert(k < sz);
            j = k;
            seen[j] = true;
            if (j == i) break;
            sign = -sign;
        }
    }

    return sign;
}

template<typename T>
int relativeSign(const T& s1, const T& s2)
{
    return relativeSign(s1.begin(), s1.end(), s2.begin(), s2.end());
}

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

