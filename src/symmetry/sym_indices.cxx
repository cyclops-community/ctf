/** Written by Devin Matthews */

#include "../interface/common.h"
#include "sym_indices.h"

using namespace CTF;

struct index_locator_
{
    int sort;
    int idx;
    int pos_A;
    int pos_B;
    int pos_C;

    index_locator_(int sort, int idx, int pos_A, int pos_B, int pos_C)
    : sort(sort), idx(idx), pos_A(pos_A), pos_B(pos_B), pos_C(pos_C) {}

    static bool sortA(const index_locator_& a, const index_locator_& b)
    {
        return a.pos_A < b.pos_A;
    }

    static bool sortB(const index_locator_& a, const index_locator_& b)
    {
        return a.pos_B < b.pos_B;
    }

    static bool sortC(const index_locator_& a, const index_locator_& b)
    {
        return a.pos_C < b.pos_C;
    }

    bool operator==(int idx)
    {
        return this->idx == idx;
    }
};
template <typename T>
int align_symmetric_indices(int order_A, T& idx_A, const int* sym_A,
                               int order_B, T& idx_B, const int* sym_B)
{
    int fact = 1;

    std::vector<index_locator_> indices;

    for (int i = 0;i < order_A;i++)
    {
        int i_in_B; for (i_in_B = 0;i_in_B < order_B && idx_A[i] != idx_B[i_in_B];i_in_B++);
        if (i_in_B == order_B) continue;

        indices.push_back(index_locator_(0, idx_A[i], i, i_in_B, 0));
    }

    while (!indices.empty())
    {
        std::vector<index_locator_> group;
        group.push_back(indices[0]);
        group.back().sort = 0;
        indices.erase(indices.begin());

        int s = 1;
        for (std::vector<index_locator_>::iterator it = indices.begin();;)
        {
            if (it == indices.end()) break;

            if ((group[0].pos_A == -1 && it->pos_A != -1) ||
                (group[0].pos_A != -1 && it->pos_A == -1) ||
                (group[0].pos_B == -1 && it->pos_B != -1) ||
                (group[0].pos_B != -1 && it->pos_B == -1))
            {
                ++it;
                continue;
            }

            bool sym_in_A = false;
            for (int k = group[0].pos_A-1;k >= 0 && sym_A[k] != NS;k--)
            {
                if (idx_A[k] == it->idx)
                {
                    sym_in_A = true;
                    break;
                }
            }
            for (int k = group[0].pos_A+1;k < order_A && sym_A[k-1] != NS;k++)
            {
                if (idx_A[k] == it->idx)
                {
                    sym_in_A = true;
                    break;
                }
            }
            if (!sym_in_A)
            {
                ++it;
                continue;
            }

            bool sym_in_B = false;
            for (int k = group[0].pos_B-1;k >= 0 && sym_B[k] != NS;k--)
            {
                if (idx_B[k] == it->idx)
                {
                    sym_in_B = true;
                    break;
                }
            }
            for (int k = group[0].pos_B+1;k < order_B && sym_B[k-1] != NS;k++)
            {
                if (idx_B[k] == it->idx)
                {
                    sym_in_B = true;
                    break;
                }
            }
            if (!sym_in_B)
            {
                ++it;
                continue;
            }

            group.push_back(*it);
            group.back().sort = s++;
            it = indices.erase(it);
        }

        if (group.size() <= 1) continue;

        std::vector<int> order_A, order_B;

        for (int i = 0;i < (int)group.size();i++)
            order_A.push_back(group[i].sort);

        std::sort(group.begin(), group.end(), index_locator_::sortB);
        for (int i = 0;i < (int)group.size();i++)
        {
            order_B.push_back(group[i].sort);
            idx_B[group[group[i].sort].pos_B] = group[i].idx;
        }
        if (sym_B[group[0].pos_B] == AS)
            fact *= relativeSign(order_A, order_B);
    }

    //if (fact != 1)
    //{
    //    std::cout << "I got a -1 !!!!!" << std::endl;
    //    for (int i = 0;i < order_A;i++) std::cout << idx_A[i] << ' ';
    //    std::cout << std::endl;
    //    for (int i = 0;i < order_B;i++) std::cout << idx_B[i] << ' ';
    //    std::cout << std::endl;
    //}

    return fact;
}

template <typename T>
int align_symmetric_indices(int order_A, T& idx_A, const int* sym_A,
                               int order_B, T& idx_B, const int* sym_B,
                               int order_C, T& idx_C, const int* sym_C)
{
    int fact = 1;

    std::vector<index_locator_> indices;

    for (int i = 0;i < order_A;i++)
    {
        int i_in_B; for (i_in_B = 0;i_in_B < order_B && idx_A[i] != idx_B[i_in_B];i_in_B++);
        if (i_in_B == order_B) i_in_B = -1;

        int i_in_C; for (i_in_C = 0;i_in_C < order_C && idx_A[i] != idx_C[i_in_C];i_in_C++);
        if (i_in_C == order_C) i_in_C = -1;

        if (i_in_B == -1 && i_in_C == -1) continue;

        indices.push_back(index_locator_(0, idx_A[i], i, i_in_B, i_in_C));
    }

    for (int i = 0;i < order_B;i++)
    {
        int i_in_A; for (i_in_A = 0;i_in_A < order_A && idx_B[i] != idx_A[i_in_A];i_in_A++);
        if (i_in_A == order_A) i_in_A = -1;

        int i_in_C; for (i_in_C = 0;i_in_C < order_C && idx_B[i] != idx_C[i_in_C];i_in_C++);
        if (i_in_C == order_C) i_in_C = -1;

        if (i_in_A != -1 || i_in_C == -1) continue;

        indices.push_back(index_locator_(0, idx_B[i], i_in_A, i, i_in_C));
    }

    while (!indices.empty())
    {
        std::vector<index_locator_> group;
        group.push_back(indices[0]);
        group.back().sort = 0;
        indices.erase(indices.begin());

        int s = 1;
        for (std::vector<index_locator_>::iterator it = indices.begin();;)
        {
            if (it == indices.end()) break;

            if ((group[0].pos_A == -1 && it->pos_A != -1) ||
                (group[0].pos_A != -1 && it->pos_A == -1) ||
                (group[0].pos_B == -1 && it->pos_B != -1) ||
                (group[0].pos_B != -1 && it->pos_B == -1) ||
                (group[0].pos_C == -1 && it->pos_C != -1) ||
                (group[0].pos_C != -1 && it->pos_C == -1))
            {
                ++it;
                continue;
            }

            if (group[0].pos_A != -1)
            {
                bool sym_in_A = false;
                for (int k = group[0].pos_A-1;k >= 0 && sym_A[k] != NS;k--)
                {
                    if (idx_A[k] == it->idx)
                    {
                        sym_in_A = true;
                        break;
                    }
                }
                for (int k = group[0].pos_A+1;k < order_A && sym_A[k-1] != NS;k++)
                {
                    if (idx_A[k] == it->idx)
                    {
                        sym_in_A = true;
                        break;
                    }
                }
                if (!sym_in_A)
                {
                    ++it;
                    continue;
                }
            }

            if (group[0].pos_B != -1)
            {
                bool sym_in_B = false;
                for (int k = group[0].pos_B-1;k >= 0 && sym_B[k] != NS;k--)
                {
                    if (idx_B[k] == it->idx)
                    {
                        sym_in_B = true;
                        break;
                    }
                }
                for (int k = group[0].pos_B+1;k < order_B && sym_B[k-1] != NS;k++)
                {
                    if (idx_B[k] == it->idx)
                    {
                        sym_in_B = true;
                        break;
                    }
                }
                if (!sym_in_B)
                {
                    ++it;
                    continue;
                }
            }

            if (group[0].pos_C != -1)
            {
                bool sym_in_C = false;
                for (int k = group[0].pos_C-1;k >= 0 && sym_C[k] != NS;k--)
                {
                    if (idx_C[k] == it->idx)
                    {
                        sym_in_C = true;
                        break;
                    }
                }
                for (int k = group[0].pos_C+1;k < order_C && sym_C[k-1] != NS;k++)
                {
                    if (idx_C[k] == it->idx)
                    {
                        sym_in_C = true;
                        break;
                    }
                }
                if (!sym_in_C)
                {
                    ++it;
                    continue;
                }
            }

            group.push_back(*it);
            group.back().sort = s++;
            it = indices.erase(it);
        }

        if (group.size() <= 1) continue;

        std::vector<int> order_A, order_B, order_C;

        if (group[0].pos_A != -1)
        {
            for (int i = 0;i < (int)group.size();i++)
                order_A.push_back(group[i].sort);

            if (group[0].pos_B != -1)
            {
                std::sort(group.begin(), group.end(), index_locator_::sortB);
                for (int i = 0;i < (int)group.size();i++)
                {
                    order_B.push_back(group[i].sort);
                    idx_B[group[group[i].sort].pos_B] = group[i].idx;
                }
                if (sym_B[group[0].pos_B] == AS)
                    fact *= relativeSign(order_A, order_B);
            }

            if (group[0].pos_C != -1)
            {
                std::sort(group.begin(), group.end(), index_locator_::sortC);
                for (int i = 0;i < (int)group.size();i++)
                {
                    order_C.push_back(group[i].sort);
                    idx_C[group[group[i].sort].pos_C] = group[i].idx;
                }
                if (sym_C[group[0].pos_C] == AS)
                    fact *= relativeSign(order_A, order_C);
            }
        }
        else
        {
            for (int i = 0;i < (int)group.size();i++)
                order_B.push_back(group[i].sort);

            std::sort(group.begin(), group.end(), index_locator_::sortC);
            for (int i = 0;i < (int)group.size();i++)
            {
                order_C.push_back(group[i].sort);
                idx_C[group[group[i].sort].pos_C] = group[i].idx;
            }
            if (sym_C[group[0].pos_C] == AS)
                fact *= relativeSign(order_B, order_C);
        }
    }

    //if (fact != 1)
    //{
    //    std::cout << "I got a -1 !!!!!" << std::endl;
    //    for (int i = 0;i < order_A;i++) std::cout << idx_A[i] << ' ';
    //    std::cout << std::endl;
    //    for (int i = 0;i < order_B;i++) std::cout << idx_B[i] << ' ';
    //    std::cout << std::endl;
    //    for (int i = 0;i < order_C;i++) std::cout << idx_C[i] << ' ';
    //    std::cout << std::endl;
    //}

    return fact;
}

template <typename T>
int overcounting_factor(int order_A, const T& idx_A, const int* sym_A,
                           int order_B, const T& idx_B, const int* sym_B,
                           int order_C, const T& idx_C, const int* sym_C)
{
    int fact = 1;

    for (int i = 0;i < order_A;i++)
    {
        int j;
        for (j = 0;j < order_B && idx_A[i] != idx_B[j];j++);
        if (j == order_B) continue;

        int k;
        for (k = 0;k < order_C && idx_A[i] != idx_C[k];k++);
        if (k != order_C) continue;

        int ninarow = 1;
        while (i < order_A &&
               j < order_B &&
               sym_A[i] != NS &&
               sym_B[j] != NS &&
               idx_A[i] == idx_B[j])
        {
            ninarow++;
            i++;
            j++;
        }
        if (i < order_A &&
            j < order_B &&
            idx_A[i] != idx_B[j]) ninarow--;

        if (ninarow >= 2){
          //if (sym_A[i-ninarow+1]!=SY) 
          for (;ninarow > 1;ninarow--) fact *= ninarow;
        }
    }

    return fact;
}

template <typename T>
int overcounting_factor(int order_A, const T& idx_A, const int* sym_A,
                           int order_B, const T& idx_B, const int* sym_B)
{
    int fact;
    int ninarow;
    fact = 1.0;

    for (int i = 0;i < order_A;i++)
    {
        int j;
        ninarow = 0;
        for (j = 0;j < order_B && idx_A[i] != idx_B[j];j++);
        if (j>=order_B){
            ninarow = 1;
            while (sym_A[i] != NS)
            {
                i++;
                for (j = 0;j < order_B && idx_A[i] != idx_B[j];j++);
                if (j>=order_B) ninarow++;
            }
        }
        if (ninarow >= 2){
            if (sym_A[i-ninarow+1]==AS) return 0.0;
            if (sym_A[i-ninarow+1]==SY) {
                /*printf("CTF error: sum over SY index pair currently not functional, ABORTING\n");
                assert(0);*/
            }
            if (sym_A[i-ninarow+1]!=SY) 
              for (;ninarow > 1;ninarow--) fact *= ninarow;
        }
    }
    return fact;
}


template int align_symmetric_indices<int*>(int order_A, int*& idx_A, const int* sym_A,
                               int order_B, int*& idx_B, const int* sym_B);

template int align_symmetric_indices<int*>(int order_A, int*& idx_A, const int* sym_A,
                               int order_B, int*& idx_B, const int* sym_B,
                               int order_C, int*& idx_C, const int* sym_C);

template int overcounting_factor<int*>(int order_A, int * const & idx_A, const int* sym_A,
                           int order_B, int * const & idx_B, const int* sym_B,
                           int order_C, int * const & idx_C, const int* sym_C);

template int overcounting_factor<int*>(int order_A, int * const & idx_A, const int* sym_A,
                           int order_B, int * const & idx_B, const int* sym_B);


template int align_symmetric_indices<std::string>(int order_A, std::string& idx_A, const int* sym_A,
                               int order_B, std::string& idx_B, const int* sym_B);

template int align_symmetric_indices<std::string>(int order_A, std::string& idx_A, const int* sym_A,
                               int order_B, std::string& idx_B, const int* sym_B,
                               int order_C, std::string& idx_C, const int* sym_C);

template int overcounting_factor<std::string>(int order_A, std::string const & idx_A, const int* sym_A,
                           int order_B, std::string const & idx_B, const int* sym_B,
                           int order_C, std::string const & idx_C, const int* sym_C);

template int overcounting_factor<std::string>(int order_A, std::string const & idx_A, const int* sym_A,
                           int order_B, std::string const & idx_B, const int* sym_B);



