#ifndef __SYM_INDICES_HXX__
#define __SYM_INDICES_HXX__


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

template<typename T>
int relativeSign(const std::vector<T>& s1, const std::vector<T>& s2)
{
    int i, j, k;
    int sign = 1;
    bool *seen = new bool[s1.size()];

    for (i = 0;i < (int)s1.size();i++) seen[i] = false;

    for (i = 0;i < (int)s1.size();i++)
    {
        if (seen[i]) continue;
        j = i;
        while (true)
        {
            for (k = 0;k < (int)s1.size() && (!(s1[k] == s2[j]) || seen[k]);k++);
            assert(k < (int)s1.size());
            j = k;
            seen[j] = true;
            if (j == i) break;
            sign = -sign;
        }
    }

    delete[] seen;

    return sign;
}

inline
double align_symmetric_indices(int ndim_A, int* idx_A, int* sym_A,
                               int ndim_B, int* idx_B, int* sym_B)
{
    int fact = 1;

    std::vector<index_locator_> indices;

    for (int i = 0;i < ndim_A;i++)
    {
        int i_in_B = (int)(std::find(idx_B, idx_B+ndim_B, idx_A[i])-idx_B);
        if (i_in_B == ndim_B) i_in_B = -1;

        if (i_in_B == -1) continue;

        indices.push_back(index_locator_(0, idx_A[i], i, i_in_B, 0));
    }

    while (!indices.empty())
    {
        std::vector<index_locator_> group;
        group.push_back(indices[0]);
        group.back().sort = 0;
        indices.erase(indices.begin());

        int s = 1;
        for (std::vector<index_locator_>::iterator it = indices.begin();
             it != indices.end();++it)
        {
            if ((group[0].pos_A == -1 && it->pos_A != -1) ||
                (group[0].pos_A != -1 && it->pos_A == -1) ||
                (group[0].pos_B == -1 && it->pos_B != -1) ||
                (group[0].pos_B != -1 && it->pos_B == -1)) continue;

            bool sym_in_A = false;
            for (int k = group[0].pos_A-1;k >= 0 && sym_A[k] != NS;k--)
            {
                if (idx_A[k] == it->idx)
                {
                    sym_in_A = true;
                    break;
                }
            }
            for (int k = group[0].pos_A+1;k < ndim_A && sym_A[k-1] != NS;k++)
            {
                if (idx_A[k] == it->idx)
                {
                    sym_in_A = true;
                    break;
                }
            }
            if (!sym_in_A) continue;

            bool sym_in_B = false;
            for (int k = group[0].pos_B-1;k >= 0 && sym_B[k] != NS;k--)
            {
                if (idx_B[k] == it->idx)
                {
                    sym_in_B = true;
                    break;
                }
            }
            for (int k = group[0].pos_B+1;k < ndim_B && sym_B[k-1] != NS;k++)
            {
                if (idx_B[k] == it->idx)
                {
                    sym_in_B = true;
                    break;
                }
            }
            if (!sym_in_B) continue;

            group.push_back(*it);
            group.back().sort = s++;
            it = indices.erase(it)-1;
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
    //    for (int i = 0;i < ndim_A;i++) std::cout << idx_A[i] << ' ';
    //    std::cout << std::endl;
    //    for (int i = 0;i < ndim_B;i++) std::cout << idx_B[i] << ' ';
    //    std::cout << std::endl;
    //}

    return (double)fact;
}

inline
double align_symmetric_indices(int ndim_A, int* idx_A, int* sym_A,
                               int ndim_B, int* idx_B, int* sym_B,
                               int ndim_C, int* idx_C, int* sym_C)
{
    int fact = 1;

    std::vector<index_locator_> indices;

    for (int i = 0;i < ndim_A;i++)
    {
        int i_in_B = (int)(std::find(idx_B, idx_B+ndim_B, idx_A[i])-idx_B);
        if (i_in_B == ndim_B) i_in_B = -1;

        int i_in_C = (int)(std::find(idx_C, idx_C+ndim_C, idx_A[i])-idx_C);
        if (i_in_C == ndim_C) i_in_C = -1;

        if (i_in_B == -1 && i_in_C == -1) continue;

        indices.push_back(index_locator_(0, idx_A[i], i, i_in_B, i_in_C));
    }

    for (int i = 0;i < ndim_B;i++)
    {
        int i_in_A = (int)(std::find(idx_A, idx_A+ndim_A, idx_B[i])-idx_A);
        if (i_in_A == ndim_A) i_in_A = -1;

        int i_in_C = (int)(std::find(idx_C, idx_C+ndim_C, idx_B[i])-idx_C);
        if (i_in_C == ndim_C) i_in_C = -1;

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
        for (std::vector<index_locator_>::iterator it = indices.begin();
             it != indices.end();++it)
        {
            if ((group[0].pos_A == -1 && it->pos_A != -1) ||
                (group[0].pos_A != -1 && it->pos_A == -1) ||
                (group[0].pos_B == -1 && it->pos_B != -1) ||
                (group[0].pos_B != -1 && it->pos_B == -1) ||
                (group[0].pos_C == -1 && it->pos_C != -1) ||
                (group[0].pos_C != -1 && it->pos_C == -1)) continue;

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
                for (int k = group[0].pos_A+1;k < ndim_A && sym_A[k-1] != NS;k++)
                {
                    if (idx_A[k] == it->idx)
                    {
                        sym_in_A = true;
                        break;
                    }
                }
                if (!sym_in_A) continue;
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
                for (int k = group[0].pos_B+1;k < ndim_B && sym_B[k-1] != NS;k++)
                {
                    if (idx_B[k] == it->idx)
                    {
                        sym_in_B = true;
                        break;
                    }
                }
                if (!sym_in_B) continue;
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
                for (int k = group[0].pos_C+1;k < ndim_C && sym_C[k-1] != NS;k++)
                {
                    if (idx_C[k] == it->idx)
                    {
                        sym_in_C = true;
                        break;
                    }
                }
                if (!sym_in_C) continue;
            }

            group.push_back(*it);
            group.back().sort = s++;
            it = indices.erase(it)-1;
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
    //    for (int i = 0;i < ndim_A;i++) std::cout << idx_A[i] << ' ';
    //    std::cout << std::endl;
    //    for (int i = 0;i < ndim_B;i++) std::cout << idx_B[i] << ' ';
    //    std::cout << std::endl;
    //    for (int i = 0;i < ndim_C;i++) std::cout << idx_C[i] << ' ';
    //    std::cout << std::endl;
    //}

    return (double)fact;
}

inline
double overcounting_factor(int ndim_A, int* idx_A, int* sym_A,
                           int ndim_B, int* idx_B, int* sym_B,
                           int ndim_C, int* idx_C, int* sym_C)
{
    int fact = 1;

    for (int i = 0;i < ndim_A;i++)
    {
        int j;
        for (j = 0;j < ndim_B && idx_A[i] != idx_B[j];j++);
        if (j == ndim_B) continue;

        int k;
        for (k = 0;k < ndim_C && idx_A[i] != idx_C[k];k++);
        if (k != ndim_C) continue;

        int ninarow = 1;
        while (i < ndim_A &&
               j < ndim_B &&
               sym_A[i] != NS &&
               sym_B[j] != NS &&
               idx_A[i] == idx_B[j])
        {
            ninarow++;
            i++;
            j++;
        }
        if (i < ndim_A &&
            j < ndim_B &&
            idx_A[i] != idx_B[j]) ninarow--;

        for (;ninarow > 1;ninarow--) fact *= ninarow;
    }

    return (double)fact;
}
#endif
