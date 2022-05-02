/* The code in this file has been written by Andreas Irmler. */

#include "../tensor/untyped_tensor.h"
#include "../shared/util.h"
#include "node_aware_dist.h"
using ivec  = std::vector<int>;
using vivec = std::vector<ivec>;


namespace CTF_int {


  struct Tree {

    //Copy
    Tree(Tree const &other) {
      order = other.order;
      sgf = other.sgf;
      ogf = other.ogf;
    }

    //Constructor 1
    Tree(int _order, vivec _sgf, vivec _ogf){
      order = _order;
      sgf = _sgf;
      ogf = _ogf;
    }

    // Constructor 2
    Tree(Tree t, int pos, int el){
      order = t.order + 1;
      sgf = t.sgf;
      ogf = t.ogf;
      assert(sgf.size() > pos);
      assert(ogf.size() > pos);
      sgf[pos].push_back(el);
      std::sort(sgf[pos].begin(), sgf[pos].end());
      auto it = std::find(ogf[pos].begin(), ogf[pos].end(), el);
      assert(it != ogf[pos].end());
      ogf[pos].erase(it);
    }

    bool find(int pos, int el) {
      if (ogf.size() <= pos) {
        printf("Find problem! order %d, size: %ld, pos: %d, el: %d\n"
              , order, ogf.size(), pos, el);
        assert(0);
      }
      auto it = std::find(ogf[pos].begin(), ogf[pos].end(), el);
      if (it == ogf[pos].end()) return false;
      return true;
    }

    int order;
    vivec sgf; // settled grid factors. ie factors which are already assigned
    vivec ogf; // open grid factors. factors which can
  };


  // return a vector of prim factors
  ivec iv_factorize(int number){
    ivec factors;
    int n(number);
    if (n < 4) factors.push_back(n);
    int d(2);
    while (d*d <= n)
    while (n>1){
      while (!(n%d)){
        factors.push_back(d);
        n /= d;
      }
      d++;
    }
    return factors;
  }

  // return vector with input arguments
  ivec lineToVint(std::string line) {
    ivec out;
    size_t pos;
    while ((pos = line.find(",")) != std::string::npos) {
      out.push_back(std::stoi(line.substr(0, pos)));
      line.erase(0, pos + 1);
    }
    out.push_back(std::stoi(line));

    return out;
  }


  std::vector< std::vector<int> > get_inter_node_grids(std::vector<int> rGrid, int nodes){
    int ranks(std::accumulate(rGrid.begin(), rGrid.end(), 1, std::multiplies<int>()));
    int ranksPerNode(ranks/nodes);
    IASSERT (ranksPerNode*nodes == ranks );

	  vivec nodeGrid; // final node Grid
    const ivec nodeFactors(iv_factorize(nodes));
    const ivec rankFactors(iv_factorize(ranks));
    vivec gridFactors; // the tensor grid expressed in prim factors
    ivec assignedFactors; // rank factors which are already assigned
    ivec openFactors; // unassigned rank factors
    for (auto r: rGrid) {
      gridFactors.push_back(iv_factorize(r));
    }
    vivec openGridFactors; // grid factors which cannot assigned to a edge

    for (auto gf: gridFactors){

      ivec others, diff;
      // all prim factors which are not at the given edge
      std::set_difference( rankFactors.begin()
                        , rankFactors.end()
                        , gf.begin()
                        , gf.end()
                        , std::back_inserter(others)
                        );
      /*
      for (auto x: others) {
        std::cout << "others: " << x << " ";
      }
      std::cout << std::endl;
      */
      // is there a node factor which lives only on a given edge?
      // if so assign this factor to this edge
      std::set_difference( nodeFactors.begin()
                        , nodeFactors.end()
                        , others.begin()
                        , others.end()
                        , std::back_inserter(diff)
                        );
      assignedFactors.insert(assignedFactors.end(), diff.begin(), diff.end());

      openGridFactors.resize(openGridFactors.size()+1);
      std::set_difference( gf.begin()
                        , gf.end()
                        , diff.begin()
                        , diff.end()
                        , std::back_inserter(openGridFactors.back())
                        );
      if (!diff.size()) diff.push_back(1);
      nodeGrid.push_back(diff);

    }

    std::sort(assignedFactors.begin(), assignedFactors.end());
    std::set_difference( nodeFactors.begin()
                      , nodeFactors.end()
                      , assignedFactors.begin()
                      , assignedFactors.end()
                      , std::back_inserter(openFactors)
                      );
    // The algorithm goes like that:
    // 1.) we pick the last element of the list, remove it from the list,
    //     then open N branches where N is the number of possible possitions
    //     for that element in the rank Grid
    // 2.) we remove identical branches
    // 3.) we go to step 1

    size_t b(0);
    size_t n(rGrid.size());
    std::vector<Tree> treeVec;
    treeVec.emplace_back(0, nodeGrid, openGridFactors);
    // we loop over all prim Factors of the number of nodes
    while (openFactors.size()){
      // take the last element of the list and remove it from the list
      auto f(openFactors.back());
      openFactors.pop_back();

      // we work only in the last layer of the tree
      // we have to find the begin/end in the whole vector
      auto o(treeVec.back().order);
      auto b(std::distance( treeVec.begin()
                          , std::find_if( treeVec.begin()
                                        , treeVec.end()
                                        , [o] (const Tree &a)
                                          { return a.order == o;}
                                        )
                          ));

      auto e(treeVec.size());
      // loop over the last layer of the tree and distribute the
      // element to all possible positions
      // however: if a potential element is already in the list,
      //          do not add it
      for (size_t t(b); t < e; t++){
        for (auto i(0); i < n; i++)
        if ( treeVec[t].find(i, f) ){
          bool distinct(true);
          auto cand = Tree(treeVec[t], i, f);
          for (size_t n(e); n < treeVec.size(); n++){
            if (cand.sgf == treeVec[n].sgf) distinct = false;
          }
          if (distinct) treeVec.push_back(cand);
        }
      }
    }

    std::vector< std::vector<int> > inter_node_grids;
    for (auto tv: treeVec) {
      if (treeVec.back().order == tv.order) {
        std::vector<int> sgf;
        for (auto s: tv.sgf) {
          sgf.push_back(std::accumulate(s.begin(), s.end(), 1, std::multiplies<int>()));
        }
        inter_node_grids.push_back(sgf);
      }
    }
    return inter_node_grids;
  }
}
