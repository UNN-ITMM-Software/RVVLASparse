#pragma once
#include <iostream>
#include <cstring>
#include <vector>
#include "CRS.h"

namespace SparseMatrixLib
{

template <typename ValT>
class spMtxSELL_C_Sigma{
public:
  size_t m = 0;
  size_t n = 0;
  size_t cnt_b = 0;
  size_t nz = 0;
  int C = 0;
  int Sigma = 0;
  std::vector<int> vCl;
  std::vector<int> vCs;
  std::vector<int> vBs;
  std::vector<int> vSBs;
  std::vector<uint32_t> vCol;
  std::vector<uint32_t> vPerm;
  std::vector<ValT> vVal;

  std::vector<int> vPartition;

  void freeMem()
  {
    vCol.resize(0);  vCol .shrink_to_fit();
    vCl.resize(0);   vCl  .shrink_to_fit();
    vCs.resize(0);   vCs  .shrink_to_fit();
    vBs.resize(0);   vBs  .shrink_to_fit();
    vSBs.resize(0);  vSBs .shrink_to_fit();
    vPerm.resize(0); vPerm.shrink_to_fit();
    vVal.resize(0);  vVal .shrink_to_fit();
  }

  spMtxSELL_C_Sigma() {}

  spMtxSELL_C_Sigma(const spMtxSELL_C_Sigma &copy):
                    m(copy.m), n(copy.n), nz(copy.nz),  cnt_b(copy.cnt_b),
                    C(copy.C), Sigma(copy.Sigma) {
    vCol = copy.vCol;
    vPerm = copy.vPerm;
    vCl = copy.vCl;
    vCs = copy.vCs;
    vBs = copy.vBs;
    vSBs = copy.vSBs;
    vVal = copy.vVal;
  }

  spMtxSELL_C_Sigma(spMtxSELL_C_Sigma &&mov):
                    m(mov.m), n(mov.n), nz(mov.nz), cnt_b(mov.cnt_b),
                    C(mov.C), Sigma(mov.Sigma),
                    vCol (std::move( mov.vCol )),
                    vCl  (std::move( mov.vCl  )),
                    vCs  (std::move( mov.vCs  )),
                    vBs  (std::move( mov.vBs  )),
                    vSBs (std::move( mov.vSBs )),
                    vPerm(std::move( mov.vPerm)),
                    vVal (std::move( mov.vVal ))
                    { }

  spMtxSELL_C_Sigma& operator=(const spMtxSELL_C_Sigma &copy) {
    if (this == &copy)
      return *this;

    vCol = copy.vCol;
    vPerm = copy.vPerm;
    vCl = copy.vCl;
    vCs = copy.vCs;
    vBs = copy.vBs;
    vSBs = copy.vSBs;
    vVal = copy.vVal;

    m = copy.m;
    n = copy.n;
    C = copy.C;
    nz = copy.nz;
    cnt_b = copy.cnt_b;
    Sigma = copy.Sigma;

    return *this;
  }

};

}