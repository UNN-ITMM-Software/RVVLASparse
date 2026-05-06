#pragma once
#include <iostream>
#include <cstring>

namespace SparseMatrixLib
{

typedef char MM_typecode[4];

template <typename ValT>
class spMtxCRS{
public:
  size_t m = 0;
  size_t n = 0;
  size_t nz = 0;
  size_t capacity = 0;
  MM_typecode matcode;
  int* Rst = nullptr;
  int* Col = nullptr;
  ValT* Val = nullptr;


  void freeMem()
  {
    if (Col)
      delete[] Col;
    if (Rst)
      delete[] Rst;
    if (Val)
      delete[] Val;
    
    Col = nullptr;
    Rst = nullptr;
    Val = nullptr;
    m = 0;
    n = 0;
    nz = 0;
    capacity = 0;
  }

  spMtxCRS() {}

  spMtxCRS(size_t _m, size_t _n): m(_m), n(_n) {
    Rst = new int[m+1]();
  }

  spMtxCRS(size_t _m, size_t _n, size_t _nz): m(_m), n(_n), nz(_nz) {
    Rst = new int[m+1];
    Col = new int[nz];
    Val = new ValT[nz];
    capacity = nz;
  }

  spMtxCRS(const spMtxCRS &copy): m(copy.m), n(copy.n), nz(copy.nz), capacity(copy.capacity) {
    for (int i = 0; i < 4; ++i)
      matcode[i] = copy.matcode[i];
    Col = new int[nz];
    memcpy(Col, copy.Col, nz*sizeof(int));
    Rst = new int[m+1];
    memcpy(Rst, copy.Rst, (m+1)*sizeof(int));
    if (copy.Val != nullptr) {
      Val = new ValT[nz];
      memcpy(Val, copy.Val, nz*sizeof(ValT));
    }
  }

  spMtxCRS(const size_t _m, const size_t _n, const size_t _nz, const int* _Rst, const int* _Col, const ValT* _Val): m(_m), n(_n), nz(_nz), capacity(_nz) {    
    Col = new int[nz];
    memcpy(Col, _Col, nz*sizeof(int));
    Rst = new int[m+1];
    memcpy(Rst, _Rst, (m+1)*sizeof(int));
    if (_Val != nullptr) {
        Val = new ValT[nz];
        memcpy(Val, _Val, nz * sizeof(ValT));
    }
  }

  spMtxCRS(spMtxCRS &&mov): m(mov.m), n(mov.n), nz(mov.nz), capacity(mov.capacity) {
    for (int i = 0; i < 4; ++i)
      matcode[i] = mov.matcode[i];
    Col = mov.Col;
    Rst = mov.Rst;
    Val = mov.Val;

    mov.Col = nullptr;
    mov.Rst = nullptr;
    mov.Val = nullptr;
  }

  ~spMtxCRS() {
    freeMem();
  }

  void resizeVals(size_t newNz) {
    if (newNz > capacity) {
      if (Col != nullptr)
        delete[] Col;
      if (Val != nullptr)
        delete[] Val;
      Col = new  int[newNz];
      Val = new ValT[newNz];
      capacity = newNz;
    }
    nz = newNz;
  }

  void resizeRows(size_t newM) {
    if (m != newM) {
      if (Rst != nullptr)
        delete[] Rst;
      Rst = new int[newM + 1]();
      m = newM;
    }
  }

  // Копирование структуры матрицы без копирования значений
  template <typename ValT2>
  void copyPattern(const spMtxCRS<ValT2> &source) {
    resizeRows(source.m);
    memcpy(Rst, source.Rst, (m + 1) * sizeof(int));
    n = source.n;

    resizeVals(source.nz);
    memcpy(Col, source.Col, nz * sizeof(int));
  }

  spMtxCRS& operator=(const spMtxCRS &copy) {
    if (this == &copy)
      return *this;

    for (int i = 0; i < 4; ++i)
      matcode[i] = copy.matcode[i];
    if (m != copy.m) {
      if (Rst)
        delete[] Rst;
      Rst = new int[copy.m + 1];
    }
    if (copy.Rst != nullptr) memcpy(Rst, copy.Rst, (copy.m + 1) * sizeof(int));
    if (capacity < copy.nz) {
      if (Col)
        delete[] Col;
      if (Val)
        delete[] Val;

      Col = new int[copy.nz];
      Val = new ValT[copy.nz];
      capacity = copy.nz;
    }
    memcpy(Col, copy.Col, copy.nz * sizeof(int));
    memcpy(Val, copy.Val, copy.nz * sizeof(ValT));

    m = copy.m;
    n  = copy.n;
    nz = copy.nz;

    return *this;
  }

  spMtxCRS& operator=(spMtxCRS &&mov) {
    if (this == &mov)
      return *this;

    m = mov.m;
    n = mov.n;
    for (int i = 0; i < 4; ++i)
      matcode[i] = mov.matcode[i];
    if (Col)
      delete[] Col;
    if (Rst)
      delete[] Rst;
    if (Val)
      delete[] Val;
    Col = mov.Col;
    Rst = mov.Rst;
    Val = mov.Val;

    mov.Col = nullptr;
    mov.Rst = nullptr;
    mov.Val = nullptr;

    m  = mov.m;
    n  = mov.n;
    nz = mov.nz;
    capacity = mov.capacity;

    return *this;
  }

  spMtxCRS extractRows(size_t begin, size_t end) const {
    spMtxCRS result(end - begin, n, Rst[end] - Rst[begin]);

    for (size_t i = 0; i <= end - begin; ++i)
      result.Rst[i] = Rst[i + begin] - Rst[begin];
    memcpy(result.Col, Col + Rst[begin], (Rst[end] - Rst[begin]) * sizeof(int));
    memcpy(result.Val, Val + Rst[begin], (Rst[end] - Rst[begin]) * sizeof(ValT));
    memcpy(result.matcode, matcode, sizeof(MM_typecode));

    return result;
  }

  bool operator==(const spMtxCRS &other) const {
    if (m != other.m || n != other.n || nz != other.nz)
      return false;
    for (size_t i = 0; i <= m; ++i)
      if (Rst[i] != other.Rst[i])
        return false;
    for (size_t j = 0; j < nz; ++j)
      if (Col[j] != other.Col[j] || Val[j] != other.Val[j])
        return false;
    return true;
  }

  void print_crs() const {
    std::cout << m << ' ' << nz << '\n';
    if (Val == nullptr) {
      for (size_t i = 0; i < m; ++i)
        for (size_t j = Rst[i]; j < Rst[i+1]; ++j)
          std::cout << i+1 << ' ' << Col[j]+1 << '\n';
    }
    else {
      for (size_t i = 0; i < m; ++i)
        for (size_t j = Rst[i]; j < Rst[i+1]; ++j)
          std::cout << i+1 << ' ' << Col[j]+1 << ' ' << Val[j] << '\n';
    }
  }

  void print_dense() const {
    for (size_t i = 0; i < m; ++i) {
      size_t k = 0;
      for (size_t j = Rst[i]; j < Rst[i+1]; ++j, ++k) {
        while (k < Col[j]) {
          std::cout << 0 << ' ';
          ++k;
        }
        std::cout << Val[j] << ' ';
      }
      while (k < n) {
        std::cout << 0 << ' ';
        ++k;
      }
      std::cout << '\n';
    }
    std::cout << '\n';
  }

  friend std::ostream& operator << (std::ostream& out, const spMtxCRS & mtx) {

    out << "spMtxCRS("
        << " m  = " << mtx.m  << ","
        << " n  = " << mtx.n  << ","
        << " nz = " << mtx.nz << ")";

    return out;
  }

};

}