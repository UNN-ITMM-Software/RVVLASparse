/* 
*========================================================
 * Copyright (c) RVVLASparse and Lobachevsky State University of 
 * Nizhny Novgorod and its affiliates. All rights reserved.
 * 
 * Copyright 2026 The RVVLASparse Authors (Evgeny Kozinov)
 *
 * Distributed under the MIT License
 * (See file LICENSE in the root directory of this 
 * source tree)
 *========================================================
 */
#include <immintrin.h>

void test()
{
  __m512i zmm = _mm512_setzero_si512();
}

int main() 
{
  test();
  return 0; 
}
