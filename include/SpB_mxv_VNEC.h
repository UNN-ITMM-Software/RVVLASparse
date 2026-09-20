#pragma once
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>
#define always_inline __inline__ __attribute__((always_inline))
#define nLanes_f32 16
#define nLanes_f64 8
#define IRD_thr_fp32 0.437
#define IRD_thr_fp64 0.477

void *align_malloc(size_t size, size_t alignment);

namespace SparseMatrixLib 
{

int get_num_threads(void);

typedef size_t SpB_Index;

enum SpB_VNEC_type
{
    SpB_VNEC_D,
    SpB_VNEC_S,
    SpB_VNEC_L
};

struct coord
{
    int x;
    int y;
};

enum SpB_Type
{
    SpB_NULL_TYPE,
    SpB_BOOL,   ///< in C: bool
    SpB_INT8,   ///< in C: int8_t
    SpB_INT16,  ///< in C: int16_t
    SpB_INT32,  ///< in C: int32_t
    SpB_INT64,  ///< in C: int64_t
    SpB_UINT8,  ///< in C: uint8_t
    SpB_UINT16, ///< in C: uint16_t
    SpB_UINT32, ///< in C: uint32_t
    SpB_UINT64, ///< in C: uint64_t
    SpB_FP32,   ///< in C: float
    SpB_FP64,   ///< in C: double
    SpB_Type_N
};

}