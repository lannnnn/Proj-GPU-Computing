#pragma once
#ifndef impl_cuh
#define impl_cuh

#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "../utilities.h"

#define ref_size 4

#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)

typedef struct{
    int alive;
    int rank; // label size
    int size; // row size
    int* rows;
    int* label; // initialized by size of ncol?
} GroupInfo;

// __device__ int change; // global value to check if clustering should finish
__device__ volatile int g_mutex;
__device__ volatile int o_mutex;
//GroupInfo *h_groupInfo;
//GroupInfo *d_groupInfo;
//cudaGetSymbolAddress((void **)&d_groupInfo, "groupInfo");
//cudaMalloc((void **)&d_groupInfo, nrows*sizeof(GroupInfo));
//Testa<<<1,1>>>();
//cudaMemcpy(&h_groupInfo, d_groupInfo, nrows*sizeof(GroupInfo), cudaMemcpyDeviceToHost);
__global__ void test(int* groupList, int* resultList);
__global__ void gpu_grouping(int* rowPtr, int* colIdx, float tau, int* groupList, GroupInfo* groupInfo, 
                                    int* resultList, int* groupSize, int goalVal, int block_cols);
__global__ void gpu_ref_grouping(int* rowPtr, int* colIdx, float* tau, int* groupList, int* groupSize, int* refRow);
__global__ void gpu_ref_grouping_O1(int* rowPtr, int* colIdx, float* tau, int* groupList, int* groupSize, int* refRow, int* rows_per_thread);

#endif