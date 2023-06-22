#pragma once
#ifndef impl_cuh
#define impl_cuh

#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "../utilities.h"

using namespace cooperative_groups; 

#define rows_per_thread 4

typedef struct{
    int alive;
    int rank; // label size
    int size; // row size
    int* rows;
    int* label; // initialized by size of ncol?
} GroupInfo;

__device__ int change = 1; // global value to check if clustering should finish

//GroupInfo *h_groupInfo;
//GroupInfo *d_groupInfo;
//cudaGetSymbolAddress((void **)&d_groupInfo, "groupInfo");
//cudaMalloc((void **)&d_groupInfo, nrows*sizeof(GroupInfo));
//Testa<<<1,1>>>();
//cudaMemcpy(&h_groupInfo, d_groupInfo, nrows*sizeof(GroupInfo), cudaMemcpyDeviceToHost);
__global__ void test(int* groupList, int* resultList);
__global__ void gpu_grouping(int* rowPtr, int* colIdx, float tau, int* groupList, GroupInfo* groupInfo, int* resultList, int groupSize, int nnz);

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
#endif