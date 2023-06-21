#pragma once
#ifndef impl_h
#define impl_h

#include <cuda.h>
#include <cuda_runtime.h>
//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>
//#include <thrust/copy.h>
#include "../utilities.h"

#define rows_per_thread 4

typedef struct{
    int alive;
    int rank; // label size
    int size; // row size
    int* rows;
    int* label; // initialized by size of ncol?
} GroupInfo;

__device__ int change; // global value to check if clustering should finish

//GroupInfo *h_groupInfo;
//GroupInfo *d_groupInfo;
//cudaGetSymbolAddress((void **)&d_groupInfo, "groupInfo");
//cudaMalloc((void **)&d_groupInfo, nrows*sizeof(GroupInfo));
//Testa<<<1,1>>>();
//cudaMemcpy(&h_groupInfo, d_groupInfo, nrows*sizeof(GroupInfo), cudaMemcpyDeviceToHost);
#endif