#include <stdio.h>

__device__ double HammingDistance(int* baseColIdx, int* refColIdx, int baseColSize, int refColSize) {
    int dist = 0;
    int cols = 0;
    int refIdx = 0;

    for (int i = 0; i < baseColSize;) {
        if(baseColIdx[i] == refColIdx[refIdx]) {
            i++;
            refIdx++;
        }else if(baseColIdx[i] < refColIdx[refIdx]) {
            dist++;
            i++;
        }else{ 
            dist++;
            refIdx++;
        }
        cols++;
        // if ref is at end, add all the base as non-equal
        if(refIdx >= refColSize) {
            dist += baseColSize - i;
            cols += baseColSize - i;
            break;
        }
    }
    // if base is at end, add all the red as non-equal
    if(refIdx < refColSize) {
        dist += refColSize - refIdx;
        cols += refColSize - refIdx;
    }
    return (double)dist / (double)cols;
}

__device__ void fine_grouping(int rowIdx, int group_size, int* rowPtr, int* colIdx, float tau) {
    int* baseColIdx;
    int* refColIdx;
    int baseColSize, refColSize = 0;
    // build the base label of the first row
    for(int i=0; i < rowPtr[rowIdx+1] - rowPtr[rowIdx]; i++) {
        baseColIdx[baseColSize] = colIdx[rowPtr[rowIdx] + i];
        baseColSize++;
    }

}

__global__ void gpu_grouping(int* rowPtr, int* colIdx, float tau, int nrows) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int group_size = 0;
    // take four rows for one thread
    if(idx * 4 < nrows) {
        group_size = (nrows - idx*4) > 4? 4:(nrows - idx*4);
        fine_grouping(idx*4, group_size, rowPtr, colIdx, tau, nrows);
    }
}