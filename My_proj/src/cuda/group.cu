#include <stdio.h>
#include "cuda_impl.h"
#include <thrust/copy.h>

__device__ double HammingDistance(thrust::device_vector<int> baseColIdx, thrust::device_vector<int> refColIdx) {
    int dist = 0;
    int cols = baseColSize.size() + refColSize.size();
    int refIdx = 0;

    for (int i = 0; i < baseColSize.size(); i++) {
        for(int j=0; j < refColSize.size(); j++) {
            if(baseColIdx[i] == refColIdx[j]) {
                cols --;
                break;
            }
        }
        dist++;
    }
    return (double)dist / (double)cols;
}

__device__ void fine_grouping(int rowIdx, int group_size, CSR &csr, double tau, int ncols) {
    // thrust::device_vector<int> baseColIdx;
    // thrust::device_vector<int> refColIdx;
    // thrust::device_vector<int> remainList;
    // int rowsRemain = remainList.size();

    // // list the remainning ref rows
    // for(int i=0; i<rowsRemain; i++) {
    //     remainList.push_back(rowIdx+i);
    // }

    // // build the base label of the first row
    // for(int i=0; i < csr.rowPtr[rowIdx+1] - csr.rowPtr[rowIdx]; i++) {
    //     baseColIdx[baseColSize] = csr.colIdx[csr.rowPtr[rowIdx] + i];
    //     baseColSize++;
    // }


    // while(rowsRemain > 0) {
    //     rowIdx = remainList[0];
    //     // build the base label of the first ref row
    //     for(int i=0; i < csr.rowPtr[rowIdx+1] - csr.rowPtr[rowIdx]; i++) {
    //         refColIdx[refColSize] = csr.colIdx[csr.rowPtr[rowIdx] + i];
    //         refColSize++;
    //     }
    //     if(HammingDistance(baseColIdx, refColIdx, baseColSize, refColSize) < tau) {

    //     } else {

    //     }
    // }
}

__device__ void calculate_label(CSR &csr, int rowIdx, int group_size, thrust::device_vector<GroupRow> groupRows) {
    for(int i=0; i<group_size, i++) {
        groupRows[rowIdx].rowIdx.push_back(rowIdx);
        for(int j=0; j < csr.rowPtr[rowIdx+1] - csr.rowPtr[rowIdx]; j++) {
            col = csr.colIdx[csr.rowPtr[rowIdx] + j];
            groupRows[rowIdx].lable.push_back(col);
        }
        rowIdx++;
    }
}

__global__ void gpu_grouping(CSR &csr, float tau, int nrows, int ncols, thrust::device_vector<GroupRow> groupRows) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int group_size = 0;
    // take four rows for one thread
    int rowIdx = idx * rows_per_thread;
    if(rowIdx < nrows) {
        group_size = (nrows - rowIdx) > rows_per_thread? rows_per_thread:(nrows - rowIdx);
    };
    // create the label for each line
    calculate_label(csr, rowIdx, group_size, groupRows);
    // calculate the distance
    //fine_grouping(rowIdx, group_size, csr, tau, ncols);
}