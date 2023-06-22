#include <stdio.h>
#include "cuda_impl.cuh"

__global__ void test(int* groupList, int* resultList) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    resultList[idx] = groupList[idx]*2;
}

__device__ double HammingDistance(GroupInfo &tarrow, GroupInfo &refrow) {
    // if both have no nz value, same
    if(tarrow.rank == 0 && refrow.rank == 0) return 0;
    // if the ref row is empty, return 1;
    if(tarrow.rank == 0 || refrow.rank == 0) return 1;

    int dist = 0;
    int cols = tarrow.rank + refrow.rank;
    int i = 0;
    int j = 0;

    //printf("tarrow.rank = %d\n", tarrow.rank);
    //printf("refrow.rank = %d\n", refrow.rank);

    while(i<tarrow.rank && j<refrow.rank) {
        if(tarrow.label[i] == refrow.label[j]) {
            cols --;
            i++; j++;
        } else if(tarrow.label[i] < refrow.label[j]) {
            dist++; i++;
        } else {
            dist++; j++;
        }
    }
    if(i >= tarrow.rank) dist += (refrow.rank - j);
    if(j >= refrow.rank) dist += (tarrow.rank - i);

    //printf("distance = %f\n", (double) dist / (double) cols);

    return (double) dist / (double) cols;
}

__device__ void combineGroup(GroupInfo &tarrow, GroupInfo &refrow) {
    int rank=0;
    int* tempLabel = new int[tarrow.rank + refrow.rank];

    int size = tarrow.size + refrow.size;
    int* tmpRow = new int[size];

    int i,j;
    
    // combine the label
    for(i=0, j=0; i<tarrow.rank, j<refrow.rank;) {
        if(tarrow.label[i] == refrow.label[j]) {
            tempLabel[rank++] = tarrow.label[i];
            i++; j++;
        } else if(tarrow.label[i] < refrow.label[j]) {
            tempLabel[rank++] = tarrow.label[i];
            i++;
        } else {
            tempLabel[rank++] = refrow.label[j];
            j++;
        }
    }

    if(i >= tarrow.rank) {
        for(i = j; i<refrow.rank; i++) {
            tempLabel[rank++] = refrow.label[i];
        }
    }

    if(j >= refrow.rank) {
        for(j = i; j<tarrow.rank; j++) {
            tempLabel[rank++] = tarrow.label[j];
        }
    }

    // combine the rows
    for(i=0; i<tarrow.size; i++) {
        tmpRow[i] = tarrow.rows[i];
    }
    i--;

    for(j=0; j<refrow.size; j++) {
        tmpRow[i+j] = refrow.rows[j];
    }

    // update the target group
    tarrow.rank = rank;
    tarrow.label = tempLabel;
    tarrow.size = size;
    tarrow.rows = tmpRow;

    // clear the refrow
    refrow.rank = 0;
    refrow.size = 0;
    refrow.label = NULL;
    refrow.rows = NULL;
}

__global__ void gpu_grouping(int* rowPtr, int* colIdx, float tau, int* groupList, GroupInfo* groupInfo, int* resultList, int groupSize, int nnz) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int group_thread = 0;
    // take (rows_per_thread) rows for one thread
    int baserow = idx * rows_per_thread;
    int tarrow, refrow;
    int gap=0;
    cudaError_t error; 

    //multi_grid_group g = this_multi_grid();

    // build the groupInfo
    if(idx < groupSize) {
        groupInfo[idx].alive = 1;
        groupInfo[idx].size = 1;
        //cudaMalloc((int**)&groupInfo[idx].rows, sizeof(int));
        groupInfo[idx].rows=(int*)malloc(sizeof(int));
        groupInfo[idx].rows[0] = idx;
        //cudaMalloc((int**)&groupInfo[idx].label, groupInfo[idx].rank*sizeof(int));
        groupInfo[idx].rank = rowPtr[idx+1] - rowPtr[idx];
        groupInfo[idx].label = (int*)malloc(groupInfo[idx].rank * sizeof(int));
        for(int j=0; j<groupInfo[idx].rank; j++) {
            groupInfo[idx].label[j] = colIdx[rowPtr[idx]+j];
        }
    }

    __syncthreads();

    if(idx%rows_per_thread==0 || idx < groupSize) {
        if((groupSize - idx * rows_per_thread) > rows_per_thread)
            group_thread = rows_per_thread;
        else
            group_thread = groupSize - idx * rows_per_thread;
    }

    // compare between each groups
    if(group_thread > 1) {
        for(int i=0; i<group_thread; i++) {
            tarrow = groupList[baserow+i]; // define the group used as base
            if(tarrow == -1) continue; 
            for(int j=i+1; j<group_thread; j++) {
                refrow = groupList[baserow+j];   // define the group which should be compared for
                if(refrow == -1) continue; 
                // grouping
                if(HammingDistance(groupInfo[tarrow], groupInfo[refrow]) < tau) {
                    combineGroup(groupInfo[tarrow], groupInfo[refrow]);
                    //printf("distance = %f\n", HammingDistance(groupInfo[tarrow], groupInfo[refrow]));
                    groupList[refrow] = -1;
                }
            }
        }
    }
    __syncthreads();

    // // after group, change the groupList in the thread 0
    // if(idx == 0) {
    //     int irow =0;
    //     gap = 0; // initialize the gap to 0
    //     for(int i=0; i<groupSize; i++) {
    //         if(groupList[i] == -1) {
    //             gap++;          // ignore one group
    //         } else {
    //             resultList[irow] = groupList[i];
    //             irow ++;
    //         }
    //     }
    //     if(gap)  {
    //         change = 1; // if any of the group changed, keep changing
    //         resultList[irow] = -1;
    //     }
    // }
    // // wait for groupList change
    // __syncthreads();

    // // organize the result 
    // int irow = 0;
    // if(idx == 0) { 
    //     for(int i=0; i<groupSize; i++) {
    //         for(int j=0; j<groupInfo[resultList[i]].size; j++) {
    //             resultList[irow] = groupInfo[resultList[i]].rows[j];
    //             irow++;
    //         }
    //     }
    // }
    // __syncthreads();
    return;
}