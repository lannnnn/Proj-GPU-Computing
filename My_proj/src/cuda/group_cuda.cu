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

    // printf("tarrow.rank = %d\n", tarrow.rank);
    // printf("refrow.rank = %d\n", refrow.rank);
    // printf("tarrow.label == NULL? %d, tarrow.rank = %d\n", tarrow.label == NULL, tarrow.rank);
    // if(tarrow.label != NULL) printf("tarrow.label[0] = %d", tarrow.label[0]);
    //printf("refrow.label[j] = %d, rank = %d\n", refrow.label[0], refrow.rank);

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
    if(refrow.size == 0) {
        refrow.alive = 0;
        return;
    }
    int size = tarrow.size + refrow.size;
    int* tempLabel = (int*)malloc((tarrow.rank+refrow.rank)*sizeof(int));
    int* tmpRow = (int*)malloc((size)*sizeof(int));

    int rank=0;
    int i=0;
    int j=0;
    
    // combine the label
    while(i<tarrow.rank && j<refrow.rank) {
        // printf("tarrow.label[%d] = %d\n", i, tarrow.label[i]);
        // printf("tarrow.label[%d] = %d\n", j, tarrow.label[j]);
        if(tarrow.label[i] == refrow.label[j]) {
            tempLabel[rank++] = tarrow.label[i++];
            j++;
        } else if(tarrow.label[i] < refrow.label[j]) {
            tempLabel[rank++] = tarrow.label[i++];
        } else {
            tempLabel[rank++] = refrow.label[j++];
        }
    }

    if(i >= tarrow.rank) {
        for(; j<refrow.rank; j++) {
            tempLabel[rank++] = refrow.label[j];
        }
    }

    if(j >= refrow.rank) {
        for(; i<tarrow.rank; i++) {
            tempLabel[rank++] = tarrow.label[i];
        }
    }

    // combine the rows
    for(i=0; i<tarrow.size; i++) {
        tmpRow[i] = tarrow.rows[i];
    }

    for(j=0; j<refrow.size; j++) {
        tmpRow[i+j] = refrow.rows[j];
    }

    // update the target group
    tarrow.rank = rank;
    tarrow.label = tempLabel;
    tarrow.size = size;
    tarrow.rows = tmpRow;

    // clear the refrow
    refrow.alive = 0;
    refrow.rank = 0;
    refrow.size = 0;
    refrow.label = NULL;
    refrow.rows = NULL;
    //printf("refrow.row[j] = %d\n", tmpRow[size-1]);
}

__global__ void gpu_grouping(int* rowPtr, int* colIdx, float tau, int* groupList, GroupInfo* groupInfo, int* resultList, int* groupSize, int nnz) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int group_thread = 0;
    // take (rows_per_thread) rows for one thread
    int baserow = idx * rows_per_thread;
    int tarrow, refrow;
    int gap=0;
    cudaError_t error; 

    // build the groupInfo
    if(idx < groupSize[0]) {
        groupInfo[idx].alive = 1;
        //group row info -> combine the last resultList according to groupInfo
        groupInfo[idx].size = groupList[idx+1]-groupList[idx];
        groupInfo[idx].rows=(int*)malloc(sizeof(int)*groupInfo[idx].size);
        for(int i=0; i<groupInfo[idx].size; i++) {
            groupInfo[idx].rows[i] = resultList[groupList[idx]+i];
        }
        //cudaMalloc((int**)&groupInfo[idx].label, groupInfo[idx].rank*sizeof(int));
        groupInfo[idx].rank = rowPtr[idx+1] - rowPtr[idx];
        if(groupInfo[idx].rank > 0) {
            groupInfo[idx].label = (int*)malloc(groupInfo[idx].rank * sizeof(int));
            for(int j=0; j<groupInfo[idx].rank; j++) {
                groupInfo[idx].label[j] = colIdx[rowPtr[idx]+j];
            }
        } 
    }

    __syncthreads();

    if(idx%rows_per_thread==0 && idx < groupSize[0]) {
        if((groupSize[0] - idx) >= rows_per_thread)
            group_thread = rows_per_thread;
        else
            group_thread = groupSize[0] - idx;
    }

    // compare between each groups
    if(group_thread > 1) {
        for(int i=0; i<group_thread; i++) {
            tarrow = idx+i; // define the group used as base
            if(tarrow == -1) continue; 
            for(int j=i+1; j<group_thread; j++) {
                refrow = idx+j;   // define the group which should be compared for
                if(refrow == -1) continue; 
                // grouping
                // printf("tarrow.label == NULL? %d, tarrow.rank = %d\n", groupInfo[tarrow].label == NULL, groupInfo[tarrow].rank);
                // printf("refroe.label == NULL? %d, refroe.rank = %d\n", groupInfo[refrow].label == NULL, groupInfo[refrow].rank);
                if(HammingDistance(groupInfo[tarrow], groupInfo[refrow]) < tau) {
                    combineGroup(groupInfo[tarrow], groupInfo[refrow]);
                    // printf("distance = %f\n", HammingDistance(groupInfo[tarrow], groupInfo[refrow]));
                }
            }
        }
    }

    __syncthreads();

    // after group, change the groupList in the thread 0
    if(idx == 0) {
        groupList[0] = 0;
        gap = 0; // initialize the gap to 0
        for(int i=0; i<groupSize[0]; i++) {
            if(groupInfo[i].alive) {
                groupList[gap+1] = groupList[gap] + groupInfo[i].size;
                for(int j=0; j<groupInfo[i].size; j++) {
                    resultList[groupList[gap]+j] = groupInfo[i].rows[j];
                }
                gap ++;
            }
        }
        if(gap < groupSize[0])  {
            change = 1; // if any of the group changed, keep changing
            groupSize[0] = gap;
        }
    }
    // update the csr info in another warp(groupInfo read only)
    if(idx == 32) {
        int irow =0;
        int* tmpRowPtr=(int*)malloc(nnz * sizeof(int));
        int* tmpColIdx=(int*)malloc(nnz * sizeof(int));
        for(int i=0; i<nnz; i++) {
            if(groupInfo[i].alive) {
                tmpRowPtr[idx+1] = tmpRowPtr[idx] + groupInfo[i].rank;
                for(int j=0; j<groupInfo[i].rank; j++) {
                    tmpColIdx[tmpRowPtr[idx]+j] = groupInfo[i].label[j];
                }
            }
        }
        rowPtr = tmpRowPtr;
        colIdx = tmpColIdx;
    }

    __syncthreads();
    // wait for groupList change
    return;
}