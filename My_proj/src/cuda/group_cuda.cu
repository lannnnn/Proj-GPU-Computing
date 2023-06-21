#include <stdio.h>
#include "cuda_impl.h"

__device__ double HammingDistance(GroupInfo &tarrow, GroupInfo&refrow) {
    // if the ref row is empty, return 1;
    if(refrow.rank == 0) return 1;

    int dist = 0;
    int cols = tarrow.rank + refrow.rank;
    int i, j;

    for(i=0, j=0; i<tarrow.rank, j<refrow.rank;) {
        if(tarrow.label[i] == refrow.label[j]) {
            cols --;
        } else if(tarrow.label[i] < refrow.label[j]) {
            dist ++;
            i++;
        } else {
            dist ++;
            j++;
        }
    }
    if(i >= tarrow.rank) dist += (refrow.rank - j);
    if(j >= refrow.rank) dist += (tarrow.rank - i);

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

__global__ void gpu_grouping(int* rowPtr, int* colIdx, float tau, int* groupList, GroupInfo* groupInfo, int groupSize, int nnz) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int group_thread = 0;
    // take (rows_per_thread) rows for one thread
    int baserow = idx * rows_per_thread;
    int tarrow, refrow;
    int gap=0;
    if(baserow < groupSize) {
        if(rows_per_thread < (groupSize-baserow)) {
            group_thread = rows_per_thread;
        } else {
            group_thread = groupSize - baserow;
        }
    }

    // build the groupInfo
    for(int i=0; i < group_thread; i++) {
        groupInfo[baserow+i].alive = 1;
        groupInfo[baserow+i].size = 1;
        *groupInfo[baserow+i].rows = baserow+i;
        groupInfo[baserow+i].rank = rowPtr[baserow+i+2] - rowPtr[baserow+i+1];
        groupInfo[baserow+i].label = new int[groupInfo[baserow+i].rank];
        for(int j=0; j<groupInfo[baserow+i].rank; j++) {
            groupInfo[baserow+i].label[j] = colIdx[rowPtr[baserow+i+1]+j];
        }
    }

    // need a global value to check if at leat one group is changed, otherwise stop
    while(change) {
        // check the number of groups in duty
        if(groupSize < idx * rows_per_thread) break; // if no group left, finish this thread
        if((groupSize-baserow) > rows_per_thread) {
            group_thread = rows_per_thread;
        } else {
            group_thread = groupSize - baserow;
        }
        // compare between each groups
        for(int i=0; i<group_thread; i++) {
            tarrow = groupList[baserow+i]; // define the group used as base
            for(int j=i+1; j<group_thread; j++) {
                refrow = groupList[baserow+j];   // define the group which should be compared for
                // grouping
                if(HammingDistance(groupInfo[tarrow], groupInfo[refrow])) {
                    combineGroup(groupInfo[tarrow], groupInfo[refrow]);
                }
            }
        }

        // after group, change the groupList in the thread 0
        if(idx == 0) {
            gap = 0; // initialize the gap to 0
            for(int i=0; i<groupSize; i++) {
                if(groupInfo[groupList[i]].size == 0) {
                    groupList[i] = -1;
                }
            }
            for(int i=0; i<groupSize; i++) {
                if(groupList[i] == -1) {
                    gap++;          // ignore one group
                    groupSize--;    // change groupSize
                }
                groupList[i] = groupList[i+gap];
            }
        }
        // wait for groupList change
        __syncthreads();
    }
}