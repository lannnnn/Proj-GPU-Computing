#include <stdio.h>

__device__ double HammingDistance(int* rowPtr, int* colIdx, int baserow, int refrow, int* groupPtr) {
    // if the ref row is empty, return 1;
    if(groupPtr[refrow+2] - groupPtr[refrow+1] == 0) return 1;

    int baseRank = rowPtr[baserow+2] - rowPtr[baserow+1];
    int refRank = rowPtr[refrow+2] - rowPtr[refrow+1];

    int dist = 0;
    int cols = baseRank + refRank;
    int i, j;

    for(i=0, j=0; i<baseRank, j<refRank;) {
        if(colIdx[rowPtr[baserow+1] + i] == colIdx[rowPtr[refrow+1] + j]) {
            cols --;
        } else if(colIdx[rowPtr[baserow+1] + i] < colIdx[rowPtr[refrow+1] + j]) {
            dist ++;
            i++;
        } else {
            dist ++;
            j++;
        }
    }
    if(i >= baseRank) dist += (refRank - j);
    if(j >= refRank) dist += (baseRank - i);

    return (double) dist / (double) cols;
}

__device__ void combineRows_i_j(int baserow, int* rowIdx, int* groupPtr, int i_index, int j_index) {
    int rank = groupPtr[baserow+j_index+2] - groupPtr[baserow+j_index+1];
    // copy the target rowIdx
    for(int k=0; k<rank; k++) {
        tmpIdx[k] = rowIdx[groupPtr[baserow+j_index+1]+k];
    }
    // change the row index
    for(int k=j_index+1; k>i_index+1; k--) {
        groupPtr[baserow + k]+= rank;
    }
    // move all the other row index back
    for(int k=groupPtr[baserow+j_index+1]; k>groupPtr[baserow+i_index+2]; k--) {
        rowIdx[k + rank] = rowIdx[k];
    }
    // add the index into blank
    for(int k=0; k<rank; k++) {
        rowIdx[groupPtr[baserow+i_index+2] + k] = tmpIdx[k];
    }
}

__device__ void combineLable(int* rowPtr, int* colIdx, int baserow, int refrow) {
    int baseRank = rowPtr[baserow+2] - rowPtr[baserow+1];
    int refRank = rowPtr[refrow+2] - rowPtr[refrow+1];

    int* tmpLabel(baseRank + refRank);
    int rankIdx = 0
    int i, j;

    for(i=0, j=0; i<baseRank, j<refRank; rankIdx++) {
        if(colIdx[rowPtr[baserow+1] + i] <= colIdx[rowPtr[refrow+1] + j]) {
            tmpLabel[rankIdx] = colIdx[rowPtr[baserow+1] + i];
        } else {
            tmpLabel[rankIdx] = colIdx[rowPtr[refrow+1] + j]
        }
    }

    if(i >= baseRank) {
        for(int k = j; k<refRank; k++, rankIdx++) {
            tmpLabel[rankIdx] = colIdx[rowPtr[refrow+1] + k];
        }
    }
    if(j >= refRank) dist += (baseRank - i) {
        for(int k = i; k<baseRank; k++, rankIdx++) {
            tmpLabel[rankIdx] = colIdx
        }
    }

    // combine the label back to the list
    int rankdiff = rankIdx - baseRank;
    for(int k=rowPtr[refrow+2]-1; k>=rowPtr[baserow+1]+rankIdx; k--) {
        colIdx[k] = colIdx[k-rankdiff];
    }

    for(int k=0; k<rankdiff; k--) {
        colIdx[rowPtr[baserow+1]+k] = tmpLabel[k];
    }

    for(int k=baserow; k<refrow; k++) {
        rowPtr[baserow+2] = rowPtr[baserow+2] + rankdiff;
    }
}

__device__ void fine_grouping(int* rowPtr, int* colIdx, float tau, int* rowIdx, int* groupPtr, 
                                    int group_size, int baserow, int nrows, int nnz) {
    int* tmpIdx(nrows);
    for(int i=0; i<group_size; i++) {
        for(int j = i+1; j<group_size; j++) {
            // if the two rows/group is close enough, group together
            if(HammingDistance(rowPtr, colIdx, baserow+i, baserow+j, groupPtr) < tau) {
                // combine the rows into one group
                combineRows(baserow, rowIdx, groupPtr, i, j);
                // build the new label
                combineLable(rowPtr, colIdx, baserow+i, baserow+j);
            }
        }
    }
}

__global__ void gpu_grouping(int* rowPtr, int* colIdx, float tau, int* rowIdx, int* groupPtr, int nrows, int nnz) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int group_size = 0;
    // take four rows for one thread
    int baserow = idx * rows_per_thread;
    if(baserow < nrows) {
        if((nrows - baserow) > rows_per_thread) {
            group_size = rows_per_thread;
        } else {
            group_size = nrows - baserow;
        }
    }

    if(idx == 0) {
        for(int i=0; i<nrows; i++) {
            rowIdx[i] = i;
            groupPtr[i] = i;
        }
        groupPtr[i+1] = i+1;
    }

    // synchronize();

    // calculate the distance
    fine_grouping(rowPtr, colIdx, tau, rowIdx, groupPtr, group_size, baserow, nrows, nnz);
}