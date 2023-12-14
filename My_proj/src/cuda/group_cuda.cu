#include <stdio.h>
#include "cuda_impl.cuh"

__global__ void test(int* groupList, int* resultList) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    resultList[idx] = groupList[idx]*2;
}

__device__ float HammingDistance(int* rowPtr, int* colIdx, int baseline, int refline) {
    int baseRank = rowPtr[baseline+1] - rowPtr[baseline];
    int refRank = rowPtr[refline+1] - rowPtr[refline];
    // printf("baseline = %d, refline = %d, baserank = %d - %d, refrank = %d\n", baseline, refline, rowPtr[baseline+1], rowPtr[baseline], refRank);
    // if both have no nz value, same
    if(baseRank == 0 && refRank == 0) return 0;
    // if the ref row is empty, return 1;
    if(baseRank == 0 || refRank == 0) return 1;

    int dist = 0;
    int cols = baseRank + refRank;
    int i = 0;
    int j = 0;

    while(i<baseRank && j<refRank) {
        if(colIdx[rowPtr[baseline]+i] == colIdx[rowPtr[refline]+j]) {
            cols --;
            i++; j++;
        } else if(colIdx[rowPtr[baseline]+i] < colIdx[rowPtr[refline]+j]) {
            dist++; i++;
        } else {
            dist++; j++;
        }
    }
    if(i >= baseRank && j<refRank) dist += (refRank - j);
    if(j >= refRank && i<baseRank) dist += (baseRank - i);

    //printf("distance = %f\n", (double) dist / (double) cols);

    return (float) dist / (float) cols;
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

__device__ void buildGroupInfo(int* rowPtr, int* colIdx, int* groupList, int* resultList, GroupInfo &groupInfo, int idx, int block_cols) {
        int realRank = 0;
        groupInfo.alive = 1;
        //group row info -> combine the last resultList according to groupInfo
        groupInfo.size = groupList[idx+1]-groupList[idx];
        groupInfo.rows=(int*)malloc(sizeof(int)*groupInfo.size);
        for(int i=0; i<groupInfo.size; i++) {
            groupInfo.rows[i] = resultList[groupList[idx]+i];
        }
        //cudaMalloc((int**)&groupInfo[idx].label, groupInfo[idx].rank*sizeof(int));
        groupInfo.rank = rowPtr[idx+1] - rowPtr[idx];
        if(groupInfo.rank > 0) {
            groupInfo.label = (int*)malloc(groupInfo.rank * sizeof(int));
            for(int j=0; j<groupInfo.rank; j++) {
                groupInfo.label[j] = colIdx[rowPtr[idx]+j];
                // if(realRank ==0 || (colIdx[rowPtr[idx]+j] / block_cols) != groupInfo.label[realRank]-1) {
                //     groupInfo.label[realRank] = colIdx[rowPtr[idx]+j]/ block_cols;
                //     realRank ++;
                // }
            }
        } 
        // groupInfo.rank = realRank;
}

__device__ void findRef(int* refRow, int* rowPtr, int* colIdx, int* groupList, float tau, int groupSize) {
    int idx = 0;
    int cnt = 0;
    //clear the ref list
    for(int i=0; i<ref_size; i++) refRow[i] = -1;
    
    for(int i=0; i<groupSize; i++) {
        // break if the list is full
        if(idx == ref_size) break;
        // if already groupped, skip
        if(groupList[i] != -1) continue;
        // if is the first, just add
        if(idx == 0) {
            refRow[idx] = i;
            groupList[i] = i;
            idx++;
            continue;
        }

        for(cnt = 0; cnt<idx; ++cnt) {
            if(HammingDistance(rowPtr, colIdx, refRow[cnt], i) < tau) break;
        }

        if(cnt >= idx) {
            refRow[idx] = i;
            groupList[i] = i;
            idx++;
        }
    }
}

__device__ void findRefPriority(int* refRow, int* ref_queue, int* groupList, int groupSize) {
    int idx = 0;
    //clear the ref list
    for(int i=0; i<ref_size; i++) refRow[i] = -1;
    
    for(int i=0; i<groupSize; i++) {
        // break if the list is full
        if(idx == ref_size) break;
        // if already groupped, skip
        if(groupList[ref_queue[i]] != -1) continue;
        // if is the first, just add
        refRow[idx] = ref_queue[i];
        groupList[ref_queue[i]] = ref_queue[i];
        idx++;
    }
}

__device__ void findRefPriorityDist(int* refRow, int* ref_queue, int* rowPtr, int* colIdx, int* groupList, int groupSize) {
    int idx = 0;
    int cnt = 0;
    //clear the ref list
    for(int i=0; i<ref_size; i++) refRow[i] = -1;
    
    for(int i=0; i<groupSize; i++) {
        // break if the list is full
        if(idx == ref_size) break;
        // if already groupped, skip
        if(groupList[ref_queue[i]] != -1) continue;
        // if is the first, just add
        if(idx == 0) {
            refRow[idx] = ref_queue[i];
            groupList[ref_queue[i]] = ref_queue[i];
            idx++;
            continue;
        }

        for(cnt = 0; cnt<idx; ++cnt) {
            if(HammingDistance(rowPtr, colIdx, refRow[cnt], ref_queue[i]) == 1) break;
        }

        if(cnt >= idx) {
            refRow[idx] = ref_queue[i];
            groupList[ref_queue[i]] = ref_queue[i];
            idx++;
        }
    }
}

__global__ void gpu_ref_grouping(int* rowPtr, int* colIdx, float* tau, int* groupList, int* groupSize, int* refRow) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tarrow;
    int gap=0;
    int goalVal = blockDim.x * gridDim.x;
    int loopIdx = 0;
    auto grid = cooperative_groups::this_grid();
    float minDist;
    float dist;

    // build the groupInfo
    // while((idx+goalVal*loopIdx) < groupSize[0]) {
    //     buildGroupInfo(rowPtr, colIdx, groupList, resultList, groupInfo[idx+goalVal*loopIdx], idx+goalVal*loopIdx, 0);
    //     groupList[idx+goalVal*loopIdx] = -1; // clear the group list
    //     loopIdx++;
    // }

    // while still have rows not groupped
    do { 
        // find the ref rows

        loopIdx = 0;
        if(idx == 0) {
            findRef(refRow, rowPtr, colIdx, groupList, tau[0], groupSize[0]);
        }
        
	    grid.sync();
        // printf("threadIdx = %d,  %d %d %d %d\n", idx, refRow[0], refRow[1], refRow[2], refRow[3]);

        while((idx+goalVal*loopIdx) < groupSize[0]) {
            minDist = tau[0];
            tarrow = -1;
            // compare between each groups
            if(groupList[idx+goalVal*loopIdx]==-1) {
                for(int i=0; i<ref_size; i++) {
                    if(refRow[i] == -1) continue;
                    dist = HammingDistance(rowPtr, colIdx, refRow[i], idx+goalVal*loopIdx);
                    //printf("threadIdx = %d, refrow = %d, checkrow = %d, dist = %f\n", idx, refRow[i], idx+goalVal*loopIdx, dist);
                    if(dist < minDist) {
                        tarrow = refRow[i];
                        minDist = dist;
                    }
                }
            }

            if(tarrow!=-1) {
                // combineGroup(groupInfo[tarrow], groupInfo[idx]);
                groupList[idx+goalVal*loopIdx] = tarrow;
            }
            loopIdx++;
            // printf("threadIdx = %d,  %d %d %d %d\n", idx, refRow[0], refRow[1], refRow[2], refRow[3]);
        } 

        grid.sync();

    } while(refRow[ref_size-1]!=-1);

    return;
}


__global__ void gpu_ref_grouping_O1(int* rowPtr, int* colIdx, float* tau, int* groupList, int* groupSize, int* refRow, int* rows_per_thread) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tarrow;
    int gap=0;
    int goalVal = (groupSize[0] + rows_per_thread[0] - 1) / rows_per_thread[0];
    int loopIdx = 0;
    auto grid = cooperative_groups::this_grid();
    float minDist;
    float dist;

    // while still have rows not groupped
    do { 
        // find the ref rows

        loopIdx = 0;
        if(idx == 0) {
            findRef(refRow, rowPtr, colIdx, groupList, tau[0], groupSize[0]);
        }
        
	    grid.sync();
        // printf("threadIdx = %d, rowStart = %d, rows_per_thread = %d\n", idx, rowStart, rows_per_thread[0]);

        if(idx*row_size < groupSize[0])
        while((idx + loopIdx*goalVal) < groupSize[0] && loopIdx < rows_per_thread[0]) {
            minDist = tau[0];
            tarrow = -1;
            // compare between each groups
            if(groupList[idx + loopIdx*goalVal]==-1) {
                for(int i=0; i<ref_size; i++) {
                    if(refRow[i] == -1) continue;
                    dist = HammingDistance(rowPtr, colIdx, refRow[i], idx + loopIdx*goalVal);
                    // printf("threadIdx = %d, refrow = %d, checkrow = %d, dist = %f\n", idx, refRow[i], idx+goalVal*loopIdx, dist);
                    if(dist < minDist) {
                        tarrow = refRow[i];
                        minDist = dist;
                    }
                }
            }

            if(tarrow!=-1) {
                // combineGroup(groupInfo[tarrow], groupInfo[idx]);
                groupList[idx + loopIdx*goalVal] = tarrow;
            }
            loopIdx++;
            // printf("threadIdx = %d,  %d %d %d %d\n", idx, refRow[0], refRow[1], refRow[2], refRow[3]);
        } 

        grid.sync();

    } while(refRow[ref_size-1]!=-1);

    return;
}

// self define priority for referring rows
__global__ void gpu_ref_grouping_O2(int* rowPtr, int* colIdx, float* tau, int* groupList, int* groupSize, int* refRow, int* rows_per_thread, int* ref_queue, int* iterCnt) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tarrow;
    int gap=0;
    int goalVal = (groupSize[0] + rows_per_thread[0] - 1) / rows_per_thread[0];
    int loopIdx = 0;
    auto grid = cooperative_groups::this_grid();
    float minDist;
    float dist;
    iterCnt[0] = 0;

    // while still have rows not groupped
    do { 
        // find the ref rows

        loopIdx = 0;
        if(idx == 0) {
            iterCnt[0] ++;
            findRefPriority(refRow, ref_queue, groupList, groupSize[0]);
            //findRefPriorityDist(refRow, ref_queue, rowPtr, colIdx, groupList, groupSize[0]);
        }
        
	    grid.sync();
        // printf("threadIdx = %d, rowStart = %d, rows_per_thread = %d\n", idx, rowStart, rows_per_thread[0]);

        if(idx*row_size < groupSize[0])
        while((idx + loopIdx*goalVal) < groupSize[0] && loopIdx < rows_per_thread[0]) {
            minDist = tau[0];
            tarrow = -1;
            // compare between each groups
            if(groupList[idx + loopIdx*goalVal]==-1) {
                for(int i=0; i<ref_size; i++) {
                    if(refRow[i] == -1) continue;
                    dist = HammingDistance(rowPtr, colIdx, refRow[i], idx + loopIdx*goalVal);
                    // printf("threadIdx = %d, refrow = %d, checkrow = %d, dist = %f\n", idx, refRow[i], idx+goalVal*loopIdx, dist);
                    if(dist < minDist) {
                        tarrow = refRow[i];
                        minDist = dist;
                    }
                }
            }

            if(tarrow!=-1) {
                // combineGroup(groupInfo[tarrow], groupInfo[idx]);
                groupList[idx + loopIdx*goalVal] = tarrow;
            }
            loopIdx++;
            // printf("threadIdx = %d,  %d %d %d %d\n", idx, refRow[0], refRow[1], refRow[2], refRow[3]);
        } 

        grid.sync();

    } while(refRow[ref_size-1]!=-1);

    return;
}