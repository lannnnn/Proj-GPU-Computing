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

// __global__ void gpu_grouping(int* rowPtr, int* colIdx, float tau, int* groupList, GroupInfo* groupInfo, 
//                             int* resultList, int* groupSize, int goalVal, int block_cols) {
//     int idx = blockDim.x * blockIdx.x + threadIdx.x;
//     int group_thread = 0;
//     // take (rows_per_thread) rows for one thread
//     int tarrow, refrow;
//     int gap=0;
//     cudaError_t error; 
//     int mutaxVal = goalVal;

//     int groups = rows_per_thread;
//     int change = 1;

//     // build the groupInfo
//     if(idx < groupSize[0]) {
//         buildGroupInfo(rowPtr, colIdx, groupList, resultList, groupInfo[idx], idx, block_cols);
//     }

//     atomicAdd((int*) &g_mutex, 1);
//     while (g_mutex != mutaxVal) {}
//     mutaxVal += goalVal;

//     // for(int n=0; n<3; n++) {
//     while(change) { // loop till only one thread have group_thread
//         if(idx%groups==0 && idx < groupSize[0]) {
//             if((groupSize[0] - idx) >= groups)
//                 group_thread = groups;
//             else
//                 group_thread = groupSize[0] - idx;
//         }

//         // compare between each groups
//         if(group_thread > 1) {
//             for(int i=0; i<group_thread; i++) {
//                 tarrow = idx+i; // define the group used as base
//                 if(groupInfo[tarrow].alive==0) continue; 
//                 for(int j=i+1; j<group_thread; j++) {
//                     refrow = idx+j;   // define the group which should be compared for
//                     if(groupInfo[refrow].alive==0) continue; 
//                     if(HammingDistance(groupInfo[tarrow], groupInfo[refrow]) < tau) {
//                         combineGroup(groupInfo[tarrow], groupInfo[refrow]);
//                     }
//                 }
//             }
//         }
        
//         if(groups >= groupSize[0]) change = 0;
//         groups = groups* 4;
//         // if(groups >= groupSize[0]) change = 0;
        
//         atomicAdd((int*) &g_mutex, 1);
//         while (g_mutex != mutaxVal) {}
//         mutaxVal +=goalVal;
//         // printf("threadIdx = %d, groups = %d, change = %d\n", idx, groups, change);
//     }

//     // after group, change the groupList in the thread 0
//     if(idx == 0) {
//         groupList[0] = 0;
//         gap = 0; // initialize the gap to 0
//         for(int i=0; i<groupSize[0]; i++) {
//             if(groupInfo[i].alive == 1) {
//                 groupList[gap+1] = groupList[gap] + groupInfo[i].size;
//                 // printf("%d ", groupList[gap+1]);
//                 for(int j=0; j<groupInfo[i].size; j++) {
//                     resultList[groupList[gap]+j] = groupInfo[i].rows[j];
//                 }
//                 gap ++;
//             }
//         }
//     }

//     return;
// }

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