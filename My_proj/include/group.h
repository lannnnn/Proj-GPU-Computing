#pragma once

#include <string>
#include <vector>
#include <map>
#include "matrices.h"
#include "row.h"

double HammingDistance(std::vector<int> &baseColIdx, std::vector<int> &refColIdx) {
    int dist = 0;
    int cols = 0;

    for (int i = 0; i < baseColIdx.size(); i++) {
        dist += (baseColIdx[i] + refColIdx[i] == 1? 1:0);
        cols += (baseColIdx[i] + refColIdx[i] > 0? 1:0);
    }
    return (double)dist / (double)cols;
}

/**** THIS METHOD BENEFITS MORE FROM LARGE BLOCK COLOMN SIZE SINCE IT COMPRESS MORE THE LABEL SIZE ****/
// coarse group according to the label message, the result will be saved in coarse_group
// if something goes wrong, return -1, else 0
// algorithm:
//      first iteration: calculate the distance of labels between ungrouped rows with the one in the group
//                       if distance <= (some given value), group the rows
//      second iteration: if there still be some rows not grouped
//                           1) all groups have similar size - group it into the group with smallest distance
//                           2) some of the groups was much larger - 
//                                  select the one with largest rank and put into the smallest group, calculate the dis as step 1)
int coarse_grouping(std::vector<std::vector<int>> coarse_group, CSR matrix, 
                        std::multimap<int, int, std::greater<int>> rankMap, int group_number, int coarse_group_rows);

// fine graind grouping, consider only the rows inside the same coarse group
// algorithm:
//      calculate the distance using HammingDistance? JaccardDistance? 
//      if distance < (some given value), group in a block
void fine_grouping(std::vector<std::vector<int>> &coarse_group, CSR &matrix, 
                        std::vector<std::vector<int>> &fine_group, int group_number, float tau, int block_rows) {
    int baseRow, targetRow;
    std::vector<int> baseColIdx(matrix.cols);
    std::vector<int> refColIdx(matrix.cols);
    std::vector<int> currentGroup;
    for(int i=0; i<group_number; i++){
        while(!coarse_group[i].empty()) {
            int size = coarse_group[i].size();
            // if we have little rows left, group together anyway?
            if(size < 2) {
                for(int cnt = 0; cnt < size; cnt++) 
                    currentGroup.push_back(coarse_group[i][cnt]);
                fine_group.push_back(currentGroup);
                return;
            }
            baseRow = coarse_group[i][0];
            coarse_group[i].erase(coarse_group[i].begin());
            currentGroup.push_back(baseRow);
            // build the initial row label
            int col = 0;
            std::fill(baseColIdx.begin(), baseColIdx.end(), 0);
            for(int i=0; i < matrix.rowPtr[baseRow+1] - matrix.rowPtr[baseRow]; i++) {
                col = matrix.colIdx[matrix.rowPtr[baseRow] + i];
                baseColIdx[col] = 1;
            }

            // iterate all the rows
            for(int cnt = 0; cnt < size; cnt++) {
                targetRow = coarse_group[i][0];
                coarse_group[i].erase(coarse_group[i].begin());
                
                // build the referring row label
                std::fill(refColIdx.begin(), refColIdx.end(), 0);
                for(int i=0; i < matrix.rowPtr[targetRow+1] - matrix.rowPtr[targetRow]; i++) {
                    col = matrix.colIdx[matrix.rowPtr[targetRow] + i];
                    refColIdx[col] = 1;
                }

                // calculate distance
                double dist = HammingDistance(baseColIdx, refColIdx);
                // if(dist<tau) add, else ignore
                // std::cout << "dist=" << dist << "tau=" << tau<< std::endl;
                if(dist <= tau) {
                    currentGroup.push_back(targetRow);
                    // calculate the new baseColIdx
                    for(int i=0; i < matrix.rowPtr[targetRow+1] - matrix.rowPtr[targetRow]; i++) {
                        col = matrix.colIdx[matrix.rowPtr[targetRow] + i];
                        baseColIdx[col] = 1;
                    }
                } else {
                    coarse_group[i].push_back(targetRow);
                }
            }
            // build the new group for rows who still inside
            fine_group.push_back(currentGroup);
            currentGroup = {};
        }
    }
}