#include "group.h"

double HammingDistance(std::vector<int> &baseColIdx, std::vector<int> &refColIdx) {
    int dist = 0;
    int cols = 0;
    int refIdx = 0;

    if(baseColIdx.size() == 0 && refColIdx.size() == 0) return 0;
    if(baseColIdx.size() == 0 || refColIdx.size() == 0) return 1; 

    for (int i = 0; i < baseColIdx.size();) {
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
        if(refIdx >= refColIdx.size()) {
            dist += baseColIdx.size() - i;
            cols += baseColIdx.size() - i;
            break;
        }
    }
    // if base is at end, add all the red as non-equal
    if(refIdx < refColIdx.size()) {
        dist += refColIdx.size() - refIdx;
        cols += refColIdx.size() - refIdx;
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
void fine_grouping(std::vector<int> &coarse_group, CSR &matrix, 
                        std::vector<std::vector<int>> &fine_group, float tau) {
    int baseRow, targetRow;
    int size;
    std::vector<int> baseColIdx;
    std::vector<int> refColIdx;
    std::vector<int> currentGroup;
    while(!coarse_group.empty()) {
        size = coarse_group.size();
        // if we have little rows left, group together anyway?
        if(size < 2) {
            for(int cnt = 0; cnt < size; cnt++) 
                currentGroup.push_back(coarse_group[cnt]);
            fine_group.push_back(currentGroup);
            return;
        }
        baseRow = coarse_group[0];
        coarse_group.erase(coarse_group.begin());
        currentGroup.push_back(baseRow);
        // build the initial row label
        int col = 0;
        baseColIdx.clear();
        for(int i=0; i < matrix.rowPtr[baseRow+1] - matrix.rowPtr[baseRow]; i++) {
            col = matrix.colIdx[matrix.rowPtr[baseRow] + i];
            baseColIdx.push_back(col);
        }

        // iterate all the rows
        for(int cnt = 0; cnt < size-1; cnt++) {
            targetRow = coarse_group[0];
            coarse_group.erase(coarse_group.begin());

            // build the referring row label
            refColIdx.clear();
            for(int i=0; i < matrix.rowPtr[targetRow+1] - matrix.rowPtr[targetRow]; i++) {
                col = matrix.colIdx[matrix.rowPtr[targetRow] + i];
                refColIdx.push_back(col);
            }

            std::sort(baseColIdx.begin(), baseColIdx.end());
            std::sort(refColIdx.begin(), refColIdx.end());

            // calculate distance
            double dist = HammingDistance(baseColIdx, refColIdx);

            // if(dist<tau) add, else ignore
            if(dist <= tau) {
                currentGroup.push_back(targetRow);
                // calculate the new baseColIdx
                for(int i=0; i < matrix.rowPtr[targetRow+1] - matrix.rowPtr[targetRow]; i++) {
                    col = matrix.colIdx[matrix.rowPtr[targetRow] + i];
                    baseColIdx.erase(std::unique(baseColIdx.begin(), baseColIdx.end()), baseColIdx.end());
                }
            } else {
            coarse_group.push_back(targetRow);
            }
        }
        // build the new group for rows who still inside
        fine_group.push_back(currentGroup);
        currentGroup = {};
    }
}

void reordering(CSR &omatrix, CSR &nmatrix, std::vector<std::vector<int>> &fine_group) {
    int targetRow = 0;
    int rank = 0;
    int currentRow = 0;
    nmatrix.rowPtr[0] = 0;
    nmatrix.colIdx.clear();
    for(int i=0; i<fine_group.size(); i++) {
        for(int j=0; j<fine_group[i].size(); j++) {
            targetRow = fine_group[i][j];
            rank = omatrix.rowPtr[targetRow+1] - omatrix.rowPtr[targetRow];
            nmatrix.rowPtr[currentRow+1] = nmatrix.rowPtr[currentRow] + rank;
            for(int k=0; k < rank; k++) {
                nmatrix.colIdx.push_back(omatrix.colIdx[omatrix.rowPtr[targetRow]+k]); 
                // nmatrix.values.push_back(omatrix.values[omatrix.rowPtr[targetRow]+k]); //just ignore value for now
            }
            currentRow++;
        }
    }
}