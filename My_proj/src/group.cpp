#include "group.h"

bool cmp(std::pair<int,float>a, std::pair<int,float>b){
    if(a.second!=b.second) return a.second>b.second;
    else return a.first<b.first;
}

float HammingDistance(std::vector<int> &baseColIdx, std::vector<int> &refColIdx) {
    int dist = 0;
    int cols = baseColIdx.size() + refColIdx.size();
    int refIdx = 0;
    int baseIdx = 0;

    if(baseColIdx.size() == 0 && refColIdx.size() == 0) return 0.0;
    if(baseColIdx.size() == 0 || refColIdx.size() == 0) return 1.0; 

    while(baseIdx < baseColIdx.size() && refIdx < refColIdx.size()) {
        if(baseColIdx[baseIdx] == refColIdx[refIdx]) {
            baseIdx++;
            refIdx++;
            cols--;
        }else if(baseColIdx[baseIdx] < refColIdx[refIdx]) {
            dist++;
            baseIdx++;
        }else{ 
            dist++;
            refIdx++;
        }
    }
    // if ref is at end, add all the red as non-equal
    if(baseIdx >= baseColIdx.size()) {
        dist += (refColIdx.size() - refIdx);
    }
    // if base is at end, add all the red as non-equal
    if(refIdx >= refColIdx.size()) {
        dist += (baseColIdx.size() - baseIdx);
    }
    return (float)dist / (float)cols;
}

void dense_priority_ref(std::vector<int> &priority_queue, CSR &csr) {
    std::vector<std::pair<int, float>> priority_map;

    for(int i=0; i<csr.rows; i++) {
        priority_map.push_back(std::make_pair(i, (float) (csr.rowPtr[i+1] - csr.rowPtr[i])/(float)csr.cols));
    }

    std::sort(priority_map.begin(), priority_map.end(), cmp);

    for(int i=0; i<csr.rows; i++) {
        priority_queue.push_back(priority_map[i].first);
        // std::cout << priority_map[i].first << " " << priority_map[i].second << std::endl;
    }
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
int fine_grouping(std::vector<int> &coarse_group, CSR &matrix, 
                        std::vector<std::vector<int>> &fine_group, float tau) {
    int baseRow, targetRow;
    int size = coarse_group.size();
    std::vector<int> baseColIdx;
    std::vector<int> refColIdx;
    std::vector<int> currentGroup;
    int index[coarse_group.size()] = {0};
    int cnt = coarse_group.size();
    int iterCnt = 0;
    while(!coarse_group.empty()) {
        //size = coarse_group.size();
        baseRow = coarse_group[0];
        coarse_group.erase(coarse_group.begin());
        if(index[baseRow] == 1) continue;
        currentGroup.push_back(baseRow);
        index[baseRow] = 1;    // label the row as groupped
        // build the initial row label
        int col = 0;
        baseColIdx.clear();
        for(int i=0; i < matrix.rowPtr[baseRow+1] - matrix.rowPtr[baseRow]; i++) {
            col = matrix.colIdx[matrix.rowPtr[baseRow] + i];
            baseColIdx.push_back(col);
        }

        // iterate all the rows
        for(int idx = 0; idx < size; idx++) {
            if(index[idx] == 1) continue;
            targetRow = idx;

            // build the referring row label
            refColIdx.clear();
            for(int i=0; i < matrix.rowPtr[targetRow+1] - matrix.rowPtr[targetRow]; i++) {
                col = matrix.colIdx[matrix.rowPtr[targetRow] + i];
                refColIdx.push_back(col);
            }

            // std::sort(baseColIdx.begin(), baseColIdx.end());
            // std::sort(refColIdx.begin(), refColIdx.end());

            // calculate distance
            float dist = HammingDistance(baseColIdx, refColIdx);

            // if(dist<tau) add, else ignore
            if(dist < tau) {
                currentGroup.push_back(targetRow);
                index[idx] = 1;
                // // calculate the new baseColIdx
                // for(int i=0; i < matrix.rowPtr[targetRow+1] - matrix.rowPtr[targetRow]; i++) {
                //     col = matrix.colIdx[matrix.rowPtr[targetRow] + i];
                //     baseColIdx.erase(std::unique(baseColIdx.begin(), baseColIdx.end()), baseColIdx.end());
                // }
            }
        }
        // build the new group for rows who still inside
        for(int i = 0; i < currentGroup.size(); i++)
            fine_group[currentGroup[0]].push_back(currentGroup[i]);
        currentGroup.clear();
        iterCnt ++;
    }
    return iterCnt;
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