#pragma once

#include <string>
#include <vector>
#include <map>
#include "matrices.h"

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
int fine_grouping(std::vector<std::vector<int>> coarse_group, CSR matrix, 
                        std::vector<std::vector<int>> fine_group, int group_number, int coarse_group_rows) {
    int groupCnt = 0;
    int baseRow, targetRow;
    for(int i=0; i<group_number; i++){
        fine_group[groupCnt][0] = coarse_group[i][0];
        while(!coarse_group[i].empty()) {
            // calculate distance
            // if(dist<tau) add, else ignore
        }
    }

}