#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include "utilities.h"

int main() {
    std::string filename = "../data/weighted/freeFlyingRobot_4.mtx";
    // int label_cols = 64;
    // int block_rows = 64;
    // int group_number = 1;   // should have better performance if same with thread number

    float fine_tau = 0.9;
    
    // method allow inordered input data
    COO coo = readMTXFileWeighted(filename);
    // print_matrix(matrix, block_rows); //print matrix message

    //int labelSize = (matrix.cols-1) / label_cols + 1;

    // init label list for distance calculation
    std::vector<std::vector<int>> label(coo.rows); //,std::vector<int>(labelSize));
    std::multimap<int, int, std::greater<int>> rankMap;

    CSR csr(coo.rows, coo.cols, coo.nnz);
    // csr to coo, build the rankMap at same time
    cooToCsr(coo, csr, rankMap);
    // free the matrix, use csr
    coo.row_message.clear();
    // print_vec(label);
    // csr.print();
    // print_map(rankMap);

    // init coarse graind group vector
    // we have a little larger buffer to save in case there be one group have more elements than others, but not too much..
    std::vector<int> coarse_group;
    // init fine graind group vector
    // give little more space for blocks in each thread which can not be totally filled
    std::vector<std::vector<int>> fine_group;

    // init the group by rank
    /*  reason: many dynamic real-world graphs,
        such as social networks, follow a skewed distribution of vertex
        degrees, where there are a few high-degree vertices and many
        low-degree vertices. */
        
    std::multimap<int, int>::iterator itr = rankMap.begin();
    coarse_group.push_back(itr->second);
    rankMap.erase(itr++); 

    // adding a group recorder? or just record by add -1?
    // if(!coarse_grouping(coarse_group, matrix, rankMap, group_number, coarse_group_rows)) {
    //     std::cout << "Failed in coarse graind grouping... " << std::endl;
    //     return -1;
    // }

    //tmp fine group test use, all as one group
    for (auto itr = rankMap.begin(); itr != rankMap.end(); itr++) { 
        coarse_group.push_back(itr->second);
        rankMap.erase(itr++); 
    }

    fine_grouping(coarse_group, csr, fine_group, fine_tau);

    //print_vec(fine_group);
    CSR new_csr(csr.rows, csr.cols, csr.nnz);
    reordering(csr, new_csr, fine_group);
    //new_csr.print();
    std::cout << "original density" << csr.calculateBlockDensity(64, 64) << std::endl;
    std::cout << "new density" << new_csr.calculateBlockDensity(64, 64) << std::endl;

    return 0;
}