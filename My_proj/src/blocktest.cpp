#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include "utilities.h"

int main() {
    // std::string filename = "/home/shuxin.zheng/Proj-GPU-Computing/My_proj/data/weighted/TEST_matrix_weighted.el";
    std::string filename = "/home/shuxin.zheng/Proj-GPU-Computing/My_proj/data/unweighted/seventh_graders.el";
    // std::string filename = "/home/shuxin.zheng/Proj-GPU-Computing/My_proj/data/unweighted/0_mycielskian13.el";

    // int label_cols = 64;
    // int block_rows = 64;
    // int group_number = 1;   // should have better performance if same with thread number
    int block_cols = 3;

    float fine_tau = 0.6;
    // printf("ready to read \n");
    // method allow inordered input data
    // COO coo = readMTXFileWeighted(filename);
    // COO coo = readELFileWeighted(filename);
    COO coo = readELFileUnweighted(filename);
    // print_matrix(coo, 1); //print matrix message

    //int labelSize = (matrix.cols-1) / label_cols + 1;

    // init label list for distance calculation
    std::vector<std::vector<int>> label(coo.rows); //,std::vector<int>(labelSize));
    // std::multimap<int, int, std::greater<int>> rankMap;

    CSR csr(coo.rows, coo.cols, coo.nnz);
    // csr to coo, build the rankMap at same time
    cooToCsr(coo, csr);
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

    for (int i=0; i<csr.rows; i++) { 
        coarse_group.push_back(i);
    }

    // print_pointer(&coarse_group[0], coarse_group.size());

    fine_grouping(coarse_group, csr, fine_group, fine_tau);

    print_vec(fine_group);
    CSR new_csr(csr.rows, csr.cols, csr.nnz);
    reordering(csr, new_csr, fine_group);
    //new_csr.print();
    std::cout << "using matrix file: " << filename << std::endl;
    std::cout << "matrix info: nrows=" << csr.rows << ", ncols=" << csr.cols << ", nnz=" << csr.nnz << std::endl;
    std::cout << "checking for using block size: (" << block_cols << "," << block_cols << ")" << std::endl;
    std::cout << "original density: " << csr.calculateBlockDensity(block_cols, block_cols) << std::endl;
    std::cout << "new density: " << new_csr.calculateBlockDensity(block_cols, block_cols) << std::endl;
    std::cout << "GPU calculation time: " << std::endl;

    return 0;
}