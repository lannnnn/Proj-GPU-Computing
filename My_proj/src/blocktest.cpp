#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <ctime>
#include "utilities.h"

clock_t start,end;

int main(int argc, char* argv[]) {
    // std::string filename = "/home/shuxin.zheng/Proj-GPU-Computing/My_proj/data/weighted/TEST_matrix_weighted.el";
    std::string filename = "/home/shuxin.zheng/Proj-GPU-Computing/My_proj/data/unweighted/seventh_graders.el";
    // std::string filename = "/home/shuxin.zheng/Proj-GPU-Computing/My_proj/data/unweighted/0_mycielskian13.el";

    // int label_cols = 64;
    // int block_rows = 64;
    // int group_number = 1;   // should have better performance if same with thread number
    int block_cols = 8;

    float fine_tau = 0.8;
    int print = 0;
    int mtx = 0, el = 0;
    int list = 0;
    float tau = 0.9;
    std::string ofilename = filename + ".reorder";

    if(argc >= 2) {
        readConfig(argc, argv, &filename, &block_cols, &tau, &print, &mtx, &el, &list, &ofilename);
    }

    std::cout << "using matrix file: " << filename << std::endl;
    std::cout << "using blocksize: " << block_cols << std::endl;
    std::cout << "using tau: " << tau << std::endl;

    COO coo ;

    if(mtx==1)  {
        coo = readMTXFileUnweighted(filename);
        std::cout << "Using MTX format" << std::endl;
    } else {
        coo = readELFileUnweighted(filename);
        std::cout << "Using EL format" << std::endl;
    }

    // init label list for distance calculation
    std::vector<std::vector<int>> label(coo.rows); //,std::vector<int>(labelSize));
    // std::multimap<int, int, std::greater<int>> rankMap;

    CSR csr(coo.rows, coo.cols, coo.nnz);
    // csr to coo, build the rankMap at same time
    cooToCsr(coo, csr);
    // print_matrix(coo, 1);
    // free the matrix, use csr
    coo.row_message.clear();

    // init coarse graind group vector
    // we have a little larger buffer to save in case there be one group have more elements than others, but not too much..
    std::vector<int> coarse_group;
    // init fine graind group vector
    // give little more space for blocks in each thread which can not be totally filled
    std::vector<std::vector<int>> fine_group;

    for (int i=0; i<csr.rows; i++) { 
        coarse_group.push_back(i);
    }

    // build priority queue
    std::vector<int> priority_queue;
    dense_priority_ref(priority_queue, csr);

    // for(int i=0; i<csr.rows; i++) {
    //     std::cout << priority_queue[i] << std::endl;
    // }

    start=clock();
    //fine_grouping(coarse_group, csr, fine_group, tau);
    fine_grouping(priority_queue, csr, fine_group, tau);
    end=clock();

    double endtime=(double)(end-start)/CLOCKS_PER_SEC;

    if(print) {
        std::cout << "Reordered row rank:" << std::endl;
        print_vec(fine_group);  
    }

    std::vector<int> res_vec(csr.rows);

    printRes(fine_group, res_vec, ofilename);

    CSR new_csr(csr.rows, csr.cols, csr.nnz);
    reordering(csr, new_csr, fine_group);

    std::cout << "matrix info: nrows=" << csr.rows << ", ncols=" << csr.cols << ", nnz=" << csr.nnz << std::endl;
    std::cout << "checking for using block size: (" << block_cols << "," << block_cols << ")" << std::endl;
    std::cout << "original density: " << csr.calculateBlockDensity(block_cols, block_cols) << std::endl;
    std::cout << "original store density: " << csr.calculateStoreSize(block_cols, block_cols)/(float)csr.rows / (float)csr.cols << std::endl;
    std::cout << "group number: " << count_group(fine_group) << std::endl;
    std::cout << "new density: " << new_csr.calculateBlockDensity(block_cols, block_cols) << std::endl;
    float store_density = new_csr.calculateStoreSize(block_cols, block_cols)/(float)new_csr.rows / (float)new_csr.cols;
    std::cout << "new store density: " << store_density << std::endl;
    std::cout << "Group calculation time(CPU):"<<endtime*1000<<"ms"<< std::endl;

    return 0;
}