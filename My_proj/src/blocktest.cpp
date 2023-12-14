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
    float list_tau[10] = { 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01 };

    if(argc >= 2) {
        readConfig(argc, argv, &filename, &block_cols, &tau, &print, &mtx, &el, &list, &ofilename);
    }

    std::cout << "using matrix file: " << filename << std::endl;
    std::cout << "using blocksize: " << block_cols << std::endl;
    std::cout << "using tau: " << tau << std::endl;

    COO coo ;
    //COO mask_coo ;

    if(mtx==1)  {
        coo = readMTXFileUnweighted(filename);
        //mask_coo = readMTXFileMask(filename, block_cols);
        std::cout << "Using MTX format" << std::endl;
    } else {
        coo = readELFileUnweighted(filename);
        //mask_coo = readELFileMask(filename, block_cols);
        std::cout << "Using EL format" << std::endl;
    }
    if(coo.rows == 0) {
        std::cout << "not acceptable matrix file" << std::endl;
        return -1;
    }
    // init label list for distance calculation
    std::vector<std::vector<int>> label(coo.rows); //,std::vector<int>(labelSize));
    // std::multimap<int, int, std::greater<int>> rankMap;

    CSR csr(coo.rows, coo.cols, coo.nnz);
    //CSR mask_csr(mask_coo.rows, mask_coo.cols, mask_coo.nnz);
    CSR new_csr(csr.rows, csr.cols, csr.nnz);
    // csr to coo, build the rankMap at same time
    cooToCsr(coo, csr);
    //cooToCsr(mask_coo, mask_csr);
    // print_matrix(coo, 1);
    // free the matrix, use csr
    coo.row_message.clear();

    // init coarse graind group vector
    // we have a little larger buffer to save in case there be one group have more elements than others, but not too much..
    std::vector<int> coarse_group;
    // init fine graind group vector
    // give little more space for blocks in each thread which can not be totally filled

    for (int i=0; i<csr.rows; i++) { 
        coarse_group.push_back(i);
    }

    // build priority queue
    std::vector<int> priority_queue;
    std::vector<int> priority_queue_dup;
    dense_priority_ref(priority_queue, csr);

    // for(int i=0; i<csr.rows; i++) {
    //     std::cout << priority_queue[i] << std::endl;
    // }

    int iter_time = 0;
    if(list) {iter_time = 10; tau = list_tau[0];}
    else {iter_time = 1;}

    std::cout << "matrix info: nrows=" << csr.rows << ", ncols=" << csr.cols << ", nnz=" << csr.nnz << std::endl;
    std::cout << "checking for using block size: (" << block_cols << "," << block_cols << ")" << std::endl;
    std::cout << "original density: " << csr.calculateBlockDensity(block_cols, block_cols) << std::endl;
    std::cout << "original store density: " << csr.calculateStoreSize(block_cols, block_cols)/(float)csr.rows / (float)csr.cols << std::endl;

    for(int itr=0; itr<iter_time; itr++) {
        std::vector<std::vector<int>> fine_group(csr.rows+1);
        if(itr > 0) tau = list_tau[itr];
        priority_queue_dup = priority_queue;
        start=clock();
        //fine_grouping(coarse_group, csr, fine_group, tau);
        int iterCnt = fine_grouping(priority_queue_dup, csr, fine_group, tau);
        end=clock();

        double endtime=(double)(end-start)/CLOCKS_PER_SEC;

        if(print) {
            std::cout << "Reordered row rank:" << std::endl;
            print_vec(fine_group);  
        }

        std::vector<int> res_vec(csr.rows);

        std::string oname = ofilename + "." + std::to_string(tau);
        printRes(fine_group, res_vec, oname);

        reordering(csr, new_csr, fine_group);
        std::cout << "using tau: " << tau << std::endl;
        std::cout << "group number: " << count_group(fine_group) << std::endl;
        std::cout << "iterating for " << iterCnt << " times" << std::endl;
        std::cout << "new density: " << new_csr.calculateBlockDensity(block_cols, block_cols) << std::endl;
        float store_density = new_csr.calculateStoreSize(block_cols, block_cols)/(float)new_csr.rows / (float)new_csr.cols;
        std::cout << "new store density: " << store_density << std::endl;
        std::cout << "Group calculation time(CPU):"<<endtime*1000<<"ms"<< std::endl;
        fine_group.clear();
    }

    return 0;
}