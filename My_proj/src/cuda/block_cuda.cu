#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include "cuda_impl.cuh"

#define BLK_SIZE 256

int main(int argc, char* argv[]) {

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    int block_cols = 8;
    int print = 0;

    if (error_id != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n",
           static_cast<int>(error_id), cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0) {
        printf("There are no available device(s) that support CUDA\n");
        return -1;
    } else {
        cudaSetDevice(0);
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

    std::string filename = "/home/shuxin.zheng/Proj-GPU-Computing/My_proj/data/weighted/TEST_matrix_weighted.el";
    // std::string filename = "/home/shuxin.zheng/Proj-GPU-Computing/My_proj/data/unweighted/0_mycielskian13.el";
    // std::string filename = "/home/shuxin.zheng/Proj-GPU-Computing/My_proj/data/unweighted/seventh_graders.el";

    float tau = 0.8;
    // method allow inordered input data

    if(argc >= 2) {
        readConfig(argc, argv, &filename, &block_cols, &tau, &print);
    }

    std::cout << "using matrix file: " << filename << std::endl;
    std::cout << "using blocksize: " << block_cols << std::endl;
    std::cout << "using tau: " << tau << std::endl;

    // COO coo = readELFileWeighted(filename);
    COO coo = readELFileUnweighted(filename);

    CSR csr(coo.rows, coo.cols, coo.nnz);
    // csr to coo, build the rankMap at same time
    cooToCsr(coo, csr);
    // free the matrix, use csr
    coo.row_message.clear();
 
    // device memory allocation
    int* d_rowPtr;
    int* d_colIdx;
    int* d_groupList;
    int* d_resultList;
    int* d_groupSize;
    int* d_refRow;
    GroupInfo* d_groupInfo;
    CHECK( cudaMalloc((int**)&d_rowPtr, (csr.rows+1) * sizeof(int)));
    CHECK( cudaMalloc((int**)&d_colIdx, csr.nnz * sizeof(int)));
    CHECK( cudaMalloc((int**)&d_groupList, (csr.rows+1) * sizeof(int)));
    CHECK( cudaMalloc((int**)&d_resultList, csr.rows * sizeof(int)));
    CHECK( cudaMalloc((int**)&d_groupSize, sizeof(int)));
    CHECK( cudaMalloc((int**)&d_refRow, ref_size * sizeof(int)));
    CHECK( cudaMemset(d_refRow, 0, ref_size * sizeof(int)));
    CHECK( cudaMalloc((GroupInfo**)&d_groupInfo, csr.rows * sizeof(GroupInfo)));
    // data copy to GPU
    CHECK( cudaMemcpy(d_rowPtr, &csr.rowPtr[0], (csr.rows+1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK( cudaMemcpy(d_colIdx, &csr.colIdx[0], csr.nnz * sizeof(int), cudaMemcpyHostToDevice));

    // groupList initialized as 0..n
    int* h_groupList = (int*)malloc((csr.rows+1) * sizeof(int));
    int* h_resultList = (int*)malloc(csr.rows * sizeof(int));
    int* h_rowPtr = (int*)malloc((csr.rows+1) * sizeof(int));
    int* h_colIdx = (int*)malloc(csr.nnz * sizeof(int));
    for(int i=0; i< csr.rows; i++) {
        h_groupList[i] = i;
        h_resultList[i] = i;
    }
    h_groupList[csr.rows] = csr.rows;

    int* h_groupSize = (int*)malloc(sizeof(int));
    h_groupSize[0] = csr.rows;
    CHECK( cudaMemcpy(d_groupList, h_groupList, (csr.rows+1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK( cudaMemcpy(d_resultList, h_resultList, csr.rows * sizeof(int), cudaMemcpyHostToDevice));
    CHECK( cudaMemcpy(d_groupSize, h_groupSize, sizeof(int), cudaMemcpyHostToDevice));

    int grd_size = (csr.rows+BLK_SIZE)/BLK_SIZE;

    dim3 block_size(BLK_SIZE, 1, 1);
    dim3 grid_size(grd_size, 1, 1);
    // gpu_grouping<<< grid_size, block_size>>>(d_rowPtr, d_colIdx, tau, d_groupList, d_groupInfo, d_resultList, 
    //                                            d_groupSize, grd_size*BLK_SIZE, block_cols);

    std::cout << "Compute finished" << std::endl;

    CHECK( cudaMemcpy(h_resultList, d_resultList, csr.rows * sizeof(int), cudaMemcpyDeviceToHost));
    //tmp here
    CHECK( cudaMemcpy(h_groupList, d_groupList, (h_groupSize[0]+1) * sizeof(int), cudaMemcpyDeviceToHost));
    // clear the cuda memory
    CHECK( cudaFree(d_rowPtr));
    CHECK( cudaFree(d_colIdx));
    CHECK( cudaFree(d_groupList));
    CHECK( cudaFree(d_resultList));
    CHECK( cudaFree(d_groupSize));
    CHECK( cudaFree(d_refRow));
    CHECK( cudaFree(d_groupInfo));

    std::vector<std::vector<int>> fine_group(1, std::vector<int>(csr.rows));
    fine_group[0] = std::vector<int>(&h_resultList[0], &h_resultList[0] + csr.rows);
    if(print) {
        std::cout << "Reordered row rank:" << std::endl;
        print_vec(fine_group);
    }

    CSR new_csr(csr.rows, csr.cols, csr.nnz);
    reordering(csr, new_csr, fine_group);

    // new_csr.print();
    std::cout << "matrix info: nrows=" << csr.rows << ", ncols=" << csr.cols << ", nnz=" << csr.nnz << std::endl;
    std::cout << "checking for using block size: (" << block_cols << "," << block_cols << ")" << std::endl;
    std::cout << "original density: " << csr.calculateBlockDensity(block_cols, block_cols) << std::endl;
    std::cout << "new density: " << new_csr.calculateBlockDensity(block_cols, block_cols) << std::endl;
    std::cout << "GPU calculation time: " << std::endl;
    return 0;
}