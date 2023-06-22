#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include "cuda_impl.cuh"

#define BLK_SIZE 256

int main() {

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

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

    std::string filename = "/home/shuxin.zheng/Proj-GPU-Computing/My_proj/data/weighted/494_bus.mtx";
    float tau = 0.8;

    // method allow inordered input data
    COO coo = readMTXFileWeighted(filename);
    // print_matrix(matrix, block_rows); //print matrix message

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
    // print_map(rankMap)
 
    // device memory allocation
    int* d_rowPtr;
    int* d_colIdx;
    int* d_groupList;
    int* resultList;
    GroupInfo* d_groupInfo;
    cudaMalloc((int**)&d_rowPtr, (csr.rows+1) * sizeof(int));
    cudaMalloc((int**)&d_colIdx, csr.nnz * sizeof(int));
    cudaMalloc((int**)&d_groupList, csr.rows * sizeof(int));
    cudaMalloc((int**)&resultList, csr.rows * sizeof(int));
    cudaMalloc((GroupInfo**)&d_groupInfo, csr.rows * sizeof(GroupInfo));
    // data copy to GPU
    cudaMemcpy(d_rowPtr, &csr.rowPtr[0], (csr.rows+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, &csr.colIdx[0], csr.nnz * sizeof(int), cudaMemcpyHostToDevice);

    // groupList initialized as 0..n
    int* h_groupList = (int*)malloc(csr.rows * sizeof(int));
    for(int i=0; i< csr.rows; i++) {
        h_groupList[i] = i;
    }

    cudaMemcpy(d_groupList, h_groupList, csr.rows * sizeof(int), cudaMemcpyHostToDevice);

    int grd_size = (csr.rows+BLK_SIZE)/BLK_SIZE;

    dim3 block_size(BLK_SIZE, 1, 1);
    dim3 grid_size(grd_size, 1, 1);
    gpu_grouping<<< grid_size, block_size>>>(d_rowPtr, d_colIdx, tau, d_groupList, d_groupInfo, resultList, csr.rows,csr.nnz);
    // test<<< grid_size, block_size>>>(d_groupList, resultList);
    cudaDeviceSynchronize();
    // copy data back
    cudaMemcpy(h_groupList, d_groupList, csr.rows * sizeof(int), cudaMemcpyDeviceToHost);
    // clear the memory
    cudaFree(d_rowPtr);
    cudaFree(d_colIdx);
    cudaFree(d_groupList);
    cudaFree(d_groupInfo);
    cudaFree(resultList);

    std::vector<std::vector<int>> fine_group(1, std::vector<int>(csr.rows));
    fine_group[0] = std::vector<int>(&h_groupList[0], &h_groupList[0] + csr.rows);
    print_vec(fine_group);
    //CSR new_csr(csr.rows, csr.cols, csr.nnz);
    //reordering(csr, new_csr, fine_group);

    //new_csr.print();
    //std::cout << "original density" << csr.calculateBlockDensity(64, 64) << std::endl;
    //std::cout << "new density" << new_csr.calculateBlockDensity(64, 64) << std::endl;

    return 0;
}