#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include "cuda_impl.cuh"

#define BLK_SIZE 256

int main(int argc, char* argv[]) {

    int deviceCount = 0;
    int device;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    int block_cols = 8;
    int print = 0;
    int mtx = 0, el = 0;

    cudaEvent_t startTime, endTime;
    float elapsedTime = 0.0;

    // This will launch a grid that can maximally fill the GPU, on the default stream with kernel arguments
    int numBlocksPerSm = 0;
    // Number of threads my_kernel will be launched with
    int numThreads = BLK_SIZE;

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
        cudaGetDevice(&device);
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

    // std::string filename = "/home/shuxin.zheng/Proj-GPU-Computing/My_proj/data/weighted/TEST_matrix_weighted.el";
    // std::string filename = "/home/shuxin.zheng/Proj-GPU-Computing/My_proj/data/unweighted/0_mycielskian13.el";
    std::string filename = "/home/shuxin.zheng/Proj-GPU-Computing/My_proj/data/unweighted/seventh_graders.el";

    float tau = 0.4;

    if(argc >= 2) {
        readConfig(argc, argv, &filename, &block_cols, &tau, &print, &mtx, &el);
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
    if(coo.rows == 0) {
        std::cout << "not acceptable matrix file" << std::endl;
        return;
    }

    CSR csr(coo.rows, coo.cols, coo.nnz);
    // csr to coo, build the rankMap at same time
    cooToCsr(coo, csr);
    // free the matrix, use csr
    coo.row_message.clear();
 
    // device memory allocation
    int* d_rowPtr;
    int* d_colIdx;
    int* d_groupList;
    int* d_groupSize;
    int* d_refRow;
    int* d_rows_per_thread;
    float* d_tau;
    CHECK( cudaMalloc((int**)&d_rowPtr, (csr.rows+1) * sizeof(int)));
    CHECK( cudaMalloc((int**)&d_colIdx, csr.nnz * sizeof(int)));
    CHECK( cudaMalloc((int**)&d_groupList, (csr.rows+1) * sizeof(int)));
    CHECK( cudaMalloc((int**)&d_groupSize, sizeof(int)));
    CHECK( cudaMalloc((int**)&d_refRow, ref_size * sizeof(int)));
    CHECK( cudaMalloc((float**)&d_tau, sizeof(float)));
    CHECK( cudaMalloc((int**)&d_rows_per_thread, sizeof(int)));
    CHECK( cudaMemset(d_refRow, 0, ref_size * sizeof(int)));
    // data copy to GPU
    CHECK( cudaMemcpy(d_rowPtr, &csr.rowPtr[0], (csr.rows+1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK( cudaMemcpy(d_colIdx, &csr.colIdx[0], csr.nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK( cudaMemcpy(d_tau, &tau, sizeof(float), cudaMemcpyHostToDevice));

    // groupList initialized as 0..n
    int* h_groupList = (int*)malloc((csr.rows+1) * sizeof(int));
    int* h_resultList = (int*)malloc(csr.rows * sizeof(int));
    int* h_rowPtr = (int*)malloc((csr.rows+1) * sizeof(int));
    int* h_colIdx = (int*)malloc(csr.nnz * sizeof(int));
    for(int i=0; i< csr.rows; i++) {
        h_groupList[i] = -1;
        h_resultList[i] = i;
    }
    h_groupList[csr.rows] = -1;

    int* h_groupSize = (int*)malloc(sizeof(int));
    h_groupSize[0] = csr.rows;
    CHECK( cudaMemcpy(d_groupList, h_groupList, (csr.rows+1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK( cudaMemcpy(d_groupSize, h_groupSize, sizeof(int), cudaMemcpyHostToDevice));

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    CHECK( cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, gpu_ref_grouping_O1, BLK_SIZE, 0) );
    int totalThreads = deviceProp.multiProcessorCount*numBlocksPerSm*BLK_SIZE;
    // void *kernelArgs[] = {(void *)&d_rowPtr, (void *)&d_colIdx, (void *)&d_tau, (void *)&d_groupList,
    //                             (void *)&d_groupSize, (void *)&d_refRow};
    int grdDim = deviceProp.multiProcessorCount*numBlocksPerSm;
    if(totalThreads > (csr.rows + row_size - 1)/row_size) {
        // ref row size >=2
        totalThreads = (csr.rows + row_size - 1)/row_size;
        grdDim = (totalThreads + BLK_SIZE - 1)/BLK_SIZE;
    }
    dim3 dimGrid(grdDim, 1, 1);
    dim3 dimBlock(BLK_SIZE, 1, 1);
    int rows_per_thread = (csr.rows + totalThreads-1) / totalThreads;
    // if(rows_per_thread < row_size) rows_per_thread = row_size;

    CHECK( cudaMemcpy(d_rows_per_thread, &rows_per_thread, sizeof(int), cudaMemcpyHostToDevice));
    void *kernelArgs[] = {(void *)&d_rowPtr, (void *)&d_colIdx, (void *)&d_tau, (void *)&d_groupList,
                            (void *)&d_groupSize, (void *)&d_refRow, (void *)&d_rows_per_thread};
    // std::cout << "matrix size: (rows, cols, nnz) = (" << csr.rows << ", " << csr.cols << ", " << csr.nnz << ")" << std::endl;
    std::cout << "Start calculating with dimGrid " << grdDim << ", dimBlock " << numThreads << "..." << std::endl;
    std::cout << "matrix info: nrows=" << csr.rows << ", ncols=" << csr.cols << ", nnz=" << csr.nnz << std::endl;
    std::cout << "rows_per_thread=" << rows_per_thread << std::endl;
    cudaEventCreate(&startTime);
    cudaEventCreate(&endTime);
    cudaEventRecord(startTime, 0);

    cudaLaunchCooperativeKernel((void*)gpu_ref_grouping_O1, dimGrid, dimBlock, kernelArgs);

    cudaEventRecord(endTime, 0);
    cudaEventSynchronize(startTime);
    cudaEventSynchronize(endTime);
    cudaEventElapsedTime(&elapsedTime, startTime, endTime);

    CHECK( cudaMemcpy(h_groupList, d_groupList, (csr.rows) * sizeof(int), cudaMemcpyDeviceToHost));
    // print_pointer(h_groupList, csr.rows);
    std::vector<std::vector<int>> fine_group(csr.rows+1);
    for(int i=0; i<csr.rows; i++) {
        fine_group[h_groupList[i]].push_back(i);
    }
    // print_pointer(h_resultList, csr.rows);
    if(print) {
        std::cout << "Reordered row rank:" << std::endl;
        print_vec(fine_group);
    }
    CSR new_csr(csr.rows, csr.cols, csr.nnz);
    reordering(csr, new_csr, fine_group);

    // clear the cuda memory
    CHECK( cudaFree(d_rowPtr));
    CHECK( cudaFree(d_colIdx));
    CHECK( cudaFree(d_groupList));
    CHECK( cudaFree(d_groupSize));
    CHECK( cudaFree(d_refRow));
    CHECK( cudaFree(d_tau));

    cudaEventDestroy(startTime);
    cudaEventDestroy(endTime);

    // new_csr.print();
    std::cout << "checking for using block size: (" << block_cols << "," << block_cols << ")" << std::endl;
    std::cout << "original density: " << csr.calculateBlockDensity(block_cols, block_cols) << std::endl;
    std::cout << "new density: " << new_csr.calculateBlockDensity(block_cols, block_cols) << std::endl;
    std::cout << "Group calculation time(GPU): " << elapsedTime << " ms" << std::endl;
    return 0;
}