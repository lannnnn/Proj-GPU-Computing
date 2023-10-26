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
    int list = 0;
    int iter_time;
    int error;

    cudaEvent_t startTime, endTime;
    float elapsedTime = 0.0;
    float totalTime = 0.0;
    float block_density = 0;
    float new_density = 0;
    float best_tau = 0;

    float list_tau[9] = { 0.9f, 0.8f, 0.6f, 0.5f, 0.4f, 0.3f, 0.2f, 0.1f, 0.01f };

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

    float tau = list_tau[4];

    if(argc >= 2) {
        readConfig(argc, argv, &filename, &block_cols, &tau, &print, &mtx, &el, &list);
    }

    std::string ofilename = filename + ".reorder";

    std::cout << "using matrix file: " << filename << std::endl;
    std::cout << "using blocksize: " << block_cols << std::endl;
    
    COO coo ;
    COO mask_coo ;

    if(mtx==1)  {
        coo = readMTXFileUnweighted(filename);
        mask_coo = readMTXFileMask(filename, block_cols);
        std::cout << "Using MTX format" << std::endl;
    } else {
        coo = readELFileUnweighted(filename);
        mask_coo = readELFileMask(filename, block_cols);
        std::cout << "Using EL format" << std::endl;
    }
    if(coo.rows == 0) {
        std::cout << "not acceptable matrix file" << std::endl;
        return;
    }

    CSR csr(coo.rows, coo.cols, coo.nnz);
    CSR mask_csr(mask_coo.rows, mask_coo.cols, mask_coo.nnz);

    if(print) {
        print_matrix(mask_coo, mask_coo.rows);
    }
    // csr to coo, build the rankMap at same time
    cooToCsr(coo, csr);
    cooToCsr(mask_coo, mask_csr);

    // build priority queue
    std::vector<int> priority_queue;
    dense_priority_ref(priority_queue, mask_csr);

    // free the matrix, use csr
    coo.clean();
    mask_coo.clean();

    if(list) {iter_time = 9; tau = list_tau[0];}
    else {iter_time = 1;}
 
    // device memory allocation
    int* d_rowPtr;
    int* d_colIdx;
    int* d_groupList;
    int* d_groupSize;
    int* d_refRow;
    int* d_rows_per_thread;
    int* d_priority_queue;
    float* d_tau;

    int* h_groupList = (int*)malloc((csr.rows+1) * sizeof(int));
    int* h_resultList = (int*)malloc(csr.rows * sizeof(int));
    int* h_rowPtr = (int*)malloc((csr.rows+1) * sizeof(int));
    int* h_colIdx = (int*)malloc(mask_csr.nnz * sizeof(int));
    // int* h_colIdx = (int*)malloc(csr.nnz * sizeof(int));
    int* h_groupSize = (int*)malloc(sizeof(int));

    int rows_per_thread = 0;
    
    CSR new_csr(csr.rows, csr.cols, csr.nnz);
    
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    CHECK( cudaMalloc((int**)&d_rowPtr, (csr.rows+1) * sizeof(int)));
    CHECK( cudaMalloc((int**)&d_colIdx, mask_csr.nnz * sizeof(int)));
    // CHECK( cudaMalloc((int**)&d_colIdx, csr.nnz * sizeof(int)));
    CHECK( cudaMalloc((int**)&d_groupList, (csr.rows+1) * sizeof(int)));
    CHECK( cudaMalloc((int**)&d_groupSize, sizeof(int)));
    CHECK( cudaMalloc((int**)&d_refRow, ref_size * sizeof(int)));
    CHECK( cudaMalloc((float**)&d_tau, sizeof(float)));
    CHECK( cudaMalloc((int**)&d_rows_per_thread, sizeof(int)));

    CHECK( cudaMemcpy(d_rowPtr, &mask_csr.rowPtr[0], (csr.rows+1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK( cudaMemcpy(d_colIdx, &mask_csr.colIdx[0], mask_csr.nnz * sizeof(int), cudaMemcpyHostToDevice));
    // CHECK( cudaMemcpy(d_rowPtr, &csr.rowPtr[0], (csr.rows+1) * sizeof(int), cudaMemcpyHostToDevice));
    // CHECK( cudaMemcpy(d_colIdx, &csr.colIdx[0], csr.nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK( cudaMalloc((int**)&d_priority_queue, csr.rows * sizeof(int)));
    CHECK( cudaMemcpy(d_priority_queue, &priority_queue[0], csr.rows * sizeof(int), cudaMemcpyHostToDevice));

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
    rows_per_thread = (csr.rows + totalThreads-1) / totalThreads;
    // if(rows_per_thread < row_size) rows_per_thread = row_size;
    CHECK( cudaMemcpy(d_rows_per_thread, &rows_per_thread, sizeof(int), cudaMemcpyHostToDevice));
    std::cout << "Start calculating with dimGrid " << grdDim << ", dimBlock " << numThreads << "..." << std::endl;
    std::cout << "matrix info: nrows=" << csr.rows << ", ncols=" << csr.cols << ", nnz=" << csr.nnz << std::endl;
    std::cout << "rows_per_thread=" << rows_per_thread << std::endl;

    cudaStream_t s1;
    cudaStreamCreate(&s1);

    for(int itr=0; itr<iter_time; itr++) {
        if(itr > 0) tau = list_tau[itr];
        std::cout << "using tau: " << tau << std::endl;

        // groupList initialized as 0..n
        for(int i=0; i< csr.rows; i++) {
            h_groupList[i] = -1;
            h_resultList[i] = i;
        }
        h_groupList[csr.rows] = -1;
        h_groupSize[0] = csr.rows;

        // data copy to GPU
        CHECK( cudaMemcpyAsync(d_tau, &tau, sizeof(float), cudaMemcpyHostToDevice, s1));
        CHECK( cudaMemsetAsync(d_refRow, 0, ref_size * sizeof(int), s1));
        CHECK( cudaMemcpyAsync(d_groupList, h_groupList, (csr.rows+1) * sizeof(int), cudaMemcpyHostToDevice, s1));
        CHECK( cudaMemcpyAsync(d_groupSize, h_groupSize, sizeof(int), cudaMemcpyHostToDevice, s1));

        // void *kernelArgs[] = {(void *)&d_rowPtr, (void *)&d_colIdx, (void *)&d_tau, (void *)&d_groupList,
        //                    (void *)&d_groupSize, (void *)&d_refRow, (void *)&d_rows_per_thread};
        // std::cout << "matrix size: (rows, cols, nnz) = (" << csr.rows << ", " << csr.cols << ", " << csr.nnz << ")" << std::endl;
        void *kernelArgs[] = {(void *)&d_rowPtr, (void *)&d_colIdx, (void *)&d_tau, (void *)&d_groupList,
                            (void *)&d_groupSize, (void *)&d_refRow, (void *)&d_rows_per_thread, (void *)&d_priority_queue};

        cudaEventCreate(&startTime);
        cudaEventCreate(&endTime);
        cudaEventRecord(startTime, 0);

        cudaLaunchCooperativeKernel((void*)gpu_ref_grouping_O2, dimGrid, dimBlock, kernelArgs, 0, s1);

        cudaEventRecord(endTime, 0);
        cudaEventSynchronize(startTime);
        cudaEventSynchronize(endTime);
        cudaEventElapsedTime(&elapsedTime, startTime, endTime);

        totalTime += elapsedTime;

        cudaEventDestroy(startTime);
        cudaEventDestroy(endTime);

        CHECK( cudaMemcpyAsync(h_groupList, d_groupList, (csr.rows) * sizeof(int), cudaMemcpyDeviceToHost, s1));
        // print_pointer(h_groupList, csr.rows);
        std::vector<std::vector<int>> fine_group(csr.rows+1);
        std::vector<int> res_vec(csr.rows);

        for(int i=0; i<csr.rows; i++) {
            fine_group[h_groupList[i]].push_back(i);
        }
        printRes(fine_group, res_vec, ofilename);
        res_vec.clear();
        // print_pointer(h_resultList, csr.rows);
        if(print) {
            std::cout << "Reordered row rank:" << std::endl;
            print_vec(fine_group);
        }
        reordering(csr, new_csr, fine_group);

        new_density = new_csr.calculateBlockDensity(block_cols, block_cols);
        std::cout << "group number: " << count_group(fine_group) << std::endl;
        std::cout << "new_density: " << new_density << std::endl;
        std::cout << "elapsed time: " << elapsedTime << " ms" << std::endl;
        if(block_density < new_density) {
            best_tau = tau;
            block_density = new_density;
        }
        fine_group.clear();
        cudaStreamSynchronize(s1);
    }

    // clear the cuda memory
    CHECK( cudaFree(d_rowPtr));
    CHECK( cudaFree(d_colIdx));
    CHECK( cudaFree(d_groupList));
    CHECK( cudaFree(d_groupSize));
    CHECK( cudaFree(d_refRow));
    CHECK( cudaFree(d_tau));
    CHECK( cudaFree(d_rows_per_thread));

    // new_csr.print();
    std::cout << "=============SUMMARY=================" << std::endl;
    std::cout << "matrix size: nrows=" << csr.rows << ", ncols=" << csr.cols << ", nnz=" << csr.nnz << std::endl;
    if(list) {
        std::cout << "checking for using block size: (" << block_cols << "," << block_cols << ")" << std::endl;
        std::cout << "original density: " << csr.calculateBlockDensity(block_cols, block_cols) << std::endl;
        std::cout << "best tau: " << best_tau << std::endl;
        std::cout << "best density: " << block_density << std::endl;
        std::cout << "Total calculation time(GPU): " << totalTime << " ms" << std::endl;
    } else {
        std::cout << "checking for using block size: (" << block_cols << "," << block_cols << ")" << std::endl;
        std::cout << "original density: " << csr.calculateBlockDensity(block_cols, block_cols) << std::endl;
        std::cout << "using tau: " << tau << std::endl;
        std::cout << "new density: " << new_csr.calculateBlockDensity(block_cols, block_cols) << std::endl;
        std::cout << "Group calculation time(GPU): " << elapsedTime << " ms" << std::endl;
    }

    return 0;
}