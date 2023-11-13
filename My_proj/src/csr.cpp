#include "csr.h"
#include "coo.h"

void CSR::clean() { 
    /*----------------------------------------------------------------------
    | Free up memory allocated for CSR structs.
    |--------------------------------------------------------------------*/

    if (rows + cols <= 1) return;

    rowPtr.clear();
    colIdx.clear();
    values.clear();
    
    rows = 0;
    cols = 0; 
}

void CSR::addRow(struct ROW &row) {
    rowPtr[row.rowIdx+1] = rowPtr[row.rowIdx] + row.nzValue.size();
    for(auto it : row.nzValue){
        colIdx.push_back(it.first);
        values.push_back(it.second);
    }
}

double CSR::calculateBlockDensity(int block_rows, int block_cols) {
    int totBlockRows = (rows+1) / block_rows;
    int totBlockCols = (cols+1) / block_cols;
    std::vector<int> blocks;
    for(int i = 0; i < rows; i++) {
        //std::cout << "i=" << i <<"rowPtr[i+1] - rowPtr[i]" << rowPtr[i+1]-rowPtr[i]<< std::endl;
        for(int j=0; j< (rowPtr[i+1]-rowPtr[i]); j++) {
            // std::cout << "i = " << i <<"j=" << j<< std::endl;
            // std::cout << "rowPtr[i]+j = " << rowPtr[i]+j <<"colIdx[rowPtr[i]+j]=" << colIdx[rowPtr[i]+j]<< std::endl;
            // std::cout << "colIdx.size = " << colIdx.size() << std::endl;
            blocks.push_back((i / block_rows) * totBlockCols + colIdx[rowPtr[i]+j]/block_cols);
            // std::cout << "i = " << i << " blockid = " << (i / block_rows) * totBlockCols + colIdx[rowPtr[i]+j]/block_cols << std::endl;
            // std::cout << "totBlockCols = " << totBlockCols << std::endl;
            // std::cout << "colIdx[rowPtr[i]+j]/block_cols = "<<  colIdx[rowPtr[i]+j]/block_cols << std::endl;
        }
    }
    //std::cout << "yet ok??" << std::endl;
    std::sort(blocks.begin(), blocks.end());
    blocks.erase(std::unique(blocks.begin(), blocks.end()), blocks.end());
    // for(int i=0; i< blocks.size(); i++) {
    //     std::cout << blocks[i] << " ";
    // }
    // std::cout << std::endl;
    return (double)nnz / (double)(blocks.size() * block_rows * block_cols);
}

double CSR::calculateBlockDensityDev(int block_rows, int block_cols) {
    int totBlockRows = (rows+1) / block_rows;
    int totBlockCols = (cols+1) / block_cols;
    int blockIdx = 0;
    int blockCnt = 0;
    std::vector<int> blocks;
    for(int i = 0; i < rows; i++) {
        for(int j=0; j< (rowPtr[i+1]-rowPtr[i]); j++) {
            blockIdx = (i / block_rows) * totBlockCols + colIdx[rowPtr[i]+j]/block_cols;
            while((blockIdx+1) > blocks.size()) {
                blocks.push_back(0);
            }
            if(blocks[blockIdx] == 0) blockCnt ++;
            blocks[blockIdx] ++;
        }
    }
    double avg = (double)nnz / (double)(blockCnt * block_rows * block_cols);
    double dev = 0;
    for(int i=0; i<blocks.size(); i++) {
        if(blocks[i] == 0) continue;
        dev += std::pow((double)blocks[i]/(double)(block_rows * block_cols)-avg, 2);
    }

    dev = std::pow(dev/(double)(blockCnt),0.5);

    return dev;
}

double CSR::calculateStoreSize(int block_rows, int block_cols) {
    int totBlockRows = (rows+1) / block_rows;
    int totBlockCols = (cols+1) / block_cols;
    std::vector<int> blocks;
    for(int i = 0; i < rows; i++) {
        for(int j=0; j< (rowPtr[i+1]-rowPtr[i]); j++) {
            blocks.push_back((i / block_rows) * totBlockCols + colIdx[rowPtr[i]+j]/block_cols);
        }
    }
    std::sort(blocks.begin(), blocks.end());
    blocks.erase(std::unique(blocks.begin(), blocks.end()), blocks.end());
    return (double)(blocks.size() * block_rows * block_cols);
}

void CSR::print(){
    std::cout << "PRINTING A CSR MATRIX (arrays only)" << std::endl;
    std::cout << "ROWS: " << rows << " COLS: " << cols << std::endl; 
    std::cout << "NZ: " << nnz << std::endl;

    std::cout << "ROWPTR:" << std::endl;
    for (int i = 0; i <= rows; i++) {
        std::cout << rowPtr[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "COLIDX:" << std::endl;
    for (int i = 0; i < nnz; i++) {
        std::cout << colIdx[i] << " "; 
    }
    std::cout << std::endl;

    // just ignore value for now
    // std::cout << "NZVALUE:" << std::endl;
    // for (int i = 0; i < nnz; i++) {
    //     std::cout << values[i] << " ";
    // }
    // std::cout << std::endl;
}

