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
        for(int j=0; j< (rowPtr[i+1]-rowPtr[i]); j++) {
            blocks.push_back((i / block_rows) * totBlockCols + colIdx[rowPtr[i]+j]/block_cols);
        }
    }
    blocks.erase(std::unique(blocks.begin(), blocks.end()), blocks.end());
    return (double)nnz / (double)(blocks.size() * block_rows * block_cols);
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

