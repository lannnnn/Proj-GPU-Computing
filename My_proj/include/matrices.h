#pragma once

#include <string>
#include <vector>
#include <fstream>
#include "row.h"

struct CSR
{
    /*--------------------------------------------------------------
    | Compressed sparse row (CSR) matrix format,
    | generally not square
    |==========
    |       cmat  = a CSR struct
    |       rows  = # of columns of the matrix
    |       cols  = # of columns of the matrix
    |       nnz = # of non-zero values of the matrix
    |--------------------------------------------------------------------   */
    int rows;
    int cols;
    int nnz;
    double blockDensity;
    std::vector<int> rowPtr;
    std::vector<int> colIdx;
    std::vector<double> values;

    CSR() : rows(0), cols(0), nnz(0), rowPtr(rows+1), colIdx(), values() {}

    CSR(int rows, int cols, int nnz) : rows(rows), cols(cols), nnz(nnz), rowPtr(rows+1), colIdx(), values() {}

    void clean() {
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

    //destructor cleans all arrays
    ~CSR() {
        clean();
    }

    void addRow(struct ROW &row) {
        rowPtr[row.rowIdx+1] = rowPtr[row.rowIdx] + row.nzValue.size();
        for(auto it : row.nzValue){
            colIdx.push_back(it.first);
            values.push_back(it.second);
        }
    }

    double calculateBlockDensity(int block_rows, int block_cols) {
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

    void print(){
        std::cout << "PRINTING A CSR MATRIX (arrays only)" << std::endl;
        std::cout << "ROWS: " << rows << " COLS: " << cols << std::endl; 
        std::cout << "NZ: " << nnz << std::endl;

        std::cout << "ROWPTR:" << std::endl;
        for (int i = 0; i <= rows; i++)
        {
                std::cout << rowPtr[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "COLIDX:" << std::endl;
        for (int i = 0; i < nnz; i++)
        {
                std::cout << colIdx[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "NZVALUE:" << std::endl;
        for (int i = 0; i < nnz; i++)
        {
                std::cout << values[i] << " ";
        }
        std::cout << std::endl;
    }

};

struct COO
{
    /*--------------------------------------------------------------
    | Coordinate sparse row (COO) matrix format,
    | design in purpose to read the inordered input file, order the matrix during reading
    | then convert to CSR
    |==========
    |       cmat  = a CSR struct
    |       rows  = # of rows of the matrix
    |       cols  = # of columns of the matrix
    |       nnz = # of non-zero values of the matrix
    |--------------------------------------------------------------------   */
    int rows;
    int cols;
    int nnz;
    // std::vector<int> rowIdx;
    // std::vector<int> colIdx;
    // std::vector<double> values;
    std::vector<struct ROW> row_message;

    void clean() {
        row_message.clear();
        rows = 0;
        cols = 0;
        nnz = 0;
    }

    COO(){}

    COO(int rows, int cols, int nnz)
        : rows(rows), cols(cols), nnz(nnz), row_message(rows) {}

    //destructor cleans all arrays
    ~COO() {
        clean();
    }
};

