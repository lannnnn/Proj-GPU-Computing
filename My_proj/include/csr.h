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

    void clean();
    void addRow(struct ROW &row);
    double calculateBlockDensity(int block_rows, int block_cols);
    void print();

    //destructor cleans all arrays
    ~CSR() {
        clean();
    }
};