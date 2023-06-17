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

    void clean();

    COO(){}
    COO(int rows, int cols, int nnz)
        : rows(rows), cols(cols), nnz(nnz), row_message(rows) {}
    //destructor cleans all arrays
    ~COO() {
        clean();
    }
};

