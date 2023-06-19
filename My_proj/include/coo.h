#pragma once

#include <string>
#include <vector>
#include <fstream>
#include "row.h"

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

