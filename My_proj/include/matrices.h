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
    std::vector<int> rowPtr;
    std::vector<int> colIdx;
    std::vector<double> values;

    void clean();

    //constructor for edgelist data
    // CSR(std::ifstream& infile, std::string delimiter = " ", bool pattern_only = true, MatrixFormat mat_fmt = mtx)
    // {
    //     read_from_edgelist(infile, delimiter, pattern_only, mat_fmt);
    // }
    
    
    //destructor cleans all arrays
    ~CSR()
    {
        clean();
    }

};

struct MATRICES
{
    /*--------------------------------------------------------------
    | Coordinate sparse row (COO) matrix format,
    | generally not square
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

    MATRICES(){}

    MATRICES(int rows, int cols, int nnz)
        : rows(rows), cols(cols), nnz(nnz), row_message(rows) {}
};

