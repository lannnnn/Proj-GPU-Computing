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

    //constructor for edgelist data
    // CSR(std::ifstream& infile, std::string delimiter = " ", bool pattern_only = true, MatrixFormat mat_fmt = mtx)
    // {
    //     read_from_edgelist(infile, delimiter, pattern_only, mat_fmt);
    // }

    CSR() : rows(0), cols(0), nnz(0), rowPtr(rows+1), colIdx(), values() {}

    CSR(int rows, int cols, int nnz) : rows(rows), cols(cols), nnz(nnz), rowPtr(rows+1), colIdx(), values() {}

    void addRow(struct ROW &row) {
        rowPtr[row.rowIdx+1] = rowPtr[row.rowIdx] + row.rank;
        for(auto it : row.nzValue){
            colIdx.push_back(it.first);
            values.push_back(it.second);
        }
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
    int max_rank = 0;
    // std::vector<int> rowIdx;
    // std::vector<int> colIdx;
    // std::vector<double> values;
    std::vector<struct ROW> row_message;

    void clean();

    MATRICES(){}

    MATRICES(int rows, int cols, int nnz)
        : rows(rows), cols(cols), nnz(nnz), row_message(rows) {}
};

