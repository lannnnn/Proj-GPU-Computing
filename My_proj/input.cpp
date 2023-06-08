#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include "include/matrices.h"

MATRICES readMTXFileWeighted(const std::string& filename) {
    std::ifstream fin;
    fin.open(filename,std::ios_base::in);
    if (!fin.is_open()) {
        std::cout << "Failed to open file: " << filename << std::endl;
        return MATRICES();
    }

    std::string line;
    int rows = 0;
    int cols = 0;
    int nnz = 0;

    int row, col;
    double value;

    // Skip header lines
    while (std::getline(fin, line) && line[0] == '%') {
        // Skip comment lines
    }

    // Read matrix size and nnz
    std::istringstream iss(line);
    iss >> rows >> cols >> nnz;

    MATRICES matrix(rows, cols, nnz);
    while (std::getline(fin, line)) {
        std::istringstream iss(line);
        iss >> row >> col >> value;

        // Adjust for 1-based indexing in MTX format
        row--;
        col--;

        matrix.row_message[row].nzRowCount++;
        matrix.row_message[row].nzValue.insert(std::make_pair(col, value));
    }

    fin.close();
    return matrix;
}

MATRICES readMTXFileUnweighted(const std::string& filename) {
    std::ifstream fin;
    fin.open(filename,std::ios_base::in);
    if (!fin.is_open()) {
        std::cout << "Failed to open file: " << filename << std::endl;
        return MATRICES();
    }

    std::string line;
    int rows = 0;
    int cols = 0;
    int nnz = 0;

    int row, col;

    // Skip header lines
    while (std::getline(fin, line) && line[0] == '%') {
        // Skip comment lines
    }

    // Read matrix size and nnz
    std::istringstream iss(line);
    iss >> rows >> cols >> nnz;

    MATRICES matrix(rows, cols, nnz);
    while (std::getline(fin, line)) {
        std::istringstream iss(line);
        iss >> row >> col;

        // Adjust for 1-based indexing in MTX format
        row--;
        col--;

        matrix.row_message[row].nzRowCount++;
        matrix.row_message[row].nzValue.insert(std::make_pair(col, 1));
    }

    fin.close();
    return matrix;
}

int main() {
    std::string filename = "../data/weighted/494_bus.mtx";

    MATRICES matrix = readMTXFileWeighted(filename);
    if (matrix.rows) {
        // Print the matrix into MATRICES 
        std::cout << "Rows: " << matrix.rows << std::endl;
        std::cout << "Columns: " << matrix.cols << std::endl;
        std::cout << "Non-zero entries: " << matrix.nnz << std::endl;

        std::cout << "Row Messgae: ";
        for (int i = 0; i < matrix.rows; i++) {
            std::cout << i << std::endl;
            std::cout << matrix.row_message[i] << std::endl;
        }
        std::cout << std::endl;
    }

    return 0;
}