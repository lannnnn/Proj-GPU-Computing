#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include "include/matrices.h"
#include "include/group.h"

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
        matrix.row_message[row].nzValue[col] = value;
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
        matrix.row_message[row].nzValue[col] = 1;
    }

    fin.close();
    return matrix;
}

void print_matrix(MATRICES matrix, int block_rows) {
    if (matrix.rows) {
        // Print the matrix into MATRICES 
        std::cout << "Rows: " << matrix.rows << std::endl;
        std::cout << "Columns: " << matrix.cols << std::endl;
        std::cout << "Non-zero entries: " << matrix.nnz << std::endl;

        std::cout << "Row Messgae: ";
        for (int i = 0; i < matrix.rows; i++) {
            // matrix.row_message[i].calculate_label(block_rows, matrix.cols);
            std::cout << i << std::endl;
            std::cout << matrix.row_message[i] << std::endl;
        }
        std::cout << std::endl;
    }
}

void print_label(std::vector<std::vector<int>> label, int rows, int labelSize) {
    for(int i=0; i<rows; i++) {
        for(int j=0; j<labelSize; j++) {
            std::cout << label[i][j] << " ";
        }
            std::cout << std::endl;
    }
}

void print_map(std::multimap<int, int, std::greater<int>> rankMap) {
    std::multimap<int, int>::iterator itr;
    for (itr = rankMap.begin(); itr != rankMap.end(); ++itr) {
        std::cout << '\t' << itr->first << '\t' << itr->second
             << '\n';
    }
}

int main() {
    std::string filename = "../data/weighted/494_bus.mtx";
    int label_cols = 64;
    int block_rows = 64;
    int group_number = 4;   // should have better performance if same with thread number
    
    // method allow inordered input data
    MATRICES matrix = readMTXFileWeighted(filename);
    // print_matrix(matrix, block_rows); //print matrix message

    int labelSize = (matrix.cols-1) / label_cols + 1;

    // init label list for distance calculation
    std::vector<std::vector<int>> label(matrix.rows,std::vector<int>(labelSize));
    std::multimap<int, int, std::greater<int>> rankMap;

    CSR csr(matrix.rows, matrix.cols, matrix.nnz);
    // calculate the label
    if (matrix.rows != 0) {
        for (int i = 0; i < matrix.rows; i++) {
            matrix.row_message[i].rowIdx = i;   // fill the row number
            matrix.row_message[i].calculate_label( label_cols, matrix.cols, label[i]);
            csr.addRow(matrix.row_message[i]);
            rankMap.insert(std::make_pair(matrix.row_message[i].rank, i));
        }
    }
    // free the matrix, use csr
    delete matrix;
    // print_label(label, matrix.rows, labelSize);
    // csr.print();
    // print_map(rankMap);

    // init coarse graind group vector
    // we have a little larger buffer to save in case there be one group have more elements than others, but not too much..
    int coarse_group_rows = csr.rows/group_number*1.5;
    std::vector<std::vector<int>> coarse_group(group_number,std::vector<int>(coarse_group_rows));
    // init fine graind group vector
    // give little more space for blocks in each thread which can not be totally filled
    int fine_group_num = (csr.rows-1)/block_rows + group_number;
    std::vector<std::vector<int>> fine_group(fine_group_num,std::vector<int>(block_rows));

    // init the group by rank
    /*  reason: many dynamic real-world graphs,
        such as social networks, follow a skewed distribution of vertex
        degrees, where there are a few high-degree vertices and many
        low-degree vertices. */
    if(group_number <= csr.rows) {
        std::multimap<int, int>::iterator itr = rankMap.begin();
        for (int cnt=0; cnt<group_number; cnt++) { 
            coarse_group[cnt][0] = itr->second;
            rankMap.erase(itr++); 
        }
    }

    // adding a group recorder? or just record by add -1?
    // if(!coarse_grouping(coarse_group, matrix, rankMap, group_number, coarse_group_rows)) {
    //     std::cout << "Failed in coarse graind grouping... " << std::endl;
    //     return -1;
    // }

    // if(!fine_grouping(coarse_group, matrix, fine_group, group_number, coarse_group_rows)) {
    //     std::cout << "Failed in fine graind grouping... " << std::endl;
    //     return -1;
    // }

    return 0;
}