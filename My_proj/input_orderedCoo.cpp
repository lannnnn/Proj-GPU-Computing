#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include "include/matrices.h"
#include "include/group.h"

CSR readMTXFileWeighted(const std::string& filename) {
    std::ifstream fin;
    fin.open(filename,std::ios_base::in);
    if (!fin.is_open()) {
        std::cout << "Failed to open file: " << filename << std::endl;
        return CSR();
    }

    std::string line;
    int rows = 0;
    int cols = 0;
    int nnz = 0;

    int row, col;
    int currentRow = 0;
    int currentCol = 0;
    double value;

    // Skip header lines
    while (std::getline(fin, line) && line[0] == '%') {
        // Skip comment lines
    }

    // Read matrix size and nnz
    std::istringstream iss(line);
    iss >> rows >> cols >> nnz;

    CSR matrix(rows, cols, nnz);
    matrix.rowPtr[0] = 0;

    while (std::getline(fin, line)) {
        std::istringstream iss(line);
        iss >> row >> col >> value;

        // Adjust for 1-based indexing in MTX format
        row--;
        col--;

        if(row >= rows || col >= cols) std::cout << "Row/col index out of range" << std::endl;

        if(row < currentRow) {
            std::cout << "Please make sure the mtx file is ordered as ascending" << std::endl;
            return CSR();
        }else if(row == currentRow) {
            if(col <= currentCol) {
                std::cout << "Please make sure the mtx file is ordered as ascending" << std::endl;
                return CSR();
            }
            matrix.rowPtr[currentRow+1]+=1; 
            matrix.colIdx.push_back(value);
        }else if(row > currentRow) {
            while(currentRow < row) {
                currentRow ++;
                matrix.rowPtr[currentCol+1] = matrix.rowPtr[currentCol];
            }
            matrix.rowPtr[currentRow+1]+=1; 
            matrix.colIdx.push_back(value);
        }
    }

    fin.close();
    return matrix;
}

CSR readMTXFileUnweighted(const std::string& filename) {
    return CSR();
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

    CSR matrix = readMTXFileWeighted(filename);
    matrix.print();

    // int labelSize = (matrix.cols-1) / block_cols + 1;

    // // init label list for distance calculation
    // std::vector<std::vector<int>> label(matrix.rows,std::vector<int>(labelSize));
    // std::multimap<int, int, std::greater<int>> rankMap;

    // // calculate the label
    // if (matrix.rows != 0) {
    //     for (int i = 0; i < matrix.rows; i++) {
    //         matrix.row_message[i].rowIdx = i;   // fill the row number
    //         matrix.row_message[i].calculate_label( block_cols, matrix.cols, label[i]);
    //         rankMap.insert(std::make_pair(matrix.row_message[i].rank, i));
    //     }
    // }
    // // print_label(label, matrix.rows, labelSize);

    // // init coarse graind group vector
    // // we have a little larger buffer to save in case there be one group have more elements than others, but not too much..
    // int coarse_group_rows = matrix.rows/group_number*1.5;
    // std::vector<std::vector<int>> coarse_group(group_number,std::vector<int>(coarse_group_rows));
    // // init fine graind group vector
    // // give little more space for blocks in each thread which can not be totally filled
    // int fine_group_num = (matrix.rows-1)/block_rows + group_number;
    // std::vector<std::vector<int>> fine_group(fine_group_num,std::vector<int>(block_rows));
    // // print_map(rankMap);

    // // init the group by rank
    // /*  reason: many dynamic real-world graphs,
    //     such as social networks, follow a skewed distribution of vertex
    //     degrees, where there are a few high-degree vertices and many
    //     low-degree vertices. */
    // if(group_number <= matrix.rows) {
    //     std::multimap<int, int>::iterator itr = rankMap.begin();
    //     for (int cnt=0; cnt<group_number; cnt++) { 
    //         coarse_group[cnt][0] = itr->second;
    //         rankMap.erase(itr++); 
    //     }
    // }

    return 0;
}