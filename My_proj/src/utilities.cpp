#include "utilities.h"

COO readMTXFileWeighted(const std::string& filename) {
    std::ifstream fin;
    fin.open(filename,std::ios_base::in);
    if (!fin.is_open()) {
        std::cout << "Failed to open file: " << filename << std::endl;
        return COO();
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

    COO coo(rows, cols, nnz);
    while (std::getline(fin, line)) {
        std::istringstream iss(line);
        iss >> row >> col >> value;

        // Adjust for 1-based indexing in MTX format
        row--;
        col--;

        coo.row_message[row].nzRowCount++;
        coo.row_message[row].nzValue[col] = value;
    }

    fin.close();
    return coo;
}

COO readMTXFileUnweighted(const std::string& filename) {
    std::ifstream fin;
    fin.open(filename,std::ios_base::in);
    if (!fin.is_open()) {
        std::cout << "Failed to open file: " << filename << std::endl;
        return COO();
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

    COO coo(rows, cols, nnz);
    while (std::getline(fin, line)) {
        std::istringstream iss(line);
        iss >> row >> col;

        // Adjust for 1-based indexing in MTX format
        row--;
        col--;

        coo.row_message[row].nzRowCount++;
        coo.row_message[row].nzValue[col] = 1;
    }

    fin.close();
    return coo;
}

void print_pointer(int* p, int size) {
    for(int i=0; i<size; i++) {
        std::cout << p[i] << " ";
    }
    std::cout << std::endl;
}

void print_matrix(COO coo, int block_rows) {
    if (coo.rows) {
        // Print the matrix into MATRICES 
        std::cout << "Rows: " << coo.rows << std::endl;
        std::cout << "Columns: " << coo.cols << std::endl;
        std::cout << "Non-zero entries: " << coo.nnz << std::endl;

        std::cout << "Row Messgae: ";
        for (int i = 0; i < coo.rows; i++) {
            // matrix.row_message[i].calculate_label(block_rows, matrix.cols);
            std::cout << i << std::endl;
            std::cout << coo.row_message[i] << std::endl;
        }
        std::cout << std::endl;
    }
}

void print_vec(std::vector<std::vector<int>> label) {
    for(int i=0; i<label.size(); i++) {
        for(int j = 0; j< label[i].size(); j++) {
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

int cooToCsr(COO &coo, CSR &csr, std::multimap<int, int, std::greater<int>> &rankMap) {
    if (coo.rows != 0) {
        csr.rowPtr[0] = 0;
        for (int i = 0; i < coo.rows; i++) {
            coo.row_message[i].rowIdx = i;   // fill the row number
            csr.addRow(coo.row_message[i]);
            if(coo.row_message[i].nzValue.size() != 0) // we do not need the rows which do not contain the nz values
                rankMap.insert(std::make_pair(coo.row_message[i].nzValue.size(), i));
        }
    }
};