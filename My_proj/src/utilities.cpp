#include "utilities.h"

void helpMesg() {
    std::cout<<"Usage parameter list:"<<std::endl;
    std::cout<<"    -f file direct"<<std::endl;
    std::cout<<"    -t tau"<<std::endl;
    std::cout<<"    -b block size"<<std::endl;
    std::cout<<"    -p SET TO PRINT GROUP INFO"<<std::endl;
    std::cout<<"    -m or -e SET MATRIX FILE FORMAT"<<std::endl;
    std::cout<<"    -l to iterate tau to find the best block density"<<std::endl;
}

void readConfig(int argc, char** argv, std::string* filename, int* block_cols, float* fine_tau, int* print, int* mtx, int* el, int* list) {
    std::string* str;
    char* endStr;
    for(int i = 1;i < argc; ++i){
        switch(argv[i][1]){
            case 'f':
                if(argc <= (i+1)) {
                    std::cout<<"no file name found, please check input...."<<std::endl;
                    helpMesg();
                    break;
                }
                endStr = std::find(argv[i+1], argv[i+1]+100, '\0');
                (*filename).assign(argv[i+1], endStr);
                i++;
                break;
            case 't':
                if(argc <= (i+1)) {
                    std::cout<<"no tau value found, please check input...."<<std::endl;
                    helpMesg();
                    break;
                }
                *fine_tau = atof(argv[i+1]) ;
                i++;
                break;
            case 'b':
                if(argc <= (i+1)) {
                    std::cout<<"no block size found, please check input...."<<std::endl;
                    helpMesg();
                    break;
                }
                *block_cols = atoi(argv[i+1]) ;
                i++;
                break;
            case 'p':
                std::cout<<"case \'-p\' found, set to print group info"<<std::endl;
                *print = 1;
                break;
            case 'm':
                std::cout<<"case \'-m\' found, set to mtx format"<<std::endl;
                *mtx = 1;
                break;
            case 'e':
                std::cout<<"case \'-e\' found, set to el format"<<std::endl;
                *el = 1;
                break;
            case 'l':
                std::cout<<"case \'-l\' found, iterating to find tau"<<std::endl;
                std::cout<<"****Notic: list func can not be use together with specific tau****"<<std::endl;
                *list = 1;
                break;
            default:
                std::cout<<"unrecognized input, pls check the help info"<<std::endl;
                helpMesg();
                break;
        }
    }
}

COO readELFileWeighted(const std::string& filename) {
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

    // initial the csr struct without size message
    COO coo(0, 0, 0);
    int current_row = -1;
    do {
        std::istringstream iss(line);
        iss >> row >> col >> value;
        // std::cout << line << std::endl;
        // std::cout << "current row = " << current_row << ", row = " << row << std::endl;
        if(row < current_row) {
            std::cout << "Please enter the row with ascending order...." << std::endl;
            return COO();
        }
        for(int i=current_row; i<row; i++) {
            coo.row_message.push_back(ROW());
            current_row ++;
        }

        coo.row_message[row].nzRowCount++;
        coo.row_message[row].nzValue[col] = value;
        coo.nnz++;
        if(col > coo.cols) coo.cols = col;
    } while((std::getline(fin, line)));
    coo.rows = current_row+1;
    coo.cols++;
    fin.close();
    return coo;
}

COO readELFileUnweighted(const std::string& filename) {
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

    // initial the csr struct without size message
    COO coo(0, 0, 0);
    int current_row = -1;
    do {
        std::istringstream iss(line);
        iss >> row >> col;
        // std::cout << line << std::endl;
        // std::cout << "current row = " << current_row << ", row = " << row << std::endl;
        // if(row < current_row) {
        //     std::cout << "Please enter the row with ascending order...." << std::endl;
        //     return COO();
        // }
        for(int i=current_row; i<row; i++) {
            coo.row_message.push_back(ROW());
            current_row ++;
        }

        coo.row_message[row].nzRowCount++;
        coo.row_message[row].nzValue[col] = 1;
        coo.nnz++;
        if(col > coo.cols) coo.cols = col;
    } while((std::getline(fin, line)));
    coo.rows = current_row+1;
    coo.cols++;
    fin.close();

    if(coo.cols > coo.rows) {
        for(int i=coo.rows; i<=coo.cols; i++) {
            coo.row_message.push_back(ROW());
        }
    }
    coo.rows = coo.cols;

    return coo;
}

COO readELFileMask(const std::string& filename, int block_col) {
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

    // initial the csr struct without size message
    COO coo(0, 0, 0);
    int current_row = 0;
    do {
        std::istringstream iss(line);
        iss >> row >> col;
        // std::cout << line << std::endl;
        // std::cout << "current row = " << current_row << ", row = " << row << std::endl;
        if(row < current_row) {
            std::cout << "Please enter the row with ascending order...." << std::endl;
            return COO();
        }
        for(int i=current_row; i<=row; i++) {
            coo.row_message.push_back(ROW());
            current_row ++;
        }
        current_row --;
        col = col/block_col;
        if(coo.row_message[row].nzValue[col] != 1) {
            coo.row_message[row].nzRowCount++;
            coo.row_message[row].nzValue[col] = 1;
            coo.nnz++;
        }
        if(col > coo.cols) coo.cols = col;
    } while((std::getline(fin, line)));
    coo.rows = current_row+2;
    coo.cols++;
    fin.close();
    return coo;
}

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
        // std::cout << line << std::endl;
        std::istringstream iss(line);
        iss >> row >> col >> value;

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

COO readMTXFileMask(const std::string& filename, int block_col) {
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
    iss >> rows >> cols;

    COO coo(rows, cols, nnz);
    while (std::getline(fin, line)) {
        std::istringstream iss(line);
        iss >> row >> col;

        // Adjust for 1-based indexing in MTX format
        row--;
        col--;

        col = col/block_col;
        if(coo.row_message[row].nzValue[col] != 1) {
            coo.row_message[row].nzRowCount++;
            coo.row_message[row].nzValue[col] = 1;
            nnz ++;
        }
    }
    coo.nnz = nnz;

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
        if(label[i].size() > 0)  {
            std::cout << std::endl;
        }
    }
}

int count_group(std::vector<std::vector<int>> label) {
    int groupSize = 0;
    for(int i=0; i<label.size(); i++) {
        if(label[i].size() > 0)  {
            groupSize++;
        }
    }
    return groupSize;
}

void print_map(std::multimap<int, int, std::greater<int>> rankMap) {
    std::multimap<int, int>::iterator itr;
    for (itr = rankMap.begin(); itr != rankMap.end(); ++itr) {
        std::cout << '\t' << itr->first << '\t' << itr->second
             << '\n';
    }
}

int cooToCsr(COO &coo, CSR &csr) {
    if (coo.rows != 0) {
        csr.rowPtr[0] = 0;
        for (int i = 0; i < coo.rows; i++) {
            coo.row_message[i].rowIdx = i;   // fill the row number
            csr.addRow(coo.row_message[i]);
            // if(coo.row_message[i].nzValue.size() != 0) // we do not need the rows which do not contain the nz values
            //     rankMap.insert(std::make_pair(coo.row_message[i].nzValue.size(), i));
        }
    }
};

int cooToCsrRank(COO &coo, CSR &csr, std::multimap<int, int, std::greater<int>> &rankMap) {
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

void printRes(std::vector<std::vector<int>> label, std::vector<int> res_vec, const std::string& filename) {
    std::ofstream outfile;
    outfile.open(filename);
    int groupCnt = 0;

    for(int i=0; i<label.size(); i++) {
        for(int j = 0; j< label[i].size(); j++) {
            res_vec[label[i][j]] = groupCnt;
        }
        if(label[i].size() > 0) groupCnt++;
    }

    for(int i=0; i< res_vec.size(); i++) {
        outfile << res_vec[i] << std::endl;
    }

    outfile.close();

    return;
};