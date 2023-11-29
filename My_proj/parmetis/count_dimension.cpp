#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "parmetis.h"
#include "utilities.h"
#include "group.h"
#include <fstream>
#include <iostream>
#include <vector>
#include<algorithm>
#include <sstream>
#include <cmath>

int main( int argc, char *argv[] ) {
    std::string* str;
    char* endStr;
    std::string filename;

    if(argc < 1) {
        std::cout << "please specify the filename" << std::endl;
        return 0;
    }

    endStr = std::find(argv[1], argv[1]+100, '\0');
    (filename).assign(argv[1], endStr);

    COO coo;
    int sync = 0;
    int total = 0;
    int maxLength = 0;
    int curLength = 0;
    int lastIdx = -1;

    coo = readMTXFileUnweighted(filename);

    // CSR csr(coo.rows, coo.cols, coo.nnz);
    // //CSR mask_csr(mask_coo.rows, mask_coo.cols, mask_coo.nnz);
    // // csr to coo, build the rankMap at same time
   
    // cooToCsr(coo, csr);
    // coo.clean();

    std::cout << "matrix built, rows = " << coo.rows << ", cols = " << coo.cols  << std::endl;

    for(int i=0; i<coo.rows; i++) {
        //if(i%1000 == 0) std::cout << "it->first " << i << std::endl;
        for(auto it=begin(coo.row_message[i].nzValue);it!=end(coo.row_message[i].nzValue);++it){
            //std::cout << "it->first = " << it->first << std::endl;
            if(it->first < coo.rows && coo.row_message[it->first].nzValue[i]) {
                sync ++;
            }
            if(lastIdx == it->first-1) {
                curLength += 1;
            } else {
                curLength = 1;
            }
            if(curLength > maxLength) maxLength = curLength;
            lastIdx = it->first;
        }
        lastIdx = -1;
    }

    

    //outfile.close();

    std::cout << "symmetric ration (symmetric/total) : " << (double)sync / (double)coo.nnz << std::endl;
    std::cout << "max consecutive nz: " << maxLength << std::endl;

    return(0);
}
