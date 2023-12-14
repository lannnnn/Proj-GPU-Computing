#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "mpi.h"
#include <GKlib.h>
#include "parmetis.h"
#include "utilities.h"
#include "group.h"
#include <fstream>
#include <iostream>

int main( int argc, char *argv[] ) {
    std::string* str;
    char* endStr;
    int mtx = 0;
    int el = 0;

    std::string filename; // = "../data/unweighted/cs_department.el";
    if(argc <2) {
        std::cout << "Please specify the dataset..." << std::endl;
    }
    endStr = std::find(argv[1], argv[1]+100, '\0');
    (filename).assign(argv[1], endStr);

    COO coo ;
    coo = readMTXFileUnweighted(filename);

    std::ofstream outfile;
    outfile.open("./out.el");

    for(int i=0; i<coo.rows; i++) {
        //if(i%1000 == 0) std::cout << "it->first " << i << std::endl;
        for(auto it=begin(coo.row_message[i].nzValue);it!=end(coo.row_message[i].nzValue);++it){
            outfile << i+1 << " " << it->first+1 << std::endl;
        }
    }

    outfile.close();

    return(0);
}
