#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

    std::ifstream fin;
    fin.open(filename,std::ios_base::in);
    if (!fin.is_open()) {
        std::cout << "Failed to open file: " << filename << std::endl;
        return 0;
    }

    std::string line;

    // Skip header lines
    while (std::getline(fin, line) && line[0] == '%') {
        // Skip comment lines
    }

    std::vector<int> dimension;
    int group_idx;
    int group_cnt=-1;
    int row_cnt=0;

    do {
        std::istringstream iss(line);
        iss >> group_idx;
        row_cnt ++;
        // std::cout << line << std::endl;
        // std::cout << "current row = " << current_row << ", row = " << row << std::endl;
        while(group_idx > group_cnt) {
            dimension.push_back(0);
            group_cnt++;
        }
        dimension[group_idx] ++ ;
    } while((std::getline(fin, line)));

    fin.close();

    std::string ofilename = filename + ".dim";
    std::ofstream outfile;
    double avg = (double) row_cnt / (double) (group_cnt+1);
    double dev = 0;
    //outfile.open(ofilename);

    for(int i=0; i<=group_cnt ;i++) {
        if(dimension[i] == 0) continue;
        //outfile << dimension[i] << std::endl;
        dev += std::pow(dimension[i] - avg, 2);
    }

    dev = std::pow(dev/(double)(group_cnt+1),0.5);

    //outfile.close();

    std::cout << "mean : " << avg << std::endl;
    std::cout << "standard deviation : " << dev << std::endl;

    return(0);
}
