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
#include <vector>

int main( int argc, char *argv[] ) {
   std::string* str;
   char* endStr;
   int mtx = 0;
   int el = 0;
   int block_cols = 64;

   if(argc < 4) {
      std::cout<<"Usage parameter list: <matrix file> <format(1,2)> <partition file>"<<std::endl;
      return 0;
   }

   std::string filename;// = "../data/unweighted/cs_department.el";
   endStr = std::find(argv[1], argv[1]+100, '\0');
   (filename).assign(argv[1], endStr);
   if(atoi(argv[2]) == 1) mtx = 1;
   if(atoi(argv[2]) == 2) el = 1;

   COO coo;
   //COO mask_coo ;
   // method allow inordered input data

   if(mtx) {
      coo = readMTXFileUnweighted(filename);
      //mask_coo = readMTXFileMask(filename, block_cols);
   } else {
      coo = readELFileUnweighted(filename);
      //mask_coo = readELFileMask(filename, block_cols);
   }

   CSR csr(coo.rows, coo.cols, coo.nnz);
   //CSR mask_csr(mask_coo.rows, mask_coo.cols, mask_coo.nnz);
   // csr to coo, build the rankMap at same time
   
   cooToCsr(coo, csr);
   coo.clean();

   std::cout << "original block density: " << csr.calculateBlockDensity(block_cols, block_cols) << std::endl;
   std::cout << "original store density: " << csr.calculateStoreSize(block_cols, block_cols)/(float)csr.rows / (float)csr.cols << std::endl;

   endStr = std::find(argv[3], argv[3]+100, '\0');
   (filename).assign(argv[3], endStr);

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

   std::vector<std::vector<int>> fine_group(csr.rows);

   int group_idx;
   int idx = 0;

   std::vector<int> dimension;
   int group_cnt=-1;

   do {
      std::istringstream iss(line);
      iss >> group_idx;
      // std::cout << line << std::endl;
      // std::cout << "current row = " << current_row << ", row = " << row << std::endl;
      fine_group[group_idx].push_back(idx);
      while(group_idx > group_cnt) {
         dimension.push_back(0);
         group_cnt++;
      }
      dimension[group_idx] ++ ;
      idx++;
      if(idx >= csr.rows) break;
      
   } while((std::getline(fin, line)));

   fin.close();

   double avg = (double) idx / (double) (group_cnt+1);
   double dev = 0;
   for(int i=0; i<=group_cnt ;i++) {
      if(dimension[i] == 0) continue;
      //outfile << dimension[i] << std::endl;
      dev += std::pow(dimension[i] - avg, 2);
   }

   dev = std::pow(dev/(double)(group_cnt+1),0.5);

   CSR new_csr(csr.rows, csr.cols, csr.nnz);
   //std::cout << "reordering" << std::endl;
   reordering(csr, new_csr, fine_group);
   //std::cout << "reorder finished" << std::endl;

   csr.clean();

   float block_density = new_csr.calculateBlockDensity(block_cols, block_cols);
   //std::cout << "yet ok ? " << std::endl;
   float store_density = new_csr.calculateStoreSize(block_cols, block_cols)/(float)new_csr.rows / (float)new_csr.cols;
   float density_dev = new_csr.calculateBlockDensityDev(block_cols, block_cols);
   //std::cout << "whats wrong?" << std::endl;

   std::cout << "new block density: " << block_density << std::endl;
   std::cout << "new store density: " << store_density << std::endl;
   std::cout << "new density deviation: " << density_dev << std::endl;
   std::cout << "new dimension avg: " << avg << std::endl;
   std::cout << "new dimension deviation: " << dev << std::endl;
   std::cout << "group number: " << count_group(fine_group) << std::endl;

   return(0);
}
