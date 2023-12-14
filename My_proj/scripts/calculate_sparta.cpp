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
#include <sstream>
#include <vector>
#include <ctime>

int main( int argc, char *argv[] ) {
   std::string* str;
   char* endStr;
   int mtx = 0;
   int el = 0;
   int block_cols = 64;

   if(argc < 3) {
      std::cout<<"Usage parameter list: <matrix file> <partition file>"<<std::endl;
      return 0;
   }

   std::string filename;// = "../data/unweighted/cs_department.el";
   endStr = std::find(argv[1], argv[1]+100, '\0');
   (filename).assign(argv[1], endStr);

   COO coo;
   //COO mask_coo ;
   // method allow inordered input data

    coo = readMTXFileUnweighted(filename);
    //mask_coo = readMTXFileMask(filename, block_cols);

   CSR csr(coo.rows, coo.cols, coo.nnz);
   //CSR mask_csr(mask_coo.rows, mask_coo.cols, mask_coo.nnz);
   // csr to coo, build the rankMap at same time
   
   cooToCsr(coo, csr);
   coo.clean();

   float ori_den = csr.calculateBlockDensity(block_cols, block_cols);

   std::cout << "original block density: " << ori_den << std::endl;
   std::cout << "original store density: " << csr.calculateStoreSize(block_cols, block_cols)/(float)csr.rows / (float)csr.cols << std::endl;

   endStr = std::find(argv[2], argv[2]+100, '\0');
   (filename).assign(argv[2], endStr);

   std::ifstream fin;
   //std::cout << "File: " << filename << std::endl;
   fin.open(filename,std::ios_base::in);
   if (!fin.is_open()) {
      std::cout << "Failed to open file: " << filename << std::endl;
      return 0;
   }

   std::string line;
   std::string groupIdx;

   std::vector<std::vector<int>> fine_group(csr.rows);

   int group_idx;
   int idx = 0;

   std::vector<int> dimension;
   int row_cnt=0;

   std::getline(fin, line);
   std::stringstream iss(line);
   std::getline(iss, groupIdx, ','); // move the grouping
   std::getline(iss, groupIdx, ',');

   int group_cnt=-1;

   do {
      //std::istringstream iss(line);
      // iss >> group_idx;
      //std::cout << groupIdx  << std::endl;
      if(row_cnt >= csr.rows) break;
      fine_group[stoi(groupIdx)].push_back(row_cnt);

      while(stoi(groupIdx) > group_cnt) {
         dimension.push_back(0);
         group_cnt++;
      }

      dimension[stoi(groupIdx)] ++ ;

      row_cnt++;
   } while((std::getline(iss, groupIdx, ',')));

   fin.close();

   std::cout << "Finish reading file " << filename << std::endl;

   double real_group = 0;
   double dev = 0;
   // for(int i=0; i<=group_cnt ;i++) {
   //    if(dimension[i] == 0) continue;
   //    //outfile << dimension[i] << std::endl;
   //    real_group ++;
   // }
   double avg = (double) row_cnt / count_group(fine_group);

   for(int i=0; i<=group_cnt ;i++) {
      if(dimension[i] == 0) continue;
      //outfile << dimension[i] << std::endl;
      dev += std::pow(dimension[i] - avg, 2);
   }

   dev = std::pow(dev/(double)(group_cnt+1),0.5);

   CSR new_csr(csr.rows, csr.cols, csr.nnz);
   clock_t start = clock();
   reordering(csr, new_csr, fine_group);
   clock_t end   = clock();
   std::cout << "reorder finished in " << (double)(end - start) / CLOCKS_PER_SEC << "seconds" << std::endl;

   csr.clean();

   float block_density = new_csr.calculateBlockDensity(block_cols, block_cols);
   //std::cout << "yet ok ? " << std::endl;
   float store_density = new_csr.calculateStoreSize(block_cols, block_cols)/(float)new_csr.rows / (float)new_csr.cols;
   float density_dev = new_csr.calculateBlockDensityDev(block_cols, block_cols);
   //std::cout << "whats wrong?" << std::endl;

   std::cout << "new block density: " << block_density << std::endl;
   std::cout << "new store density: " << store_density << std::endl;
   std::cout << "density ratio: " << block_density / ori_den << std::endl;
   std::cout << "new density deviation: " << density_dev << std::endl;
   std::cout << "new dimension avg: " << avg << std::endl;
   std::cout << "new dimension deviation: " << dev << std::endl;
   std::cout << "group number: " << count_group(fine_group) << std::endl;

   return(0);
}
