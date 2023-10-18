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

   std::string filename = "../data/unweighted/cs_department.el";
   if(argc >=2) {
      endStr = std::find(argv[1], argv[1]+100, '\0');
      (filename).assign(argv[1], endStr);
   }
   if(argc >=3) {
      if(atoi(argv[2]) == 1) mtx = 1;
      if(atoi(argv[2]) == 2) el = 1;
   }

   COO coo ;
   // method allow inordered input data
   if(mtx) {
      coo = readMTXFileUnweighted(filename);
   } else {
      coo = readELFileUnweighted(filename);
   }

   int edge = coo.nnz;
   // build edge for graph(if (1,2) then (2,1))
   for(int i=0; i<coo.rows; i++) {
      for(auto it=begin(coo.row_message[i].nzValue);it!=end(coo.row_message[i].nzValue);){
         if(it->first == i) {
            it = coo.row_message[i].nzValue.erase(it);
            coo.nnz--;
            continue;
         }
	      if(coo.row_message[it->first].nzValue[i]) {
            edge --;
         } else {
            coo.row_message[it->first].nzValue[i] = 1;
            coo.nnz++;
         }
         ++it;
      }
   }

   //coo.nnz = edge * 2;

   CSR csr(coo.rows, coo.cols, coo.nnz);
   // csr to coo, build the rankMap at same time
   cooToCsr(coo, csr);

   std::ofstream outfile;
   outfile.open("../data/convert/out.graph");

   outfile << csr.rows << " " << coo.nnz/2 << std::endl;
   for(int i=0; i< csr.rows; i++) {
      // if(csr.rowPtr[i+1] - csr.rowPtr[i] == 0) continue;
      for(int j=csr.rowPtr[i]; j<csr.rowPtr[i+1]; j++) {
         outfile << csr.colIdx[j]+1 << " ";
      }
      outfile  << std::endl;
   }

   outfile.close();

   return(0);
}
