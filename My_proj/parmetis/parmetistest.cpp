#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "mpi.h"
#include "parmetis.h"
#include "utilities.h"
#include "group.h"

int main( int argc, char *argv[] ) {

   int irank,nproc; // always int?

   std::string filename = "./data/unweighted/cs_department.el";
   // method allow inordered input data
   COO coo = readELFileWeighted(filename);
   // print_matrix(matrix, block_rows); //print matrix message

   // init label list for distance calculation
   std::vector<std::vector<int>> label(coo.rows); //,std::vector<int>(labelSize));
   std::multimap<int, int, std::greater<int>> rankMap;

   CSR csr(coo.rows, coo.cols, coo.nnz);
   // csr to coo, build the rankMap at same time
   cooToCsr(coo, csr);

   // construct adjacency array

   std::cout << "Try to test with metis ............." << std::endl;

   idx_t nVertices = csr.rows;
   idx_t nWeights = 1;
   idx_t nParts = 2;
   idx_t objval;
   std::vector<idx_t> part(nVertices, 0);

   int ret = METIS_PartGraphKway(&nVertices, &nWeights, csr.rowPtr.data(), csr.colIdx.data(),
                                NULL, NULL, NULL, &nParts, NULL, NULL, NULL,
                                &objval, part.data());

   std::cout << ret << std::endl;

   CSR metis_csr(csr.rows, csr.cols, csr.nnz);
   std::vector<std::vector<int>> mesh_group(part.size());
   for(int i=0; i<part.size(); i++) {
        mesh_group[part[i]].push_back(i);
   }

   print_vec(mesh_group);
   //reordering(csr, metis_csr, mesh_group);

   return(0);
}
