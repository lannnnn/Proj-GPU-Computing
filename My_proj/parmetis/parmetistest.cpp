#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "mpi.h"
#include "parmetis.h"
#include "utilities.h"

int main( int argc, char *argv[] ) {

   int irank,nproc; // always int?

   std::string filename = "../data/weighted/freeFlyingRobot_4.mtx";
   // method allow inordered input data
   COO coo = readMTXFileWeighted(filename);
   // print_matrix(matrix, block_rows); //print matrix message

   // init label list for distance calculation
   std::vector<std::vector<int>> label(coo.rows); //,std::vector<int>(labelSize));
   std::multimap<int, int, std::greater<int>> rankMap;

   CSR csr(coo.rows, coo.cols, coo.nnz);
   // csr to coo, build the rankMap at same time
   cooToCsr(coo, csr, rankMap);

   MPI_Init(&argc, &argv);

   /* Get process rank */
   MPI_Comm_rank(MPI_COMM_WORLD, &irank);

   /* Get number of processes */
   MPI_Comm_size(MPI_COMM_WORLD, &nproc);

   // example from ParMETIS manual (page 12)
   // hard-coded for 3 processor example
   if(nproc!=1) {
      if(irank==0) {
         printf("error: must run with 1 processes\n");
      }
      MPI_Finalize();
      return(0);
   }

   // construct adjacency arrays

   idx_t vtxdist[4] = {0, 5, 10, 15}; // Note that the vtxdist array will always be identical for every processor.

   idx_t *iadj;
   idx_t *jadj;

   if(irank==0) {
      memcpy(iadj, &csr.rowPtr[0], csr.rowPtr.size()*sizeof(int));  
      memcpy(jadj, &csr.colIdx[0], csr.colIdx.size()*sizeof(int)); 
   }

   idx_t *iwgt = NULL;
   idx_t *jwgt = NULL;

   idx_t wgtflag = 0;
   idx_t numflag = 0;
   idx_t ncon = 1;
   idx_t nparts = 4;

   real_t *tpwgts;
   tpwgts = (real_t*) malloc(ncon*nparts*sizeof(real_t));
   for(int i=0;i<ncon*nparts;++i) {
      tpwgts[i] = 1.0/nparts;
   }
   real_t ubvec[1] = {1.05};

   idx_t options[4] = {0, 0, 0, 0}; // iopt[3] = PARMETIS_PSR_UNCOUPLED; // # part != # processors
   idx_t edgecut;
   idx_t part[5];

   MPI_Comm comm = MPI_COMM_WORLD;

   int ier; // always int?

   ier = ParMETIS_V3_PartKway( vtxdist, iadj, jadj, iwgt, jwgt,
                               &wgtflag, &numflag, &ncon, &nparts,
                               tpwgts, ubvec,
                               options, &edgecut, part, &comm );
 
   // gather partition vector, pvec

   idx_t pvec[15];

   if (sizeof(idx_t)==4) {
      ier = MPI_Gather( part, 5, MPI_INT, pvec, 5, MPI_INT, 0, comm );
   }
   else if (sizeof(idx_t)==8) {
      ier = MPI_Gather( part, 5, MPI_LONG, pvec, 5, MPI_LONG, 0, comm );
   }
   else {
      // error
   } 
 
   // print partition vector of the graph
   if(irank==0) {
      printf("\n");
      printf("edgecut = %ld\n", (long) edgecut);
      printf("\n");
      for(int i=0;i<1;++i) {
         for(int j=0;j<5;++j) {
            printf("%ld ", (long) pvec[5*i+j]);
         }
         printf("\n");
      }
      printf("\n");
   }

   MPI_Finalize();

   return(0);
}
