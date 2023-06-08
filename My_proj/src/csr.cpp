#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <random> //std::shuffle
#include <stdexcept>

#include <algorithm>    // std::sort
#include <numeric> //std::iota
#include <iterator>
#include "csr.h"

using namespace std;

void CSR::clean()
{
    /*----------------------------------------------------------------------
    | Free up memory allocated for CSR structs.
    |--------------------------------------------------------------------*/

    if (rows + cols <= 1) return;

    if (rowPtr) 
    {
        delete[] rowPtr;
    }

    if (colIdx) 
    {
        delete[] colIdx;
    }
    
    rows = 0;
    cols = 0;
    nnz = 0 
}


void CSR::multiply(DataT* mat_B, intT B_cols, DataT_C* mat_C)
{
    //A*B, A is CSR, B and C column-wise storage
}

void CSR::permute_rows(vector<intT> permutation)
//permute the rows of the matrix according to "permutation"
{   

}

void CSR::permute_cols(vector<intT> permutation)
//permute the rows of the matrix according to "permutation"
{   

}

void CSR::reorder(vector<intT> grouping)
//permute the rows so that row i and row j are adjacent if grouping[i] == grouping[j]
{

}

void CSR::reorder_by_degree(bool descending)
{

}

void CSR::scramble()
{
    //randomly permute rows (TODO better randomness)
}


void CSR::save_to_edgelist(std::ofstream& outfile, std::string delimiter, bool pattern_only, MatrixFormat mat_fmt)
{

}


void CSR::read_from_edgelist(ifstream& infile, string delimiter, bool pattern_only, MatrixFormat mat_fmt, bool symmetrize)
//reads edgelist into the CSR.
{

}

void CSR::print(intT verbose)
{
    cout << "PRINTING A CSR MATRIX (arrays only)" << endl;
    cout << "ROWS: " << rows << " COLS: " << cols << " PATTERN_only: " << pattern_only << endl; 
    cout << "NZ: " << nztot() << endl;

    if (verbose > 1)
    {
        cout << "JA:" << endl;
        for (intT i = 0; i < rows; i++)
        {
            cout << "-- ";
            for (intT j = 0; j < nzcount[i]; j++)
            {
                cout << ja[i][j] << " ";
            }
            cout << endl;

        }

        if (!pattern_only)
        {        
            cout << "MA:" << endl;
            for (intT i = 0; i < rows; i++)
            {
                cout << "-- ";
                for (intT j = 0; j < nzcount[i]; j++)
                {
                    cout << ma[i][j] << " ";
                }
                cout << endl;
            }
        }

        cout << "NZCOUNT:" << endl;
        for (intT i = 0; i < rows; i++)
        {
                cout << nzcount[i] << " ";
        }
        cout << endl;
    }

    if (verbose > 0)
        {
            //loop through rows
            for (intT i = 0; i < rows; i++)
            {
                intT j = 0;
                for (intT nzs = 0; nzs < nzcount[i]; nzs++) 
                {
                    intT nz_column = ja[i][nzs]; //find column (row) index of next nonzero element
                    
                    DataT elem;
                    if (!pattern_only) elem = ma[i][nzs]; //value of that element;
                    else elem = 1;
                
                    while (j < nz_column)
                    {
                        j++;
                        cout << 0 << " ";
                    }
                    cout << elem << " ";
                    j++;
                }
                while (j < cols)
                {
                    j++;
                    cout << 0 << " ";
                }

            cout << endl;
            }
        }

}
