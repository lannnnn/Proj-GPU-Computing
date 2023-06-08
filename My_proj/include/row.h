#pragma once

#include <string>
#include <vector>
#include <map>
#include <bits/stdc++.h>

struct ROW
{
    /*--------------------------------------------------------------
    | Row message recording
    |==========
    |       ncol = # of columns
    |       nzRowCount  = # of nz values in this row
    |       nzValue  = exact value of the nv element with column message
    |       label  = label according to the block message
    |--------------------------------------------------------------------   */
    int nzRowCount;
    std::map<int, double> nzValue; //ranked by column index in default
    std::vector<int> label;

    void calculate_label(int row_block_size, int ncols) {
        int label_size = (ncols-1) / row_block_size + 1;
        int next_edge = row_block_size; 
        int current_label = 0;

        label.reserve(label_size);

        for(auto it : nzValue){
            if(it.first > ncols) {
                std::cout << "ERROR: CLOUMN OUT OF RANGE" << std::endl;
                break;
            }
	        if(it.first > next_edge) {
                label.push_back(current_label);     // the last block is all counted
                current_label = 0;                  // init for the new block
                while(it.first > next_edge) {       // if it's not in the next block, 
                    next_edge += row_block_size;    // goto the one after next
                    label.push_back(current_label); // and fill the next label as 0
                }
            }
            current_label++;
        }
    }

    friend std::ostream& operator << (std::ostream& os, ROW& row)
	{
		for(auto it : row.nzValue){
            os<<"col: " << it.first<<", value "<<it.second << std::endl;
        }
		return os;
	}

    ROW() : nzRowCount(0), nzValue(), label() {}

};