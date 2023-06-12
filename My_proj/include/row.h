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
    int labelSize;
    int rank;
    int rowIdx = 0;
    std::map<int, double> nzValue; //ranked by column index in default

    void calculate_label(int row_block_size, int ncols, std::vector<int> &label) {
        rank = 0;
        labelSize = (ncols-1) / row_block_size + 1;
        int next_edge = 1; 
        int current_label = 0;

        //label.reserve(labelSize);

        for(auto it : nzValue){
            if(it.first > ncols) {
                std::cout << "ERROR: CLOUMN OUT OF RANGE" << std::endl;
                break;
            }
	        if(it.first >= next_edge * row_block_size) {
                rank += current_label;
                label[next_edge-1] = current_label>0?1:0;     // the last block is all counted
                next_edge ++;
                current_label = 0;                  // init for the new block
                while(it.first >= next_edge * row_block_size) {       // if it's not in the next block, 
                    next_edge ++;    // goto the one after next
                    // rank += current_label;
                    // label[next_edge-1] = current_label;  // and fill the next label as 0
                }
            }
            current_label++;
        }

        rank += current_label;
        label[next_edge-1] = current_label>0?1:0; 
        next_edge ++;
        current_label = 0;

        // fill the rest as 0
        // while(next_edge < labelSize) {
        //     next_edge ++;                // goto the one after next
        //     label.push_back(current_label);             // and fill the next label as 0
        // }
    }

    friend std::ostream& operator << (std::ostream& os, ROW& row)
	{
		for(auto it : row.nzValue){
            os<<"col: " << it.first<<", value "<<it.second << std::endl;
        }
		return os;
	}

    ROW() : nzRowCount(0), labelSize(0), nzValue() {}

};