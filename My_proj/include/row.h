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

    friend std::ostream& operator << (std::ostream& os, ROW& row)
	{
		for(auto it : row.nzValue){
            os<<"col: " << it.first<<", value "<<it.second << std::endl;
        }
		return os;
	}

    ROW() : nzRowCount(0), labelSize(0), nzValue() {}

};