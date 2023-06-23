#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include "csr.h"
#include "coo.h"
#include "group.h"

COO readMTXFileWeighted(const std::string& filename);
COO readMTXFileUnweighted(const std::string& filename);

int cooToCsr(COO &coo, CSR &csr, std::multimap<int, int, std::greater<int>> &rankMap);

void print_matrix(COO coo, int block_rows);
void print_pointer(int* p, int size);
void print_vec(std::vector<std::vector<int>> label);
void print_map(std::multimap<int, int, std::greater<int>> rankMap);