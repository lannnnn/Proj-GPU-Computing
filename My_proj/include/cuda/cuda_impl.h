#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include "../csr.h"

#define rows_per_thread 4;

struct GroupRow
{
	thrust::device_vector<int> rowIdx;
    thrust::device_vector<int> lable;

    ~GroupRow() {
        thrust::device_delete(rowIdx);
        thrust::device_delete(lable);
    }
};