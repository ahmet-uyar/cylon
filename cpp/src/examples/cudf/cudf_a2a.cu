//
// Created by auyar on 3.02.2021.
//
#include "cudf_a2a.cuh"

__global__ void rebaseOffsets(int32_t * arr, int size, int32_t base) {
    int i = threadIdx.x;
    if (i < size) {
        arr[i] -= base;
    }
}

void callRebaseOffsets(int32_t * arr, int size, int32_t base){
    rebaseOffsets<<<1, size>>>(arr, size, base);
}

