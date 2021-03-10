//
// Created by 李亘杰 on 2021/2/2.
//

#include "newNestedHelloWorld.cuh"
#include "../common.h"
#include <stdio.h>
#include "../include/cuda_runtime.h"

__global__ void nestedHelloWorld(int const iSize, int minSize, int iDepth)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Recursion=%d: Hello World from thread %d ,bloclId.x %d  blockDim.x  %d  gridDim.x %d  iSize %d   minSize  %d\n", iDepth, tid, blockIdx.x,blockDim.x,gridDim.x,iSize,minSize);

    // condition to stop recursive execution
    if (iSize == minSize) return;

    // reduce nthreads by half
    int nthreads = iSize >> 1;

    // thread 0 launches child grid recursively
    if(tid == 0 && nthreads > 0)
    {
        int blocks = (nthreads + blockDim.x - 1) / blockDim.x;
        nestedHelloWorld<<<blocks, blockDim.x>>>(nthreads, minSize, ++iDepth);
        printf("-------> nested execution depth: %d\n", iDepth);
    }
}

int main8(int argc, char **argv)
{
    int igrid = 2;
    int blocksize = 8;

    if(argc > 1)
    {
        igrid = atoi(argv[1]);
    }

    if (argc > 2)
    {
        blocksize = atoi(argv[2]);
    }

    int size = igrid * blocksize;

    dim3 block (blocksize, 1);
    dim3 grid  ((size + block.x - 1) / block.x, 1);
    printf("size = %d\n", size);
    printf("igrid = %d\n", igrid);
    printf("%s Execution Configuration: grid %d block %d\n", argv[0], grid.x,
           block.x);

    nestedHelloWorld<<<grid, block>>>(size, grid.x, 0);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    return 0;
}
