//
// Created by 李亘杰 on 2021/1/27.
//


#include "include/cuda_runtime.h"
#include "include/device_launch_parameters.h"
#include "sys/time.h"
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "include/cuda_profiler_api.h"
#include "common.h"


__global__ void nestedHelloWorld(int const iSize, int iDepth, int maxDepth)
{
    int tid = threadIdx.x;
    printf("Recursion=%d: Hello World from thread %d block %d\n", iDepth, tid,
           blockIdx.x);

    // condition to stop recursive execution
    if (iSize == 1 || iDepth >= maxDepth) return;

    // reduce block size to half
    int nthreads = iSize << 1;

    // thread 0 launches child grid recursively
    if(tid == 0 && nthreads > 0)
    {
        nestedHelloWorld<<<1, nthreads>>>(nthreads, ++iDepth, maxDepth);
        printf("-------> nested execution depth: %d\n", iDepth);
    }
}

int main1(int argc, char **argv)
{
    int size = 8;
    int blocksize = 8;   // initial block size
    int igrid = 1;

    if(argc > 1)
    {
        igrid = atoi(argv[1]);
        size = igrid * blocksize;
    }

    dim3 block (blocksize, 1);
    dim3 grid  ((size + block.x - 1) / block.x, 1);
    printf("%s Execution Configuration: grid %d block %d\n", argv[0], grid.x,
           block.x);

    nestedHelloWorld<<<grid, block>>>(block.x, 0, 2);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceReset());
    return 0;
}
