//
// Created by 李亘杰 on 2021/1/25.
//


#include "include/cuda_runtime.h"
#include "include/device_launch_parameters.h"
#include "sys/time.h"
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "include/cuda_profiler_api.h"



#define PRECISION 1e-5
#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))

static void HandleError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
               file, line);
        exit(EXIT_FAILURE);
    }
}

void sumMatrix2DOnHost(float *h_A, float *h_B, float *hostRef, int nx, int ny)
{
    for (int i = 0; i< nx*ny; i++)
        hostRef[i] = h_A[i] + h_B[i];
}


__global__ void sumMatrix2DKernel(float *d_MatA, float *d_MatB, float *d_MatC, int nx, int ny)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    int tid = nx*idy + idx;

    if (idx < nx && idy < ny)
        d_MatC[tid] = d_MatA[tid] + d_MatB[tid];
}

int main5(int argc, char **argv)
{
    //
    printf("%s Program Starting...\n",argv[0]);

    // set up device
    int devID = 0; cudaDeviceProp deviceProp;
    HANDLE_ERROR(cudaGetDeviceProperties(&deviceProp, devID));

    //
    printf("Using Device %d: %s\n", devID, deviceProp.name);
    HANDLE_ERROR(cudaSetDevice(devID));

    // set up date size of matrix
    int nx = 1<<14;
    int ny = 1<<14;
    int nxy = nx*ny;
    int nBytes = nxy * sizeof(float);
    //
    printf("Matrix size: nx= %d, ny= %d\n",nx, ny);

    // malloc host memory
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    // initialize data at host side
    for(int i=0;i<nxy;i++)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }
    memset(hostRef, 0, nBytes); memset(gpuRef, 0, nBytes);

    // add matrix at host side for result checks
    float iElaps;
    clock_t iStart,iEnd;
    iStart = clock();

    // time counter
    sumMatrix2DOnHost(h_A, h_B, hostRef, nx,ny);
    iEnd = clock();
    //
    iElaps = (double)(iEnd-iStart)/CLOCKS_PER_SEC;

    // second
    iElaps = (double)(iEnd-iStart)/1000;

    // ms
    printf("--sumMatrix2DOnHost() elapsed %f ms..\n", iElaps);

    // malloc device global memory
    float *d_MatA, *d_MatB, *d_MatC;
    cudaMalloc((void **)&d_MatA, nBytes);
    cudaMalloc((void **)&d_MatB, nBytes);
    cudaMalloc((void **)&d_MatC, nBytes);

    // transfer data from host to device
    cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice);


    // invoke kernel at host side
    int dimx = 32;
    //int dimx = 16;
    int dimy = 32;
    //int dimy = 16;

    if (argc > 2)
        //配置block的维度
    {
        dimx = atoi(argv[1]);
        dimy = atoi(argv[2]);
    }

    dim3 block(dimx, dimy);
    dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y);
    // calculate run time on GPU

    float elapsedTime;
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));

    sumMatrix2DKernel <<< grid, block >>>(d_MatA, d_MatB, d_MatC, nx, ny);

    cudaDeviceSynchronize();
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

    printf("--sumMatrix2DOnGPU<<<(%d,%d),(%d,%d)>>> elapsed %f ms..\n", grid.x, grid.y, block.x, block.y, elapsedTime);
    //
    // copy kernel result back to host side
    cudaProfilerStart();
    cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost);
    cudaProfilerStop();
    // check device results
    for(int i=0; i< nxy; i++)
    {
        if(fabs(gpuRef[i]-hostRef[i]) > PRECISION)
        {
            fprintf(stderr,"Result verification failed at elemnt %d\n", i);
            exit(EXIT_FAILURE);
        }
    }

    // free device global memory
    cudaFree(d_MatA);
    cudaFree(d_MatB);
    cudaFree(d_MatC);

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    // reset device
    cudaDeviceReset();
    //
    printf("Test Passed..\n"); return 0;

}