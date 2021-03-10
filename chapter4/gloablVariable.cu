//
// Created by 李亘杰 on 2021/2/23.
//

#include "gloablVariable.cuh"
#include <stdio.h>

const int nElem = 5;
__device__ float devData[nElem];

__global__ void MutiplyVariable(){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<nElem){
        devData[tid] *= tid;
    }
}

int main0(){
    float values[5] = { 3.14f, 3.14f, 3.14f, 3.14f, 3.14f };
//    CHECK(cudaMemcpyToSymbol(devData, values, 5 * sizeof(float)))
    float *dptr = NULL;
    //非常重要，不能在主机端的设备变量使用运算符&，他只是一个在GPU上表示物理位置的符号
    cudaGetSymbolAddress((void**)&dptr,devData);
    cudaMemcpy(dptr,values,5 * sizeof(float),cudaMemcpyHostToDevice);
    printf("Host:   copied [ %f %f %f %f %f ] to the global variable\n",
           values[0], values[1], values[2], values[3], values[4]);

    // invoke the kernel
    MutiplyVariable<<<1, 5>>>();

    // copy the global variable back to the host
//    CHECK(cudaMemcpyFromSymbol(values, devData, 5 * sizeof(float)));
    cudaMemcpy(values,dptr,5 * sizeof(float),cudaMemcpyDeviceToHost);

    printf("Host:   the values changed by the kernel to [ %f %f %f %f %f ]\n",
           values[0], values[1], values[2], values[3], values[4]);

    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}