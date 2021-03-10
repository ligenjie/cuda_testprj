//
// Created by 李亘杰 on 2021/1/20.
//

#ifndef CUDA_TESTPRJ_SUMARRAY_TIMER_CUH
#define CUDA_TESTPRJ_SUMARRAY_TIMER_CUH

#include "include/cuda_runtime.h"
#include "include/device_launch_parameters.h"
#include <iostream>
#include "sys/time.h"

#define CHECK(call)


class sumArray_timer {

};

double cpuSecond();
void initData(float *ip,int size);
void sumArraysOnHost(float *A,float *B,float *C,const int N);
__global__ void sumArraysOnGpu(float *A,float *B,float *C,const int N);

#endif //CUDA_TESTPRJ_SUMARRAY_TIMER_CUH
