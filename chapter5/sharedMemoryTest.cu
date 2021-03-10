//
// Created by 李亘杰 on 2021/3/1.
//

#include <string>
#include "sharedMemoryTest.cuh"

int main(){
    cudaSharedMemConfig *pconfig;
    cudaDeviceGetSharedMemConfig(pconfig);
    printf("%d\n",pconfig);

    cudaSharedMemConfig config = cudaSharedMemBankSizeFourByte;
    printf("config  %d\n",config);

    cudaDeviceSetSharedMemConfig(config);
    cudaDeviceGetSharedMemConfig(pconfig);

//    cudaError_t errorCode = cudaDeviceSetCacheConfig(cudaFuncCachePreferEqual);
    printf("%d\n",pconfig);

}