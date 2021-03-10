//
// Created by 李亘杰 on 2021/1/20.
//

#include "sumArray_timer.cuh"



double cpuSecond(){
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec+(double )tp.tv_usec*1.e-6);
}

void initData(float *ip,int size){
    time_t t;
    srand((unsigned )time(&t));
    for (int i=0;i<size;i++){
        ip[i] = (float )(rand()& 0xFF)/10.0f;
    }
}

void sumArraysOnHost(float *A,float *B,float *C,const int N){
    for (int idx=0;idx<N;idx++)
        C[idx] = A[idx] + B[idx];
}

void sumMatrixOnHost(float *A,float *B,float *C,const int nx,const int ny){
    float *ia = A;
    float *ib = B;
    float *ic = C;

    for (int iy=0;iy<ny;iy++){
        for (int ix=0;ix<nx;ix++){
            ic[ix] = ia[ix] + ib[ix];
        }
        ia+=nx;
        ib+=nx;
        ic+=nx;
    }
}

void checkResult(float *hostRef,float *gpuRef,const int N){
    double epsilon = 1.0e-8;
    bool match = 1;
    for (int i=0;i<N;i++){
        if (abs(hostRef[i]-gpuRef[i])>epsilon){
            match = 0;
            printf("Arrays does not match!\n");
            printf("i:%d  host %5.2f, gpu %5.2f",i,hostRef[i],gpuRef[i]);
            break;
        }
    }
}

void initialInt(int *ip,int size){
    for (int i=0;i<size;i++){
        ip[i] = i;
    }
}

void printMatrix(int *C,const int nx,const int ny){
    int *ic = C;
    printf("\nMatrix:(%d,%d)\n",nx,ny);
    for (int iy=0;iy<ny;iy++){
        for (int ix=0;ix<nx;ix++){
            printf("%3d",ic[ix]);
        }
        ic += nx;
        printf("\n");
    }
    printf("\n");
}

__global__ void printThreadIndex(int *A,const int nx,const int ny){
    int ix = threadIdx.x + blockDim.x*blockIdx.x;
    int iy = threadIdx.y + blockDim.y*blockIdx.y;
    int idx = iy*nx+ix;

    printf("thread_id (%d,%d) block_id (%d,%d)  coordinate (%d,%d)  global index %2d ival %2d\n",
           threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,ix,iy,idx,A[idx]);
}

__global__ void sumArraysOnGpu(float *A,float *B,float *C,const int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<N) C[i] = A[i]+B[i];
}

__global__ void sumMatrixOnGPU(float *MatrixA,float *MatrixB,float *MatrixC,const int nx,const int ny){
    unsigned int ix = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int iy = threadIdx.y + blockDim.y * blockIdx.y;
    unsigned int idx = iy*nx +ix;
    if (ix<nx && iy<ny){
        MatrixC[idx] = MatrixA[idx] + MatrixB[idx];
    }
}

__global__ void sumMatrixOnGPU1D2(float *MatrixA,float *MatrixB,float *MatrixC,const int nx,const int ny){
    unsigned int ix = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int iy = threadIdx.y + blockDim.y;
    if (ix<nx){
        for (int j=0;j<2;j++){
            int idx = nx*iy+j*nx + ix;
            MatrixC[idx] = MatrixA[idx] + MatrixB[idx];
        }
    }
}
__global__ void sumMatrixOnGPU1DY(float *MatrixA,float *MatrixB,float *MatrixC,const int nx,const int ny){
    unsigned int ix = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int iy = threadIdx.y;
    unsigned int idx = iy*nx+ix;
    if (ix<nx && iy<ny){
        MatrixC[idx] = MatrixA[idx] + MatrixB[idx];
    }
}

// grid 2D block 1D
__global__ void sumMatrixOnGPUMixGrid2Block1(float *MatA, float *MatB, float *MatC, int nx,
                                  int ny)
{
    unsigned int nxthreads = gridDim.x * blockDim.x;
    unsigned int iy1 = 2*blockIdx.y;
    unsigned int iy2 = 2*blockIdx.y+1;
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
//    unsigned int ix2 = ix + nxthreads;

    unsigned int idx = iy1 * nx + ix;
    unsigned int idx2 = idx+nxthreads;

    if (ix < nx)
    {
        if (iy1 < ny)
            MatC[idx] = MatA[idx] + MatB[idx];
        if (iy2 < ny)
            MatC[idx2] = MatA[idx2] + MatB[idx2];
//        if (ix2 < nx)
//            MatC[idx2] = MatA[idx2] + MatB[idx2];
    }
}


__global__ void sumMatrixOnGPU2D(float *A,float *B,float *C,const int nx,const int ny){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = iy * nx + ix;

    if (ix<nx && iy<ny){
        C[idx] = A[idx] + B[idx];
    }
}

int computeTime(){
    printf("Starting----------\n");

    //set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp,dev))
    printf("Using device %d:\n",dev);
    CHECK(cudaSetDevice(dev));

    //set up size of vectors
    int nElem = 1<<24;
    printf("Vector size %d\n",nElem);

    size_t nBytes = nElem*sizeof(float);

    float *h_A,*h_B,*hostRef,*gpuRef;
    h_A = (float *) malloc(nBytes);
    h_B = (float *) malloc(nBytes);
    hostRef = (float *) malloc(nBytes);
    gpuRef = (float *) malloc(nBytes);

    double iStart,iElaps;

    iStart = cpuSecond();
    initData(h_A,nElem);
    initData(h_B,nElem);
    iElaps = cpuSecond()-iStart;

    memset(hostRef,0,nBytes);
    memset(gpuRef,0,nBytes);

    iStart = cpuSecond();
    sumArraysOnHost(h_A,h_B,hostRef,nElem);
    iElaps = cpuSecond()-iStart;
    printf("sumArraysOnHost  Time elapsed %f sec\n",iElaps);


    float *d_A,*d_B,*d_C;
    cudaMalloc((float **)&d_A,nBytes);
    cudaMalloc((float **)&d_B,nBytes);
    cudaMalloc((float **)&d_C,nBytes);

    cudaMemcpy(d_A,h_A,nBytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,nBytes,cudaMemcpyHostToDevice);

    int iLen = 64;
    dim3 block(iLen);
    dim3 grid((nElem+block.x-1)/block.x);

    iStart = cpuSecond();
    sumArraysOnGpu<<<grid,block>>>(d_A,d_B,d_C,nElem);
    iElaps = cpuSecond()-iStart;
    printf("sumArraysOnGpu<<<%d,%d>>> Time elapsed %f sec\n",grid.x,block.x,iElaps);

    cudaMemcpy(gpuRef,d_C,nBytes,cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);
}

int printThreadIdx(){
    printf("Starting ... \n");
    int dev = 0;
    cudaDeviceProp cudaDeviceProp;
    CHECK(cudaGetDeviceProperties(&cudaDeviceProp,dev));
    printf("Using Device %d\n",dev);
    CHECK(cudaSetDevice(dev));

    int nx=8;
    int ny=6;
    int nxy = nx*ny;
    int nBytes = nxy*sizeof(float);

    int *h_A;
    h_A = (int *) malloc(nBytes);

    initialInt(h_A,nxy);
    printMatrix(h_A,nx,ny);

    int *d_MatA;
    cudaMalloc((void**)&d_MatA,nBytes);
    cudaMemcpy(d_MatA,h_A,nBytes,cudaMemcpyHostToDevice);

    dim3 block(4,2);
    dim3 grid((nx+block.x-1)/block.x,(ny+block.y-1)/block.y);
    printThreadIndex<<<grid,block>>>(d_MatA,nx,ny);
    cudaDeviceSynchronize();

    cudaFree(d_MatA);
    free(h_A);

    cudaDeviceReset();
    return 0;
}

int main3(int argc,char* argv[]){
    int dimx;
    int dimy;

    if (argc>2){
        dimx = atoi(argv[1]);
        dimy = atoi(argv[2]);
    }
    printf("Starting ... \n");
    int dev = 0;
    cudaDeviceProp cudaDeviceProp;
    CHECK(cudaGetDeviceProperties(&cudaDeviceProp,dev));
    printf("Using Device %d\n",dev);
    CHECK(cudaSetDevice(dev));

    int nx = 1<<14;
    int ny = 1<<14;

    int nxy = nx*ny;
    int nBytes = nxy*sizeof(float );
    printf("Matrix size :nx %d,ny %d\n",nx,ny);

    float *h_A,*h_B,*hostRef,*gpuRef;
    h_A = (float *) malloc(nBytes);
    h_B = (float *) malloc(nBytes);
    hostRef = (float *) malloc(nBytes);
    gpuRef = (float *) malloc(nBytes);

    double iStart,iElaps;
    iStart= cpuSecond();
    initData(h_A,nxy);
    initData(h_B,nxy);
    iElaps = cpuSecond()-iStart;

    memset(hostRef,0,nBytes);
    memset(gpuRef,0,nBytes);

    iStart= cpuSecond();
    sumMatrixOnHost(h_A,h_B,hostRef,nx,ny);
    iElaps = cpuSecond()-iStart;
    printf("sumMatrix on CPU cost time %f sec\n",iElaps);

    float *d_MatA,*d_MatB,*d_MatC;
    cudaMalloc((void **)&d_MatA,nBytes);
    cudaMalloc((void **)&d_MatB,nBytes);
    cudaMalloc((void **)&d_MatC,nBytes);

    cudaMemcpy(d_MatA,h_A,nBytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_MatB,h_B,nBytes,cudaMemcpyHostToDevice);

//    int dimx = 128;
//    int dimy = 1;
    dim3 block(dimx,dimy);
    dim3 grid((nx+block.x-1)/block.x,(ny+block.y-1)/block.y);

    iStart= cpuSecond();
    sumMatrixOnGPU2D<<<grid,block>>>(d_MatA,d_MatB,d_MatC,nx,ny);
    cudaDeviceSynchronize();
    iElaps = cpuSecond()-iStart;
    printf("sumMatrix on GPU <<(%d,%d),(%d,%d)>> cost time:%f sec\n",grid.x,grid.y,block.x,block.y,iElaps);

    cudaMemcpy(gpuRef,d_MatC,nBytes,cudaMemcpyDeviceToHost);
    checkResult(hostRef,gpuRef,nxy);

    cudaFree(d_MatA);
    cudaFree(d_MatB);
    cudaFree(d_MatC);

    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    cudaDeviceReset();
    return 0;
}