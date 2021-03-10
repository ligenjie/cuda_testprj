#include "../common.h"
#include <stdio.h>
#include "../include/cuda_runtime.h"
#include "../include/device_launch_parameters.h"

#include <stdio.h>

#define MEGABYTE    (1024 * 1024)

int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    CHECK(cudaSetDevice(dev));

    if (argc != 2) {
        printf("usage: %s <size-in-mbs>\n", argv[0]);
        return 1;
    }

    // memory size
    int n_mbs = atoi(argv[1]);
    unsigned int nbytes = n_mbs * MEGABYTE;

    // get device information
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting at ", argv[0]);
    printf("device %d: %s nbyte %5.2fMB\n", dev,
           deviceProp.name, nbytes / (1024.0f * 1024.0f));

    // allocate the host memory
    double start = seconds();
    float *h_a = (float *)malloc(nbytes);
    double elapsed = seconds() - start;
    printf("Host memory allocation took %2.10f us\n", elapsed * 1000000.0);

    // allocate the device memory
    start = seconds();
    float *d_a;
    CHECK(cudaMalloc((float **)&d_a, nbytes));
    elapsed = seconds() - start;
    printf("Allocate device memory allocation took %2.10f us\n", elapsed * 1000000.0);

    // initialize the host memory
    for(unsigned int i = 0; i < nbytes / sizeof(float); i++) h_a[i] = 0.5f;

    // transfer data from the host to the device
    start = seconds();
    CHECK(cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyHostToDevice));
    elapsed = seconds() - start;
    printf("Transfer data from the host to the device took %2.10f us\n", elapsed * 1000000.0);

    // transfer data from the device to the host
    start = seconds();
    CHECK(cudaMemcpy(h_a, d_a, nbytes, cudaMemcpyDeviceToHost));
    elapsed = seconds() - start;
    printf("Transfer data from the device to the host took %2.10f us\n", elapsed * 1000000.0);

    // free memory
    start = seconds();
    CHECK(cudaFree(d_a));
    elapsed = seconds() - start;
    printf("Free cuda memory took %2.10f us\n", elapsed * 1000000.0);


    start = seconds();
    free(h_a);
    elapsed = seconds() - start;
    printf("Host memory deallocation took %2.10f us\n", elapsed * 1000000.0);

    // reset device
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}
