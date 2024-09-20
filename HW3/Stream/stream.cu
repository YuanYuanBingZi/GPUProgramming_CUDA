#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>
// CUDA kernel for the "copy" operation
__global__ void copy_kernel(float *a, float *c, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        c[id] = a[id];
    }
}

// CUDA kernel for the "scale" operation
__global__ void scale_kernel(float *b, float *c, float scalar, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        b[id] = scalar * c[id];
    }
}

// CUDA kernel for the "add" operation
__global__ void add_kernel(float *a, float *b, float *c, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        c[id] = a[id] + b[id];
    }
}

// CUDA kernel for the "triad" operation
__global__ void triad_kernel(float *a, float *b, float *c, float scalar, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        a[id] = b[id] + scalar * c[id];
    }
}


int main(int argc, char *argv[]) {
    //check input parameter
    if (argc < 3) {
        std::cerr << "Correct Format: " << argv[0] << " -size <n>" << std::endl;
        return 1;
    }

    int n = atoi(argv[2]);
    float *a, *b, *c;
    float *d_a, *d_b, *d_c;
    float scalar = 5.0;

      // Allocate host memory
    a = new float[n];
    b = new float[n];
    c = new float[n];

    // Initialize arrays
    std::fill_n(a, n, 1.0f);
    std::fill_n(b, n, 2.0f);
    std::fill_n(c, n, 0.0f);

    // Allocate device memory
    cudaMalloc((void **)&d_a, n * sizeof(float));
    cudaMalloc((void **)&d_b, n * sizeof(float));
    cudaMalloc((void **)&d_c, n * sizeof(float));

    // Copy host memory to device memory
    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, n * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch each kernel and measure the time
    struct timeval start, end;
    gettimeofday(&start, NULL);
    // Copy kernel
    copy_kernel<<<blocks, threadsPerBlock>>>(d_a, d_c, n);
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    long seconds = end.tv_sec - start.tv_sec;
    long micros = end.tv_usec - start.tv_usec;
    double totalTime = seconds*100000 + micros;
    std::cout << "Execution time for COPY: " << totalTime << "ms" << std::endl;

    // Scale kernel
    gettimeofday(&start, NULL);
    scale_kernel<<<blocks, threadsPerBlock>>>(d_b, d_c,scalar, n);
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    seconds = end.tv_sec - start.tv_sec;
    micros = end.tv_usec - start.tv_usec;
    totalTime = seconds*100000 + micros;
    std::cout << "Execution time for SCALE: " << totalTime << "ms" << std::endl;

    //Add kernel
    gettimeofday(&start, NULL);
    add_kernel<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    seconds = end.tv_sec - start.tv_sec;
    micros = end.tv_usec - start.tv_usec;
    totalTime = seconds *100000 + micros;
    std::cout << "Execution time for ADD: " << totalTime << "ms" << std::endl;

    //Triad kernel
    gettimeofday(&start, NULL);
    triad_kernel<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, scalar, n);
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    seconds = end.tv_sec - start.tv_sec;
    micros = end.tv_usec - start.tv_usec;
    totalTime = seconds*100000 + micros;
    std::cout << "Execution time for TRIAD: " << totalTime << "ms" << std::endl;


    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] a;
    delete[] b;
    delete[] c;


    return 0;
}
                     