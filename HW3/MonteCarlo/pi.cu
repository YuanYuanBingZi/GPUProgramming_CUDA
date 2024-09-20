#include <stdio.h>
#include <curand_kernel.h>
#include <curand.h>
#define THREADS_PER_BLOCK 64


__global__ void computePi(int *count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(clock64() + index, 0, 0, &state);
    double x = curand_uniform_double(&state);
    double y = curand_uniform_double(&state);
    int local_count = 0;
    if (x * x + y * y <= 1.0) {
        local_count++;
    }
    count[index] = local_count;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Correct format: %s -numpoints<N>\n", argv[0]);
        exit(1);
    }

    int N = atoi(argv[2]);
    int *d_count, *count;
    count = new int[N];

    cudaMalloc(&d_count, sizeof(int));

    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    computePi<<<blocks, THREADS_PER_BLOCK>>>(d_count);
    cudaDeviceSynchronize();

    cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    int total_hits = 0;
    for(int i = 0; i < N; i++){
        total_hits += count[i];
    }
    double pi = 4.0 * total_hits/ N;
    printf("Pi approximation: %.4f\n", pi);

    cudaFree(d_count);
    return 0;
}