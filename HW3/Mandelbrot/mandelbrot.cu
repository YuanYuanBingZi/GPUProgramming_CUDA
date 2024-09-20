#include <iostream>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>


const int max_iterations = 1000;

__device__ int mandelbrot(float real, float imag)
{
    float r = real;
    float i = imag;
    for (int iter = 0; iter < max_iterations; ++iter)
    {
        float r2 = r * r;
        float i2 = i * i;
        if (r2 + i2 > 4.0f)
        {
            return iter;
        }
        i = 2.0f * r * i + imag;
        r = r2 - i2 + real;
    }
    return max_iterations;
}

__global__ void generateMandelbrotSet(int *output, int height, int width)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float real = -2.0 + x*3.0/width;
    float imag = -2.0 + y*4.0/height;

    int value = mandelbrot(real, imag);

    output[y * width + x] = value;
}

int main(int argc, char *argv[])
{
    //check input parameter
    if (argc != 5) {
        std::cerr << "Correct Format: " << argv[0] << " -numx <n> -numy <m>" << std::endl;
        return 1;
    }

    int width = atoi(argv[2]);
    int height = atoi(argv[4]);

    int *host_output = new int[width * height];
    int *device_output;

    cudaMalloc((void **)&device_output, width * height * sizeof(int));

    dim3 blockDim(32, 32);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    generateMandelbrotSet<<<gridDim, blockDim>>>(device_output, height, width);

    cudaMemcpy(host_output, device_output, width * height * sizeof(int), cudaMemcpyDeviceToHost);
    //csv output
    std::ofstream file("mandelbrot.csv");
    if(!file.is_open()){
        std::cerr << "Unable to open file" << std::endl;
        return 1;
    }

    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            file << host_output[j * height + i];
            if (i < width - 1) {
                file << ", ";
            }
        }
        file << "\n";
    }

    file.close();

    cudaFree(device_output);


    delete[] host_output;

    return 0;
}
