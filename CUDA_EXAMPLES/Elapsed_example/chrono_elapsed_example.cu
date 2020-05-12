#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <chrono>


__global__ void mulKernel(int* c, int* a, const int* b) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = y * (gridDim.x + blockDim.x) + x;

    int sum = 0;
    for (int k = 0; k < gridDim.x + blockDim.x; k++) {
        sum += a[y * (gridDim.y + blockDim.y) + k] + b[k * (gridDim.x + blockDim.x) + x];
    }
    c[i] = sum;
}

__global__ void mulKernel2(int* c, int* a, const int* b, const int WIDTH) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = y * WIDTH + x;

    int sum = 0;
    for (int k = 0; k < WIDTH; k++) {
        sum += a[y * WIDTH + k] * b[k * WIDTH + x];
    }
    c[i] = sum;
}


__host__ int main() {

    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    
    const int WIDTH = 256;
    const int TILE_WIDTH = 16;
    const int GRID_WIDTH = WIDTH / TILE_WIDTH;

    int a[WIDTH][WIDTH];
    int b[WIDTH][WIDTH];
    int c[WIDTH][WIDTH];

    for (int y = 0; y < WIDTH; y++) {
        for (int x = 0; x < WIDTH; x++) {
            a[y][x] = y * 10 + x;
            b[y][x] = (y * 10 + x) * 100;
        }
    }

    int* dev_a, * dev_b, * dev_c = 0;
    cudaMalloc((void**)&dev_a, WIDTH * WIDTH * sizeof(int));
    cudaMalloc((void**)&dev_b, WIDTH * WIDTH * sizeof(int));
    cudaMalloc((void**)&dev_c, WIDTH * WIDTH * sizeof(int));

    cudaMemcpy(dev_a, a, WIDTH * WIDTH * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, WIDTH * WIDTH * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimGrid(GRID_WIDTH, GRID_WIDTH, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    //MulKernel
    //mulKernel <<< dimGrid,dimBlock >>> (dev_c, dev_a, dev_b);

    //cudaMemcpy(c, dev_c, WIDTH * WIDTH * sizeof(int), cudaMemcpyDeviceToHost);


    //MulKernel2
    mulKernel2 << < dimGrid, dimBlock >> > (dev_c, dev_a, dev_b, WIDTH);

    cudaMemcpy(c, dev_c, WIDTH * WIDTH * sizeof(int), cudaMemcpyDeviceToHost);


    for (int y = 0; y < WIDTH; y++) {
        for (int x = 0; x < WIDTH; x++) {
            printf("%9d ", c[y][x]);
        }
        printf("\n");   //parallel  Multiplication result
    }

    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    std::chrono::nanoseconds duration_nano = end - start;
    printf("%lld\n", duration_nano);
}