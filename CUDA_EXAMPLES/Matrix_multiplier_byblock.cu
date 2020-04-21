#include <stdlib.h>
#include <stdio.h>
 
__global__ void addKernel(int *c, int  *a,const int *b){
    int x = threadIdx.x;
    int y = threadIdx.y;
    int i = y * (blockDim.x) + x;   //index = y * WIDTH + x
    c[i] = a[i] + b[i];
}

__global__ void mulKernel(int* c, int* a, const int* b,int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n) {
        for (int i = 0; i < n; ++i) {
            c[row * n + col] += a[row * n + i] * b[i * n + col];
        }
    }
}
 

__host__ int main(){
    const int WIDTH = 16;
    const int TILE_WIDTH = 4;


    int a[WIDTH][WIDTH];
    int b[WIDTH][WIDTH];
    int c[WIDTH][WIDTH];

    for (int y = 0; y < WIDTH; y++) {
        for (int x = 0; x < WIDTH; x++) {
            a[y][x] = y * 10 + x;
            b[y][x] = (y * 10 + x) * 100;
        }
    }

    printf("A Matrix--------------------------------\n");

    for (int y = 0; y < WIDTH; y++) {
        for (int x = 0; x < WIDTH; x++) {
            printf("%5d ", a[y][x]); // A Matrix
        }
        printf("\n");
    }
    printf("B Matrix--------------------------------\n");
    for (int y = 0; y < WIDTH; y++) {
        for (int x = 0; x < WIDTH; x++) {
            printf("%5d ", b[y][x]); // B Matrix
        }
        printf("\n");
    }
    printf("\n\n");


    int* dev_a, * dev_b, * dev_c = 0;
    cudaMalloc((void**)&dev_a, WIDTH * WIDTH * sizeof(int));
    cudaMalloc((void**)&dev_b, WIDTH * WIDTH * sizeof(int));
    cudaMalloc((void**)&dev_c, WIDTH * WIDTH * sizeof(int));

    cudaMemcpy(dev_a, a, WIDTH * WIDTH * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, WIDTH * WIDTH * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimGrid(WIDTH / TILE_WIDTH, WIDTH / TILE_WIDTH, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    mulKernel <<< dimGrid,dimBlock >>> (dev_c, dev_a, dev_b, WIDTH);

    cudaMemcpy(c, dev_c, WIDTH * WIDTH * sizeof(int), cudaMemcpyDeviceToHost);

    for (int y = 0; y < WIDTH; y++) {
        for (int x = 0; x < WIDTH; x++) {
            printf("%6d ", c[y][x]);
        }
        printf("\n");   //parallel  Multiplication result
    }
}