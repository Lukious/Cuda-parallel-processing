#include <stdlib.h>
#include <stdio.h>
#define SIZE 5
 
__global__ void KernelFunc() {
    dim3 DimGrid(100, 50);
    dim3 DimBlock(4, 8, 8);
}


__global__ void addKernel(int *c, int  *a,const int *b){
    int x = threadIdx.x;
    int y = threadIdx.y;
    int i = y * (blockDim.x) + x;   //index = y * WIDTH + x
    c[i] = a[i] + b[i];
}
 
__host__ int main(){
    const int WIDTH = 5;
    int a[WIDTH][WIDTH];
    int b[WIDTH][WIDTH];
    int c[WIDTH][WIDTH];
    
    for (int y = 0; y < WIDTH; y++) {
        for (int x = 0; x < WIDTH; x++) {
            a[y][x] = y * 10 + x;
            b[y][x] = (y * 10 + x)*100;
        }
    }

    for (int y = 0; y < WIDTH; y++) {
        for (int x = 0; x < WIDTH; x++) {
            c[y][x] = a[y][x] + b[y][x];    //Normal Addition
        }
    }

    for (int y = 0; y < WIDTH; y++) {
        for (int x = 0; x < WIDTH; x++) {
            printf("%5d ",c[y][x]); //Normal Addition Result
        }
        printf("\n");
    }

    printf("\n----------------------------------------------------------\n");
    
    int* dev_a, * dev_b, * dev_c = 0;
    cudaMalloc((void**)&dev_a, WIDTH * WIDTH * sizeof(int));
    cudaMalloc((void**)&dev_b, WIDTH * WIDTH * sizeof(int));
    cudaMalloc((void**)&dev_c, WIDTH * WIDTH * sizeof(int));

    cudaMemcpy(dev_a, a, WIDTH * WIDTH * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, WIDTH * WIDTH * sizeof(int), cudaMemcpyHostToDevice);

    dim3 DimBlock(WIDTH, WIDTH);
    addKernel << <1, DimBlock >> > (dev_c, dev_a, dev_b);   //Pararell Addition

    cudaMemcpy(c, dev_c, WIDTH * WIDTH * sizeof(int), cudaMemcpyDeviceToHost);

    for (int y = 0; y < WIDTH; y++) {
        for (int x = 0; x < WIDTH; x++) {
            printf("%5d ", c[y][x]);
        }
        printf("\n");   //Pararell Addition result
    }
}
