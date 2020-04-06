#include <stdlib.h>
#include <stdio.h>
#define SIZE 5
 
__global__ void addKernel(int *c, int  *a,const int *b){
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
 
__host__ int main(){
    const int a[SIZE] = { 3,6,9,12,15 };
    const int b[SIZE] = { 4,7,10,13,16 };
    int c[SIZE] = { 0 };

    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    
    cudaMalloc((void**)&dev_c, SIZE * sizeof(int));
    cudaMalloc((void**)&dev_a, SIZE * sizeof(int));
    cudaMalloc((void**)&dev_b, SIZE * sizeof(int));

    cudaMemcpy(dev_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, SIZE * sizeof(int), cudaMemcpyHostToDevice);


    addKernel <<<1, SIZE >>>(dev_c, dev_a, dev_b);

    cudaMemcpy(c, dev_c, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    printf("{%d, %d, %d, %d, %d} + {%d, %d, %d, %d, %d} = {%d, %d, %d, %d, %d}\n", a[0], a[1], a[2], a[3], a[4], b[0], b[1], b[2], b[3], b[4], c[0], c[1], c[2], c[3], c[4]);

    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return 0;
}