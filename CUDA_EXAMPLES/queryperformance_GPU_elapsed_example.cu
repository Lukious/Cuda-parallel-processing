#include <stdio.h>
#include <stdlib.h>
#include <io.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <Windows.h>
#include <time.h>
#include <chrono>

#define GRIDSIZE 8*1024
#define BLOCKSIZE 1024
#define TOTALSIZE (GRIDSIZE*BLOCKSIZE)

void genData(float* ptr, unsigned int size) {
    while (size--) {
        *ptr++ = (float)(rand() % 1000) / 1000.0F; 
    }
}

__global__ void adjDiff(float* result, float* input) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > 0) {
        float x_i = input[i]; //load input[i] form golbal memory to register
        float x_i_m1 = input[i - 1]; //load input[i-1] from global memory to register
        result[i] = x_i = x_i + x_i_m1; //calculate and store the result to global memory
    }
}

__host__ int main(void) {

    float* pSource = NULL;
    float* pResult = NULL;
    int i;

    long long cntStart = 0LL, cntEnd = 0LL, freq = 0LL;


    QueryPerformanceFrequency((LARGE_INTEGER*)(&freq));


    pSource = (float*)malloc(TOTALSIZE * sizeof(float));
    pResult = (float*)malloc(TOTALSIZE * sizeof(float));

    genData(pSource, TOTALSIZE);

    float* pSourceDev = NULL;
    float* pResultDev = NULL;

    pResult[0] = 0.0F; //exceptional case for i = 0
    cudaMalloc((void**)&pSourceDev, TOTALSIZE * sizeof(float));
    cudaMalloc((void**)&pResultDev, TOTALSIZE * sizeof(float));

    cudaMemcpy(pSourceDev, pSource, TOTALSIZE * sizeof(float), cudaMemcpyHostToDevice);

    //start the timer
    QueryPerformanceCounter((LARGE_INTEGER*)(&cntStart));
    //calculate the adjacent differnce
    // adjDiff(pResult, pSource, TOTALSIZE);

    ///------------------------------------------------------------------
    dim3 dimGrid(GRIDSIZE, 1, 1);
    dim3 dimBlock(BLOCKSIZE, 1, 1);
    adjDiff << <dimGrid, dimBlock >> > (pResultDev, pSourceDev);

    ///------------------------------------------------------------------

    //end the timer 
    QueryPerformanceCounter((LARGE_INTEGER*)(&cntEnd));

    cudaMemcpy(pResult, pResultDev, TOTALSIZE * sizeof(float), cudaMemcpyDeviceToHost);
    printf("elapsed time  = %f usec\n", (double)(cntEnd - cntStart) * 1000000.0 / (double)(freq));

    i = 1;
    printf("i = %7d : %f = %f-%f\n", i, pResult[i], pSource[i], pSource[i - 1]);
    i = TOTALSIZE - 1;
    printf("i = %7d : %f = %f-%f\n", i, pResult[i], pSource[i], pSource[i - 1]);
    i = TOTALSIZE / 2;
    printf("i = %7d : %f = %f-%f\n", i, pResult[i], pSource[i] , pSource[i - 1]);


    free(pSource);
    free(pResult);
    cudaFree(pSourceDev);
    cudaFree(pResultDev);
}

