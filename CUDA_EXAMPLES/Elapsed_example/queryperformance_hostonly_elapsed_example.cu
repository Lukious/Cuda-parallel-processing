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

void adjDiff(float* dst, const float* src, unsigned int size) {
    for (int i = 1; i < size; ++i) {
        dst[i] = src[i] - src[i - 1];
    }
}

int main(void) {

    float* pSource = NULL;
    float* pResult = NULL;
    int i;

    long long cntStart = 0LL , cntEnd = 0LL , freq = 0LL;

    QueryPerformanceFrequency((LARGE_INTEGER*)(&freq));

    pSource = (float*)malloc(TOTALSIZE * sizeof(float));
    pResult = (float*)malloc(TOTALSIZE * sizeof(float));
    
    genData(pSource, TOTALSIZE);

    //start the timer
    QueryPerformanceCounter((LARGE_INTEGER*)(&cntStart));
    //calculate the adjacent differnce
    pResult[0] = 0.0F; //exceptional case for i = 0
    adjDiff(pResult, pSource, TOTALSIZE);

    ///------------------------------------------------------------------


    ///------------------------------------------------------------------

    //end the timer 
    QueryPerformanceCounter((LARGE_INTEGER*)(&cntEnd));
    printf("elapsed time  = %f usec\n", (double)(cntEnd - cntStart) * 1000000.0 / (double)(freq));

    i = 1;
    printf("i = %7d : %f = %f-%f\n", i, pResult[i], pSource[i], pSource[i - 1]);
    i = TOTALSIZE - 1;
    printf("i = %7d : %f = %f-%f\n", i, pResult[i], pSource[i], pSource[i - 1]);
    i = TOTALSIZE / 2;
    printf("i = %7d : %f = %f-%f\n", i, pResult[i], pSource[i] , pSource[i - 1]);


    free(pSource);
    free(pResult);
}

