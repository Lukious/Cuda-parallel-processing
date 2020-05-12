#include <stdio.h>
#include <stdlib.h>
#include <io.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <Windows.h>
#include <time.h>
#include <chrono>

#define GRIDSIZE 16*1024
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

    long long cntStart = 0LL, cntEnd = 0LL, freq = 0LL;

    pSource = (float*)malloc(TOTALSIZE * sizeof(float));
    pResult = (float*)malloc(TOTALSIZE * sizeof(float));

    genData(pSource, TOTALSIZE);

    //start the timer
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    //calculate the adjacent differnce
    pResult[0] = 0.0F; //exceptional case for i = 0
    adjDiff(pResult, pSource, TOTALSIZE);

    ///------------------------------------------------------------------


    ///------------------------------------------------------------------

    //end the timer 
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    std::chrono::nanoseconds duration_nano = end - start;
    printf("%lld nano second \n", duration_nano);


    free(pSource);
    free(pResult);
}

