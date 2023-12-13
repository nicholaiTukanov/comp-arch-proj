#include <iostream>
#include <chrono>
#include <Accelerate/Accelerate.h>
// #include <cblas.h>
#include <random>
#include <limits>
#include <algorithm>

using namespace std;

void sgemm(int m, int n, int k) {
    // Allocate memory for matrices A, B, and C
    float* A = (float*)malloc(m * k * sizeof(float));
    float* B = (float*)malloc(k * n * sizeof(float));
    float* C = (float*)malloc(m * n * sizeof(float));

    // Initialize matrices A and B with random values from -1 to 1
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (int i = 0; i < m * k; i++) {
        A[i] = dis(gen);
    }
    for (int i = 0; i < k * n; i++) {
        B[i] = dis(gen);
    }

    // Perform SGEMM and measure the execution time
    float bestTime = 1e9;
    for (int run = 0; run < 100; run++) {
        auto start = std::chrono::high_resolution_clock::now();

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, A, k, B, n, 1.0f, C, n);

        auto end = std::chrono::high_resolution_clock::now();
        float duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        bestTime = std::min(bestTime, duration);
    }

    printf("%f\n", bestTime);

    // Free memory for matrices A, B, and C
    free(A);
    free(B);
    free(C);

    // std::cout << bestTime << std::endl;
}

int main(int argv, char** argc) {
    
    int m = atoi(argc[1]);
    int n = atoi(argc[2]);
    int k = atoi(argc[3]);

    sgemm(m, n, k);

    return 0;
}
