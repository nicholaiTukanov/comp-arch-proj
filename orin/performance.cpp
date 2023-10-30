#include <iostream>
#include <cstdlib>
#include <string.h>
#include <chrono>

// number of trials of SGEMM kernel 
#define NUM_RUNS 10

#define GPU

#ifdef GPU
#include "cuda.h"
#include "cublas_v2.h"
#include "cuda_runtime.h"
// cublasStatus_t cublasSgemm(cublasHandle_t handle,
//                            cublasOperation_t transa, cublasOperation_t transb,
//                            int m, int n, int k,
//                            const float           *alpha,
//                            const float           *A, int lda,
//                            const float           *B, int ldb,
//                            const float           *beta,
//                            float           *C, int ldc)
void cuda_check(cudaError_t err) {
    if(err != cudaSuccess) {
        printf("[ERROR] cuda runtime error %d", err);
        exit(-1);
    }
}
void cuda_check(cublasStatus_t err) {
    if(err != CUBLAS_STATUS_SUCCESS) {
        printf("[ERROR] cuda blas error %d", err);
        exit(-1);
    }
}
#elif defined(CPU)
#include "armpl.h"
// void sgemm_(const char *transa, const char *transb, const armpl_int_t *m,
//             const armpl_int_t *n, const armpl_int_t *k, const float *alpha,
//             const float *a, const armpl_int_t *lda, const float *b,
//             const armpl_int_t *ldb, const float *beta, float *c,
//             const armpl_int_t *ldc, ... );
#endif

// allocs and initalize an array from [-1,1]
void init(float **arr, uint64_t elems) {
    *arr = (float*)malloc(elems*sizeof(float));
    for(uint64_t i=0; i<elems; i++) {
        int v1 = rand(), v2 = rand(), s = rand()%2;
        int min = std::min(v1,v2);
        int max = std::max(v1,v2);
        float value = ((float)min)/((float)max);
        arr[i] = (s==0 ? value : -value);
    }
}

// outputs execution time for a given SGEMM kernel
// assumes alpha = beta = 1
// assumes C+=AB
// does NOT time data transfer time
void get_performance(uint64_t m, uint64_t n, uint64_t k) {

    float alpha=1.0,beta=1.0;

    float *h_a, *h_b, *h_c;
    init(&h_a, m*k);
    init(&h_b, k*n);
    init(&h_c, m*n);

    // if we are using GPUs, alloc and copy data
    #ifdef GPU
    float *d_a, *d_b, *d_c;
    cuda_check(cudaMalloc(&d_a, m*k*sizeof(float)));
    cuda_check(cudaMalloc(&d_b, k*n*sizeof(float)));
    cuda_check(cudaMalloc(&d_c, m*n*sizeof(float)));
    cuda_check(cudaMemcpy(d_a, h_a, m*k*sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(d_b, h_b, k*n*sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(d_c, h_c, m*n*sizeof(float), cudaMemcpyHostToDevice));
    cublasHandle_t handle;
    cuda_check(cublasCreate(&handle));
    cudaEvent_t d_s,d_e;
    cudaEventCreate(&d_s); cudaEventCreate(&d_e);
    #endif

    double best=1e9, elapsed; // record best time after NUM_RUNS
    for(int IRUNS=0; IRUNS < NUM_RUNS; IRUNS++) {
        
        #ifdef GPU
        // assume row major for all
        cudaEventRecord(d_s);
        cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k,
            &alpha,
            d_a, k,
            d_b, n,
            &beta,
            d_c, n
        );
        cudaEventRecord(d_e);
        cudaEventSynchronize(d_e);
        float elapsed_;
        cudaEventElapsedTime(&elapsed_, d_s, d_e);
        elapsed = (double) elapsed_;
        #elif defined(CPU)
        std::chrono::steady_clock::time_point s = std::chrono::stead_clock::now();
        sgemm_(
            'n', 'n',
            &m, &n, &k,
            &alpha,
            h_a, &k,
            h_b, &n,
            &beta,
            h_c, &n
        );
        std::chrono::steady_clock::time_point e = std::chrono::steady_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(e-s).count();
        #endif

        best = std::min(best, elapsed);
    }

    // we only print time
    // use python scripts to extract this and plot data
    printf("%f\n",best);

    free(h_a);
    free(h_b);
    free(h_c);

    #ifdef GPU
    cuda_check(cudaFree(d_a));
    cuda_check(cudaFree(d_b));
    cuda_check(cudaFree(d_c));
    // cuda_check(cublasDestory(handle));
    #endif
}

int main(int argc, char **argv) {

    uint64_t m = atoi(argv[1]);
    uint64_t n = atoi(argv[2]);
    uint64_t k = atoi(argv[3]);

    get_performance(m,n,k);

}