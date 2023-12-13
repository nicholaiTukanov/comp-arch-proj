#include <iostream>
#include <cstdlib>
#include <string.h>
#include <chrono>
// #include "oneapi/mkl/blas.hpp"

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <oneapi/mkl.hpp>
#include <dpct/blas_utils.hpp>

// number of trials of SGEMM kernel 
#define NUM_RUNS 100

// allocs and initalize an array from [-1,1]
void init(float **arr, uint64_t elems) {
    *arr = (float*)malloc(elems*sizeof(float));
    for(uint64_t i=0; i<elems; i++) {
        int v1 = rand(), v2 = rand(), s = rand()%2;
        int min = std::min(v1,v2);
        int max = std::max(v1,v2);
        float value = ((float)min)/((float)max);
        (*arr)[i] = (s==0 ? value : -value);
    }
}

// outputs execution time for a given SGEMM kernel
// assumes alpha = beta = 1
// assumes C+=AB
// does NOT time data transfer time
void get_performance(uint64_t m, uint64_t n, uint64_t k, sycl::queue *q) {

    float alpha=1.0,beta=1.0;

    float *h_a, *h_b, *h_c;
    init(&h_a, m*k);
    init(&h_b, k*n);
    init(&h_c, m*n);

    float *d_a = (float*)sycl::malloc_device(m*k*sizeof(float), *q);
    float *d_b = (float*)sycl::malloc_device(k*n*sizeof(float), *q);
    float *d_c = (float*)sycl::malloc_device(m*n*sizeof(float), *q);

    q->memcpy(d_a,h_a,m*k*sizeof(float)).wait();
    q->memcpy(d_b,h_b,n*k*sizeof(float)).wait();
    q->memcpy(d_c,h_c,m*n*sizeof(float)).wait();

    // sycl::buffer<float> a{h_a,sycl::range{m*k}};
    // sycl::buffer<float> b{h_b,sycl::range{n*k}};
    // sycl::buffer<float> c{h_c,sycl::range{m*n}};

    sycl::event e;

    double best=1e9, elapsed; // record best time after NUM_RUNS
    for(int IRUNS=0; IRUNS < NUM_RUNS; IRUNS++) {
        
        std::chrono::steady_clock::time_point s = std::chrono::steady_clock::now();
        e = oneapi::mkl::blas::row_major::gemm(
            *q,
            oneapi::mkl::transpose::nontrans,
            oneapi::mkl::transpose::nontrans,
            m, n, k,
            alpha,
            // dpct::get_value(&alpha, handle),
            d_a, k,
            d_b, n,
            beta,
            // dpct::get_value(&beta, beta),
            d_c, n
        );
        e.wait();
        std::chrono::steady_clock::time_point e = std::chrono::steady_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(e-s).count();

        best = std::min(best, elapsed);
    }

    // we only print time
    // use python scripts to extract this and plot data
    printf("%f\n",best);

    free(h_a);
    free(h_b);
    free(h_c);

    #ifdef GPU
    sycl::free(d_a, *q);
    sycl::free(d_b, *q);
    sycl::free(d_c, *q);
    #endif
}

int main(int argc, char **argv) {

    sycl::property_list propList{sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()};
    #ifdef CPU
    sycl::queue q(sycl::cpu_selector_v, propList);
    #elif
    sycl::queue q(sycl::gpu_selector_v, propList);
    #endif

    // mkl_set_num_threads(112);

    #ifdef ERROR
    printf("[ERROR] no device has been specified\n");
    exit(-1);
    #endif

    if(argc != 4) {
        printf("usage error\n");
        printf("./performance <m> <n> <k>\n");
        exit(-1);
    }

    uint64_t m = atoi(argv[1]);
    uint64_t n = atoi(argv[2]);
    uint64_t k = atoi(argv[3]);

    get_performance(m,n,k,&q);

    return 0;

}