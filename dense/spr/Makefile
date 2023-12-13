
CXX = icpx

MKLROOT = /opt/intel/oneapi/mkl/latest

MKL_LDFLAGS = -L${MKLROOT}/lib/intel64 -lpthread -lm -ldl -lmkl_sycl
SYCL_LDFLAGS = -lsycl 
LDFLAGS = $(MKL_LDFLAGS) $(SYCL_LDFLAGS)

cpu_performance.x:
	$(CXX) -DCPU -std=c++20 -O3 -fsycl -qmkl=parallel sycl_performance.cpp -o cpu_performance.x $(LDFLAGS)

gpu_performance.x:
	$(CXX) -DGPU -std=c++20 -O3 -fsycl -qmkl=parallel sycl_performance.cpp -o gpu_performance.x $(LDFLAGS)

clean:
	rm -rf *.x 