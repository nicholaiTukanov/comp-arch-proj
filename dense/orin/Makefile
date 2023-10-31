
NVCC = /usr/local/cuda/bin/nvcc
CXX = armclang++

gpu_performance.x:
	$(NVCC) -DGPU performance.cpp -lcublas -o gpu_performance.x

cpu_performance.x:
	$(CXX) -DCPU -O3 performance.cpp -armpl -fopenmp -mcpu=generic -o cpu_performance.x

clean:
	rm -rf *.x