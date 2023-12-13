CC = clang++
CFLAGS = -O3 -std=c++17 -framework Accelerate

LDFLAGS = -lm -lblas -llapack 

performance.x: performance.cpp
	$(CC) $(CFLAGS) -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64 $< -o $@  $(LDFLAGS)

clean:
	rm -f *.x