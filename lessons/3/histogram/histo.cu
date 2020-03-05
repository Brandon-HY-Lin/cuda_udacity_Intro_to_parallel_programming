#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <string.h>

int log2 (int i)
{
	int r = 0;
	while (i >>= 1) r++;
	return r;
}

int bit_reverse (int w, int bits)
{
	int r = 0;
	for (int i = 0; i < bits; i++)
	{
		int bit = (w & (1 << i)) >> i;
		r |= bit << (bits - i - 1);
	}
	return r;
}
__global__ void simple_histo (int* d_bins, const int* d_in, const int BIN_COUNT)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	int bin = d_in[index] % BIN_COUNT;
	
	atomicAdd(&(d_bins[bin]), 1);
}


int main (int argc, char **argv)
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0) {
		fprintf(stderr, "error: no devices supporting CUDA.\n");
		exit(EXIT_FAILURE);
	}
	int dev = 0;
	cudaSetDevice(dev);
	
	cudaDeviceProp devProps;
	if (cudaGetDeviceProperties(&devProps, dev) == 0)
	{
		printf("Using device %d:\n", dev);
		printf("%s; global mem: %zdB; compute v%d.%d; clock: %d kHz\n",
			devProps.name, (int)devProps.totalGlobalMem,
			(int)devProps.major, (int)devProps.minor,
			(int)devProps.clockRate);
	}
	
	const int ARRAY_SIZE = 65536;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);
	const int BIN_COUNT = 16;
	const int BIN_BYTES = BIN_COUNT * sizeof(int);
		


	int h_in[ARRAY_SIZE];

	for (int i = 0; i < ARRAY_SIZE; i++) {
		h_in[i] = bit_reverse(i, log2(ARRAY_SIZE));
	}

	int h_bins[BIN_COUNT];
	memset(h_bins, 0, BIN_BYTES);
	
	// declare GPU memory pointers
	int *d_in, *d_bins;

	// allocate GPU memory
	cudaMalloc(&d_in, ARRAY_BYTES);
	if (d_in == NULL) {
		fprintf(stderr, "Failed to alloc GPU mem\n");
		exit(EXIT_FAILURE);
	}

	cudaMalloc(&d_bins, BIN_BYTES);
	if (d_bins == NULL) {
		fprintf(stderr, "Failed to alloc GPU mem\n");
		exit(EXIT_FAILURE);
	}
	
    // transfer the arrays to the GPU
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_bins, h_bins, BIN_BYTES, cudaMemcpyHostToDevice); 

	simple_histo<<<ARRAY_SIZE / 64, 64>>>(d_bins, d_in, BIN_COUNT);

    // copy back the sum from GPU
    cudaMemcpy(h_bins, d_bins, BIN_BYTES, cudaMemcpyDeviceToHost);

    for(int i = 0; i < BIN_COUNT; i++) {
        printf("bin %d: count %d\n", i, h_bins[i]);
    }

    // free GPU memory allocation
	cudaFree(d_in);
	cudaFree(d_bins);

	return 0;
}
