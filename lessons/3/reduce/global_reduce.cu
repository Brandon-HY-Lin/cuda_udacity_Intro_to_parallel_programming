#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>


struct GpuTimer
{
	cudaEvent_t start;
	cudaEvent_t stop;

	GpuTimer()
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start()
	{
		cudaEventRecord(start, 0);
	}

	void Stop()
	{
		cudaEventRecord(stop, 0);
	}

	float Elapsed()
	{
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
	}

};


__global__ void global_reduce_kernel (float* d_out, float* d_in)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	int tid = threadIdx.x;

	for (unsigned int stride = blockDim.x/2; stride > 0; stride >>= 1 ) {
		// intra-block sum
		if (tid < stride) {
			d_in[index] += d_in[index + stride];
		}

		__syncthreads();	// make sure all adds at one stage are done!
	}

	// only thread 0 writes result for this block back to global mem
	if (tid == 0)
	{
		d_out[blockIdx.x] = d_in[index];
	}
}


void reduce (float* d_out, float* d_intermediate, float* d_in,
			int size)
{
	// assumes that size is not greater than maxThreadsPerBlock^2
	// and that size is a multiple of maxThreadsPerBlock
	const int maxThreadsPerBlock = 1024;
	int threads = maxThreadsPerBlock;
	int blocks = size / maxThreadsPerBlock;

	global_reduce_kernel<<<blocks, threads>>>
		(d_intermediate, d_in);

	// now we're down to one block left, so reduce it
	threads = blocks; // launch one thread for each block in prev step
	blocks = 1;

	global_reduce_kernel<<<blocks, threads>>>
		(d_out, d_intermediate);
}

int main (int argc, char **argv)
{
	// print GPU info
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
	
	GpuTimer timer;
	const int ARRAY_SIZE = 1 << 20;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

	// generate the input array on the host
	float h_in[ARRAY_SIZE];
	float sum = 0.0f;
	for (int i = 0; i < ARRAY_SIZE; i++) {
		// generate random float in [-1.0f, 1.0f]
		h_in[i] = -1.0f + (float)random() / ((float)RAND_MAX/2.0f);
		sum += h_in[i];
	}

	// declare GPU memory pointers
	float *d_in, *d_intermediate, *d_out;

	// allocate GPU memory
	cudaMalloc(&d_in, ARRAY_BYTES);
	if (d_in == NULL) {
		std::cout << "Failed to alloc GPU mem\n";
		return -1;
	}

	cudaMalloc(&d_intermediate, ARRAY_BYTES); // overallocated
	if (d_intermediate == NULL) {
		std::cout << "Failed to alloc GPU mem\n";
		cudaFree(d_in);
		return -1;
	}

	cudaMalloc(&d_out, sizeof(float));
	if (d_out == NULL) {
		std::cout << "Failed to alloc GPU mem\n";
		cudaFree(d_in);
		cudaFree(d_intermediate);
		return -1;
	}

	// transfer the input array to the GPU
	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

        printf("Running global reduce\n");
	
	timer.Start();
	
	for (int i=0; i < 100; i++) {

		reduce(d_out, d_intermediate, d_in, ARRAY_SIZE);
	}
	timer.Stop();

	float elapsedTime;
	elapsedTime = timer.Elapsed();
	elapsedTime /= 100.0f;		// 100 trials


	float h_out;
	cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

	printf("average time elapsed: %f\n", elapsedTime);

    	// free GPU memory allocation
    	cudaFree(d_in);
    	cudaFree(d_intermediate);
    	cudaFree(d_out);
        
    	return 0;
}
