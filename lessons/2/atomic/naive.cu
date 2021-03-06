#include <cuda.h>
#include <stdio.h>
#include <iostream>


#define NUM_THREADS 1000000
#define ARRAY_SIZE 100

#define BLOCK_WIDTH 1000

struct GpuTimer
{
	cudaEvent_t start;
	cudaEvent_t stop;

	GpuTimer () 
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer ()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start ()
	{
		cudaEventRecord(start, 0);
	}

	void Stop ()
	{
		cudaEventRecord(stop, 0);
	}

	float Elapsed ()
	{
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
	}
};


struct GpuTimerWrapper
{
	GpuTimer timer;

	GpuTimerWrapper ()
	{
		timer.Start();
	}

	~GpuTimerWrapper ()
	{
		timer.Stop();
		printf("Time elapsed = %g ms\n", timer.Elapsed());
	}
};


void print_array (int *array, int size)
{
	printf("{");
	for (int i = 0; i < size; i++) { printf("%d ", array[i]); }
	printf("}\n");
}

__global__ void increment_naive (int *g)
{
	// which thread is this?
	int i = blockIdx.x * blockDim.x + threadIdx.x; 

	// each thread to increment consecutive elements, wrapping at ARRAY_SIZE
	i = i % ARRAY_SIZE;  
	g[i] = g[i] + 1;
}


int main ()
{
    //GpuTimer timer;
    printf("%d total threads in %d blocks writing into %d array elements\n",
           NUM_THREADS, NUM_THREADS / BLOCK_WIDTH, ARRAY_SIZE);

    // declare and allocate host memory
    int h_array[ARRAY_SIZE];
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);
 
    // declare, allocate, and zero out GPU memory
    int * d_array;

	// init data in host
    cudaMalloc((void **) &d_array, ARRAY_BYTES);
	if (d_array == NULL) {
		std::cout << "Failed to alloc GPU mem\n";
		return -1;
	}

    cudaMemset((void *) d_array, 0, ARRAY_BYTES); 

    // launch the kernel - comment out one of these
    {
    	GpuTimerWrapper();
	    //timer.Start();
		increment_naive<<<NUM_THREADS/BLOCK_WIDTH, BLOCK_WIDTH>>>(d_array);
		//timer.Stop();
	}

    // copy back the array of sums from GPU and print
    cudaMemcpy(h_array, d_array, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    print_array(h_array, ARRAY_SIZE);
    //printf("Time elapsed = %g ms\n", timer.Elapsed());
 
    // free GPU memory allocation and exit
    cudaFree(d_array);

	return 0;
}

