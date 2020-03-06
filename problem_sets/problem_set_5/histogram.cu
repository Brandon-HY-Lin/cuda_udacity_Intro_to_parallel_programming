/* Udacity HW5
   Histogramming for Speed
   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.
   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.
   Here the bin is just:
   bin = val
   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
	 histo[val[i]]++;
   That's it!  Your job is to make it run as fast as possible!
   The values are normally distributed - you may take
   advantage of this fact in your implementation.
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

#include "utils.cuh"

__global__ void naive_histo(const unsigned int* const val,
	unsigned int* histo,
	const unsigned int numElems)
{
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < numElems) {
		unsigned int key = val[index];
		atomicAdd(&histo[key], 1);
	}
}

__global__ void shared_mem_histo(const unsigned int* const val,
	unsigned int* histo,
	const unsigned int numElems,
	const unsigned int numBins)
{
	extern __shared__ unsigned int shared_histo[];

	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int tid = threadIdx.x;

	// init shared data
	if (tid < numBins) {
		shared_histo[tid] = 0;
	}
	__syncthreads();

	atomicAdd(&shared_histo[val[index]], 1);

	__syncthreads();

	// copy data to GPU memory
	if (tid < numBins) {
		atomicAdd(&histo[tid], shared_histo[tid]);
	}
}

__global__
void perBlockHisto(const unsigned int* const vals, //INPUT
	unsigned int* const histo,      //OUPUT
	int numVals, int numBins) {

	extern __shared__ unsigned int sharedHisto[]; //size as original histo

	//coalesced initialization: multiple blocks could manage the same shared histo
	for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
		sharedHisto[i] = 0;
	}

	__syncthreads();

	int globalid = threadIdx.x + blockIdx.x * blockDim.x;
	atomicAdd(&sharedHisto[vals[globalid]], 1);

	__syncthreads();

	for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
		atomicAdd(&histo[i], sharedHisto[i]);
	}


}

void computeHistogram(const unsigned int* const d_vals, 		// INPUT
	unsigned int* const d_histo, 	// OUTPUT
	const unsigned int numBins,
	const unsigned int numElems)
{
	const unsigned int BLOCK_SIZE = 1024;

	//naive_histo<<<numElems / BLOCK_SIZE + 1, BLOCK_SIZE, numBins * sizeof(unsigned int)>>>(d_vals, d_histo, numElems);
	
	shared_mem_histo <<<(numElems-1) / BLOCK_SIZE + 1, BLOCK_SIZE, numBins * sizeof(unsigned int) >>> 
		(d_vals, d_histo, numElems, numBins);
	
	cudaDeviceSynchronize();

	// check error message
	checkCudaErrors(cudaGetLastError());

}
