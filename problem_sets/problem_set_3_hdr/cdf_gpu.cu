#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cassert>

#include "utils.h"

__global__ void min_reduce_kernel(const float* const d_input, float* d_output, int numElems)
{
    extern __shared__ float shmem[];

    const float identityElement = (2 ^ 20);

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    if (index < numElems) {
        shmem[tid] = d_input[index];
    }
    else {
        shmem[tid] = identityElement;
    }
    __syncthreads();

    // tree based reduce
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shmem[tid] = min(shmem[tid], shmem[tid + stride]);
        }
    }

    __syncthreads();
    // aggregate results to 1st block, and move to GPU memory
    if (tid == 0) {
        d_output[blockIdx.x] = shmem[tid];
    }
}



float min_gpu(float* d_output, float* d_itermediate, const float* const d_input, int numElems)
{
    float h_result;

    // step 1
    int totalThreads = numElems;
    int blockSize = 1024;
    int gridSize = (totalThreads - 1) / blockSize + 1;
    min_reduce_kernel<< <gridSize, blockSize, blockSize * sizeof(float) >> > (d_input, d_itermediate, totalThreads);

    // step 2
    totalThreads = gridSize;
    blockSize = totalThreads;
    gridSize = 1;
    min_reduce_kernel<< <gridSize, blockSize, blockSize * sizeof(float) >> > (d_itermediate, d_output, totalThreads);

    checkCudaErrors(cudaMemcpy(&h_result, d_output, sizeof(float), cudaMemcpyDeviceToHost));

    return h_result;
}


__global__ void max_reduce_kernel(const float* const d_input, float* d_output, int numElems)
{
    extern __shared__ float shmem[];

    const float identityElement = -(2 ^ 20);

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    if (index < numElems) {
        shmem[tid] = d_input[index];
    }
    else {
        shmem[tid] = identityElement;
    }
    __syncthreads();

    // tree based reduce
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shmem[tid] = max(shmem[tid], shmem[tid + stride]);
        }
    }

    __syncthreads();
    // aggregate results to 1st block, and move to GPU memory
    if (tid == 0) {
        d_output[blockIdx.x] = shmem[tid];
    }
}



float max_gpu(float* d_output, float* d_itermediate, const float* const d_input, int numElems)
{
    float h_result;

    // step 1
    int totalThreads = numElems;
    int blockSize = 1024;
    int gridSize = (totalThreads - 1) / blockSize + 1;
    max_reduce_kernel<< <gridSize, blockSize, blockSize * sizeof(float) >> > (d_input, d_itermediate, totalThreads);

    // step 2
    totalThreads = gridSize;
    blockSize = totalThreads;
    gridSize = 1;
    max_reduce_kernel<< <gridSize, blockSize, blockSize * sizeof(float) >> > (d_itermediate, d_output, totalThreads);

    checkCudaErrors(cudaMemcpy(&h_result, d_output, sizeof(float), cudaMemcpyDeviceToHost));

    return h_result;
}


void min_max_gpu(float* min_val, float* max_val, const float* d_input, const unsigned int numElems)
{
    float* d_intermediate, * d_output; 
    
    const int bytes = sizeof(float) * numElems;

    checkCudaErrors(cudaMalloc(&d_intermediate, bytes));
    checkCudaErrors(cudaMalloc(&d_output, bytes));

    *min_val = min_gpu(d_output, d_intermediate, d_input, numElems);
    *max_val = max_gpu(d_output, d_intermediate, d_input, numElems);

    if (d_intermediate) cudaFree(d_intermediate);
    if (d_output) cudaFree(d_output);
}



__global__ void histogram_gpu(const float* const d_input, unsigned int* d_histo,
                            const float min_val, const float max_val, const float range_val,
                            const int numElems, const int numBins)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (index < numElems) {
        int bin = max((int)(numBins - 1),
                        (int)((d_input[index] - min_val) / range_val * numBins));

        atomicAdd(&d_histo[bin], 1);
    }
}

/*
 * inclusive scan
 */
__global__ void histogram_to_cdf(const unsigned int* d_histo, unsigned int* d_cdf, const unsigned int numBins)
{
    
    extern __shared__ unsigned int shmem[];
    int index_in = 1, index_out = 0;
    int tid = threadIdx.x;

    // move data from GPU memory to shared memory
    //  in the begining of for-loop, index_in and index_out will swap.
    shmem[index_out * numBins + tid] = d_histo[tid];    // inclusive scan
    //shmem[index_out * numBins + tid] = (tid > 0) ? d_histo[tid - 1] : 0; // exclusive scan

    __syncthreads();

    if (tid < numBins) {
        for (unsigned int stride = 1; stride < numBins; stride *= 2) {
            index_out = 1 - index_out;
            index_in = 1 - index_out;

            if (tid >= stride) {
                shmem[index_out * numBins + tid] = shmem[index_in * numBins + tid] + 
                                                    shmem[index_in * numBins + tid - stride];
            }
            else {
                shmem[index_out * numBins + tid] = shmem[index_in * numBins + tid];
            }

            __syncthreads();
        }
    }
    
    d_cdf[tid] = shmem[index_out * numBins + tid];
}

void cdf_gpu(const float* const d_logLuminance,
    unsigned int* const d_cdf,
    float& min_logLum,
    float& max_logLum,
    const size_t numRows,
    const size_t numCols,
    const size_t numBins)
{
    //TODO
    /*Here are the steps you need to implement
      1) find the minimum and maximum value in the input logLuminance channel
         store in min_logLum and max_logLum
      2) subtract them to find the range
      3) generate a histogram of all the values in the logLuminance channel using
         the formula: bin = (lum[i] - lumMin) / lumRange * numBins
      4) Perform an exclusive scan (prefix sum) on the histogram to get
         the cumulative distribution of luminance values (this should go in the
         incoming d_cdf pointer which already has been allocated for you)       */
    assert(numBins <= 1024);

    const unsigned int numElems = numRows * numCols;
    
    min_max_gpu(&min_logLum, &max_logLum, d_logLuminance, numElems);

    // alloc GPU mem for histogram
    unsigned int* d_histo;
    checkCudaErrors(cudaMalloc(&d_histo, sizeof(unsigned int) * numBins));

    // compute histogram
    histogram_gpu<<<(numElems-1)/1024, 1024>>>(d_logLuminance, d_histo, 
                                                min_logLum, max_logLum,  (max_logLum - min_logLum),
                                                numElems, numBins);

    // numBins must be lower 1024.
    histogram_to_cdf<<<1, numBins, sizeof(unsigned int) * numBins*2 >>>(d_histo, d_cdf, static_cast<unsigned int>(numBins));


    if (d_histo) cudaFree(d_histo);
}