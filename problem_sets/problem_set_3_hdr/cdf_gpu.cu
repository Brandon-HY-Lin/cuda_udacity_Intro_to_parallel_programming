#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cassert>

template <typename operatorFunc,
          float identityElement>
__global__ void reduce_kernel(float *d_input, float *d_output, numElems)
{
    extern __shared__ float shmem[];

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
    for (unsigned int stride = blockDim.x / 2; i > 0; i >>= 1) {
        if (tid < stride) {
            shmen[tid] = operatorFunc(shmem[tid], shmem[tid + stride]);
        }
    }

    __syncthreads();
    // aggregate results to 1st block, and move to GPU memory
    if (tid == 0) {
        d_output[blockIdx.x] = shmem[tid];
    }
}


template <typename operatorFunc,
          float identityElement>
float reduce(float* d_input, int numElems)
{
    float* d_temp;
    float* d_result;
    float h_result;
    checkCudaErrors(cudaMalloc(&d_temp, sizeof(float) * numElems));
    checkCudaErrors(cudaMalloc(&d_result, sizeof(float) * numElems));

    // step 1
    int totalThreads = numElems;
    int blockSize = 1024;
    int gridSize = (totalThreads - 1) / blockSize + 1;
    reduce_kernel<operatorFunc, identityElement>
        <<<gridSize, blockSize, blockSize*sizeof(float)>>>(d_input, d_temp, totalThreads);

    // step 2
    totalThreads = gridSize;
    blockSize = totalThreads;
    gridSize = 1;
    reduce_kernel<operatorFunc, identityElement>
        <<<gridSize, blockSize, blockSize*sizeof(float) >>> (d_temp, d_result, totalThreads);

    checkCudaErrors(cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));

    if (d_temp) checkCudaErrors(cudaFree(d_temp));
    if (d_result) chekcCudaErrors(cudaFree(d_result));

    return h_result;
}


float min_gpu(float* d_input, int numElems)
{
    return reduce<min, CUDART_INF_F>(d_input, numElmens);
}


float max_gpu(float* d_input, int numElems)
{
    float neg_inf = static_cast<float>(-2 ^ 16);
    return reduce<max, neg_inf>(d_input, numElmens);
}

__global__ void histogram_gpu(const float* const d_input, unsigned int* d_histo,
                            const float min_val, const float max_val, 
                            const int numElems, const int numBins)
{
    const float range = (max_val - min_val);
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (index < numElemes) {
        int bin = max(numBins - 1,
            (d_input[index] - min_val) / range * numBins);

        atomicAdd(d_histo[bin], 1);
    }
}

/*
 * inclusive scan
 */
__global__ void histogram_to_cdf(const int* const d_histo, int* d_cdf, const int numBins)
{
    
    extern __shared__ int shmem[];
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
    unsigned int* d_histo;
    float* d_min_logLum;
    float* d_max_logLum;
    float* h_min_logLum = new float;
    float* h_max_logLum = new float;

    //float logLumRange;
    checkCudaErrors(cudaMalloc(&d_histo, sizeof(unsigned int) * numBins));
    checkCudaErrors(cudaMalloc(&d_max_logLum, sizeof(float)* numElems));

    // how to set float as zero??????
    checkCudaErrors(cudaMemset(d_min_logLum, 0.0, sizeof(float)));

    float min_logLum = min_gpu(h_logLuminance, numElmens);
    float max_logLum = max_gpu(h_logLuminance, numElmens);

    histogram_gpu<<<(numElems-1)/1024, 1024>>>(d_logLuminance, d_histo, 
                                            min_logLum, max_logLum, 
                                            numElems, numBins);

    // numBins must be lower 1024.
    histogram_to_cdf<<<1, numBins, sizeof(int) * numBins*2 >>>(d_histo, d_cdf, numBins);


    if (d_histo) cudaFree(d_histo);
    if (d_min_logLum) cudaFree(d_min_logLum);
    if (d_max_logLum) cudaFree(d_max_logLum);
    if (h_min_logLum) delete h_min_logLum;
    if (h_max_logLum) delete h_max_logLum;
}