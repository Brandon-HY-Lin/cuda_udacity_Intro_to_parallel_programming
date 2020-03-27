
#include <cuda.h>
#include <cuda_runtime.h>
#include "utils.h"

template <typename T = unsigned int>
void swap_address(T** p_a, T** p_b)
{
    T* tmp;
    tmp = *p_a;
    *p_a = *p_b;
    *p_b = tmp;
}

__device__ int bit_at(const unsigned int value, const int bitIndex)
{
    return (value >> bitIndex) & 1;
}

__global__ void negate(const unsigned int* const d_input, unsigned int* d_negate, const int bitIndex, const int numElems)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < numElems) {
        int bit = bit_at(d_input[index], bitIndex);

        d_negate[index] = (1 - bit);
    }
}


__global__ void exclusive_scan_single_thread (const unsigned int* const d_input, unsigned int* d_scan, const int numElems)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index != 0) return;

    d_scan[0] = 0;

    for (int i = 1; i < numElems; ++i) {
        d_scan[i] = d_scan[i - 1] + d_input[i - 1];
    }
}

unsigned int get_total_false(const unsigned int* const d_negate, const unsigned int* const d_scan, const int numElems)
{
    int index = numElems - 1;
    int negate_value;
    int scan_value;

    checkCudaErrors(cudaMemcpy(&negate_value, d_negate + index, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&scan_value, d_scan + index, sizeof(unsigned int), cudaMemcpyDeviceToHost));

    // equation: e[n-1] + f[n-1]
    return negate_value + scan_value;

}


__global__ void destination_address (const unsigned int* const d_bit_negate, const unsigned int* const d_scan, unsigned int* d_destination, const int totalFalse, const int numElems)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < numElems) {
        int bit = 1 - d_bit_negate[index];

        if (bit) {
            d_destination[index] = index - d_scan[index] + totalFalse;
        }
        else {
            d_destination[index] = d_scan[index];
        }
    }
}


__global__ void scatter (const unsigned int* const input, unsigned int * output, const unsigned int* const d_destination, const int numElems)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < numElems) {
        int new_index = d_destination[index];

        output[new_index] = input[index];
    }
}

void sort_cuda(unsigned int* const d_inputVals,
                unsigned int* const d_inputPos,
                unsigned int* const d_outputVals,
                unsigned int* const d_outputPos,
                const size_t numElems)
{
    const int bytes_data = sizeof(unsigned int) * numElems;
    const int n_bits = sizeof(unsigned int) * 8;    // number of bits of unsigned int.

    const dim3 blockSize(1024);
    const dim3 gridSize((numElems - 1) / blockSize.x + 1);

    unsigned int* d_inputVals_tmp;
    unsigned int* d_outputVals_tmp = d_outputVals;

    unsigned int* d_inputPos_tmp;
    unsigned int* d_outputPos_tmp = d_outputPos;

    unsigned int* d_negate;
    unsigned int* d_scan;
    unsigned int* d_destination;


    // allocate temp memory of val
    checkCudaErrors(cudaMalloc(&d_inputVals_tmp, bytes_data));
    checkCudaErrors(cudaMalloc(&d_inputPos_tmp, bytes_data));
    checkCudaErrors(cudaMalloc(&d_negate, bytes_data));
    checkCudaErrors(cudaMalloc(&d_scan, bytes_data));
    checkCudaErrors(cudaMalloc(&d_destination, bytes_data));

    // copy input to output
    checkCudaErrors(cudaMemcpy(d_inputVals_tmp, d_inputVals, bytes_data, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(d_inputPos_tmp, d_inputPos, bytes_data, cudaMemcpyDeviceToDevice));

    // swap input/ouput first, because it will be swap at the begining of for-loop.
    swap_address(&d_inputVals_tmp, &d_outputVals_tmp);
    swap_address(&d_inputPos_tmp, &d_outputPos_tmp);

    // iterate bits of d_inputVals.
    for (int i = 0; i < n_bits; ++i) {

        // swap pointer
        swap_address(&d_inputVals_tmp, &d_outputVals_tmp);
        swap_address(&d_inputPos_tmp, &d_outputPos_tmp);

        // negate bits.
        negate << <gridSize, blockSize >> > (d_inputVals_tmp, d_negate, i, numElems);

        // scan bit of negated bits which value is 1.
        exclusive_scan_single_thread << <1, 1 >> > (d_negate, d_scan, numElems);

        // get total number of bits with value is 0
        int totalFalse = get_total_false(d_negate, d_scan, numElems);

        // calculate destination address.
        destination_address << <gridSize, blockSize >> > (d_negate, d_scan, d_destination, totalFalse, numElems);

        // scatter d_inputVals to d_outputVals.
        scatter << <gridSize, blockSize >> > (d_inputVals_tmp, d_outputVals_tmp, d_destination, numElems);

        // scatter d_inputPos to d_outputPos.
        scatter << <gridSize, blockSize >> > (d_inputPos_tmp, d_outputPos_tmp, d_destination, numElems);
    }
       
        
    if (d_inputVals_tmp) cudaFree(d_inputVals_tmp);
    if (d_inputPos_tmp) cudaFree(d_inputPos_tmp);
    if (d_negate) cudaFree(d_negate);
    if (d_scan) cudaFree(d_scan);
    if (d_destination) cudaFree(d_destination);
}