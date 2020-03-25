
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


#include <vector>
#include <algorithm>

#include <cuda.h>
#include "utils.h"


__device__
int bit_at(const int& value, const int& bit_index)
{
    return (value >> bit_index) & 1;
}


__global__ void negate(const int* const input, int* output_bitwise, const int bitIndex, const int numElems)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int bit;

    if (index < numElems) {
        bit = bit_at(input[index], bitIndex);

        output_bitwise[index] = (1-bit);
    }
}


__global__ void exclusive_scan_single_thread (const int* const input, int * output, const int numElems)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index != 0) return;

    output[0] = 0;

    for (int i = 1; i < numElems; ++i) {
        output[i] = output[i - 1] + input[i - 1];
    }
}

int get_total_false(int* d_negate, const int* const d_scan, const int numElems)
{
    int negate_value;
    int scan_value;
    const int index = numElems - 1;

    checkCudaErrors(cudaMemcpy(& negate_value, (d_negate + index), sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(& scan_value, (d_scan + index), sizeof(int), cudaMemcpyDeviceToHost));

    // equation: e[n-1] + f[n-1]
    return negate_value + scan_value;
}


__global__ void destination_address(const int* const bit_negate, const int* const d_scan, int* d_destination, const int totalFalses, const int numElems)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < numElems) {
        int bit = 1 - bit_negate[index];

        // equation: t = i -f + totalFalses
        if (bit) {
            d_destination[index] = index - d_scan[index] + totalFalses;
        }
        else {
            d_destination[index] = d_scan[index];
        }
    }
}


__global__ void scatter (const int* const input, int* output, const int* const d_destination, const int numElems)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int new_index;

    if (index < numElems) {
         new_index = d_destination[index];

        output[new_index] = input[index];
    }
}


void sort_gpu_multiple_threads(const int* input, int** output, const int numElems)
{
    // loop bits.
    const int n_bits = sizeof(int) * 8;
    const int n_data_bytes = sizeof(int) * numElems;

    const dim3 blockSize(1024);
    const dim3 gridSize((numElems - 1) / blockSize.x + 1);

    int* d_tmp_data;
    int* d_negate_bitwise;
    int* d_scan;
    int* d_destination;

    // copy input to temp data
    checkCudaErrors(cudaMalloc(&d_tmp_data, n_data_bytes));
    checkCudaErrors(cudaMemcpy(d_tmp_data, input, n_data_bytes, cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaMalloc(&d_negate_bitwise, n_data_bytes));
    checkCudaErrors(cudaMalloc(&d_scan, n_data_bytes));
    checkCudaErrors(cudaMalloc(&d_destination, n_data_bytes));

    int totalFalse;

    for (int i = 0; i < n_bits; i++) {

        // negate bit.
        negate << <gridSize, blockSize >> > (d_tmp_data, d_negate_bitwise, i, numElems);

        // scan bit with value 1.
        //exclusive_scan<<<girdSize, blockSize>>> (d_negate_bitwise, d_scan, numElems);
        exclusive_scan_single_thread<<<1, 1>>>(d_negate_bitwise, d_scan, numElems);

        // generate map with new positions.
        totalFalse = get_total_false(d_negate_bitwise, d_scan, numElems);
        destination_address<<<gridSize, blockSize>>> (d_negate_bitwise, d_scan, d_destination, totalFalse, numElems);

        // scatter data from input to output based on pre-defined map.
        scatter << <gridSize, blockSize >> > (d_tmp_data, *output, d_destination, numElems);


        // swap pointer
        int* tmp;
        tmp = *output;
        *output = d_tmp_data;
        d_tmp_data = tmp;
    }

    // swap pointer
    int* tmp;
    tmp = *output;
    *output = d_tmp_data;
    d_tmp_data = tmp;

    if (d_tmp_data) cudaFree(d_tmp_data);
    if (d_negate_bitwise) cudaFree(d_negate_bitwise);
    if (d_scan) cudaFree(d_scan);
    if (d_destination) cudaFree(d_destination);
}


template <typename T>
void sort_cpu_std(const T* const input, T* output, const int numElems)
{
    std::vector<T> input_vec(input, input + numElems);

    std::sort(input_vec.begin(), input_vec.end());

    std::copy(input_vec.begin(), input_vec.end(), output);
}


int main()
{
    const int numElems = 8;
    const int bytes = numElems * sizeof(unsigned int);

	unsigned int h_input[numElems] = { 4, 7, 2, 6, 3, 5, 1, 0  };
	unsigned int h_result[numElems];
	int* d_input;
	int* d_result;

    unsigned int ref_result[numElems];

    // allocate GPU memory.
    checkCudaErrors(cudaMalloc(&d_input, bytes));

    // move data from host to device.
    checkCudaErrors(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    // allocate output GPU memory.
    checkCudaErrors(cudaMalloc(&d_result, bytes));
    checkCudaErrors(cudaMemset(d_result, 0, bytes));

    // sort using radix sort and run on GPU.
    sort_gpu_multiple_threads(d_input, &d_result, numElems);
    checkCudaErrors(cudaMemcpy(h_result, d_result, bytes, cudaMemcpyDeviceToHost));

    // sort using C++ STD.
    sort_cpu_std(h_input, ref_result, numElems);

    std::cout << "CPU result = ";
    print_array(ref_result, numElems);

    std::cout << "GPU result = ";
    print_array(h_result, numElems);

    // compare CPU and GPU results.
    compare<unsigned int>(h_result, ref_result, numElems);


    if (d_input) cudaFree(d_input);
    if (d_result) cudaFree(d_result);

    return 0;
}
