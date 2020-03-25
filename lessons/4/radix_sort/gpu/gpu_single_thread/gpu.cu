
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


#include <vector>
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>
#include "utils.h"


#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>

int* d_histo;
int* d_cdf;
unsigned int* d_tmp_data;
int bin_bytes;


void reset_histo()
{
    checkCudaErrors(cudaMemset(d_histo, 0, bin_bytes));
}

void allocate_memory(const unsigned int numBins, const unsigned int numElems)
{

    bin_bytes = sizeof(int) * numBins;
    checkCudaErrors(cudaMalloc(&d_histo, bin_bytes));
    checkCudaErrors(cudaMemset(d_histo, 0, bin_bytes));

    checkCudaErrors(cudaMalloc(&d_cdf, bin_bytes));
    checkCudaErrors(cudaMemset(d_cdf, 0, bin_bytes));

    const int data_size = sizeof(unsigned int) * numElems;
    checkCudaErrors(cudaMalloc(&d_tmp_data, data_size));
    checkCudaErrors(cudaMemset(d_tmp_data, 0, data_size));
}


void free_memory()
{
    if (d_histo) cudaFree(d_histo);
    if (d_cdf) cudaFree(d_cdf);
    if (d_tmp_data) cudaFree(d_tmp_data);
}


__device__
unsigned int bit_at(const unsigned int& value, const int bit_index)
{
    return (value >> bit_index) & 1;
}


__global__
void histogram_naive(int* histo, const unsigned int* data, const int bit_index, const int num_elems)
{
    int bit;
    
    for (int i = 0; i < num_elems; ++i) {
        bit = bit_at(data[i], bit_index);
        atomicAdd(&histo[bit], 1);
    }
}

/*
 * bit-wise histogram only has 2 value: 0 or 1
 */
__global__
void histo_to_cdf(int* cdf, const int* histo)
{
    cdf[0] = 0;	// exclusive scan

    cdf[1] = histo[0];
}


__global__
void move(unsigned int* output, const unsigned int* data, const int bit_index, int* cdf, const int num_elems)
{
    int bit;
    int index;

    for (int i = 0; i < num_elems; ++i) {
        bit = bit_at(data[i], bit_index);
        index = cdf[bit];

        output[index] = data[i];

        atomicAdd(&cdf[bit], 1);
    }
}

/*
 * address of data might be changed.
 */
void count_sort(unsigned int** data, const int bit_index, const int numElems)
{

    reset_histo();

    // calculate histogram
    histogram_naive << <1, 1 >> > (d_histo, *data, bit_index, numElems);

	// calculate CDF based on histogram (exclusive)
	histo_to_cdf<<<1, 1>>>(d_cdf, d_histo);

    // map
    move << <1, 1 >> > (d_tmp_data, *data, bit_index, d_cdf, numElems);

    // swap memory
    unsigned int* tmp;
    tmp = d_tmp_data;
    d_tmp_data = *data;
    *data = tmp;	// change address of *data to d_temp_data.
}


/*
 * Use Cuda to sort data.
 */
void sort_gpu(const unsigned int* const input, unsigned int** output, const int numElems)
{
    const int totalBits = 8 * sizeof(unsigned int);

    // copy input to output.
    checkCudaErrors(cudaMemcpy(*output, input, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));

    // allocate temporary memory.
    allocate_memory(2, numElems);

    // iterate bits
    for (int i = 0; i < totalBits; i++) {
        // count sort.
        count_sort(output, i, numElems);


    }

    // free temporary memory.
    free_memory();
}


/*
 * Use Nvidia Thurst library to sort data.
 */
void sort_thrust_v1(const unsigned int* const input, unsigned int* output, const int numElems)
{
    thrust::device_vector<unsigned int> input_vec(input, input + numElems);

    thrust::sort(input_vec.begin(), input_vec.end());

    thrust::copy(input_vec.begin(), input_vec.end(), output);
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

	unsigned int h_input[numElems] = { 1 << 8, 1 << 2, 1 << 3, 1 << 2, 1 << 10, 1 << 5, 1 << 12, 1<<0 };  //{ 170, 45, 75, 90, 802, 24, 2, 66 };
	unsigned int h_result[numElems];
	unsigned int* d_input;
	unsigned int* d_result;

    unsigned int ref_result[numElems];

    // allocate GPU memory.
    checkCudaErrors(cudaMalloc(&d_input, bytes));

    // move data from host to device.
    checkCudaErrors(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    // allocate output GPU memory.
    checkCudaErrors(cudaMalloc(&d_result, bytes));
    checkCudaErrors(cudaMemset(d_result, 0, bytes));

    // sort using radix sort and run on GPU.
    sort_gpu(d_input, &d_result, numElems);
    checkCudaErrors(cudaMemcpy(h_result, d_result, bytes, cudaMemcpyDeviceToHost));

    //sort_thrust_v1(d_input, h_result, numElems);

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
