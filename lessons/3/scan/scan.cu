#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

template <typename T>
void compare(const T* const test, const T* const ref, const int size)
{
	for (std::size_t i = 0; i < size; ++i) {
		if (test[i] != ref[i]) {
			std::cerr << "Difference occurred at index " << i << std::endl;
			std::cerr << "Test value= " << test[i] << std::endl;
			std::cerr << "Ref value= " << ref[i] << std::endl;

			exit(EXIT_FAILURE);
		}
	}
}

template <typename T>
void print (const T* const data, const int size)
{
	for (std::size_t i = 0; i < size; ++i) {
		std::cout << data[i] << ", ";
	}

	std::cout << std::endl;
}

void scan_cpu( int* output, const  int* const input, const int size)
{
	output[0] = input[0];

	for (std::size_t i = 1; i < size; ++i) {
		output[i] = output[i-1] + input[i];
	}

	return;
}


__global__ void scan_gpu(int* output, int* input, int size)
{
	extern __shared__ int shmem[];
	int tid = threadIdx.x;
	int index_out = 0, index_in = 1;	// indicate input bank and output bank

	// move data to output shared memory
	//	the begining of for-loop will swap index
	shmem[index_out * size + tid] = input[tid];  // inclusive scan 
	//shmem[index_out * size + tid] = (tid > 0) ? input[tid-1]: 0;  // exclusive scan 
	__syncthreads();

	// scan
	for (int stride = 1; stride < size; stride *= 2) {
		
		index_out = 1 - index_out; // swap index
		index_in = 1 - index_out;	// swap index

		if (tid >= stride) {
			shmem[index_out * size + tid] = shmem[index_in * size + tid] + shmem[index_in * size + tid - stride];
		}
		else {
			shmem[index_out * size + tid] = shmem[index_in * size + tid];
		}

		__syncthreads();
	}

	// move data from shared memory to GPU memory
	output[tid] = shmem[index_out * size + tid];
}


int main()
{
	 int* h_input, * h_output;
	 int* d_input, *d_output;
	 int* output_reference;

	const int size = 8;
	const  int bytes = sizeof( int) * size;

	// alloc CPU mem
	h_input = new  int[size];
	h_output = new  int[size];
	output_reference = new  int[size];
	
	// init value
	std::memset(h_output, 0, bytes);

	for (std::size_t i = 0; i < size; ++i) {
		h_input[i] = i + 1;
	}

	// alloc GPU mem
	cudaMalloc(&d_input, bytes);
	cudaMalloc(&d_output, bytes);

	// init GPU data
	cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

	// inclusive sum scan using CPU
	scan_cpu(output_reference, h_input, size);

	// inclusive sum can using GPU
	int blockSize = size;
	scan_gpu<<<1, blockSize, bytes*2>>>(d_output, d_input, size);

	cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

	std::cout << "Input" << std::endl;
	print(h_input, size);

	std::cout << "Reference" << std::endl;
	print(output_reference, size);

	std::cout << "GPU" << std::endl;
	print(h_output, size);

	// compare data
	compare< int>(h_output, output_reference, size);


	return 0;
}
