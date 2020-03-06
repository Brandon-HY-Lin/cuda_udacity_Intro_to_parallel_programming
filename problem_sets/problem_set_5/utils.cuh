#ifndef UTILS_H__
#define UTILS_H__
#include <iostream>
#include <iomanip>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cassert>
#include <cmath>
#define checkCudaErrors(val)	check( (val), #val, __FILE__, __LINE__)

template <typename T>
void check(T err, const char* const func, const char* const file, const int line)
{

	if (err != cudaSuccess) {
		std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
		std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
		exit(1);
	}
}

template <typename T>
void checkResultsExact(const T* const ref, const T* const gpu, std::size_t numElem)
{
	// check that the GPU result matches the CPU result
	for (std::size_t i = 0; i < numElem; ++i) {
		if (ref[i] != gpu[i]) {
			std::cerr << "Difference at pos " << i << std::endl;
			// the + is magic to convert char to int without messing
			// with other types
			std::cerr << "Reference: " << std::setprecision(17) << +ref[i] <<
				"\nGPU      : " << +gpu[i] << std::endl;
			exit(1);
		}
	}
}
#endif /* UTILS_H__ */