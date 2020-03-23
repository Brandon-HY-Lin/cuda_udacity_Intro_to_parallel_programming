#ifndef UTILS_H__
#define UTILS_H__

#include <iostream>
#include <vector>

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
	if (err != cudaSuccess) {
		std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
		std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
		exit(1);
	}
}

template <typename T>
void print_array(const T& array, int size)
{
	for (int i = 0; i < size; i++) {
		std::cout << array[i] << ", ";
	}

	std::cout << std::endl;
}


template <typename T>
void compare (const std::vector<T>& values, const std::vector<T>& ref_values)
{
	if (values.size() != ref_values.size()) {
		std::cerr << "Mismatch size of values and ref_values on line " << __func__ << std::endl;
		exit(EXIT_FAILURE);
	}

	for (std::size_t i = 0; i < values.size(); ++i) {
		if (values[i] != ref_values[i]) {
			std::cerr << "Mismatch value at index " << i << std::endl;
			std::cerr << "value= " << values[i] << ", ref_value= " << ref_values[i] << std::endl;
			exit(EXIT_FAILURE);
		}
	}

	std::cout << "Pass" << std::endl;
}

template <typename T>
void compare(const T* values, const T* ref_values, const int size)
{
	for (std::size_t i = 0; i < size; ++i) {
		if (values[i] != ref_values[i]) {
			std::cerr << "Mismatch value at index " << i << std::endl;
			std::cerr << "value= " << values[i] << ", ref_value= " << ref_values[i] << std::endl;
			exit(EXIT_FAILURE);
		}
	}

	std::cout << "Pass" << std::endl;
}

#endif /*UTILS_H__*/