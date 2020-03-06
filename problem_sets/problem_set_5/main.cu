
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <cstdlib>
#include <iostream>
#include <cstdio>

#include "utils.cuh"
#include "timer.cuh"
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/normal_distribution.h>
#include "reference_calc.h"

#include <algorithm>

#if defined(_WIN16) || defined(_WIN32) || defined(_WIN64)
#include <Windows.h>
#else
#include <sys/time.h>
#endif

void computeHistogram(const unsigned int* const d_vals,
	unsigned int* const d_histo,
	const unsigned int numBins,
	const unsigned int numElems);

int main(int argc, char** argv)
{


	const unsigned int numBins = 1024;
	const unsigned int numElems = 10000 * numBins;
	const float stddev = 100.f;

	unsigned int* vals = new unsigned int[numElems];
	unsigned int* h_vals = new unsigned int[numElems];
	unsigned int* h_histo = new unsigned int[numBins];
	unsigned int* h_refHisto = new unsigned int[numBins];

#if defined(_WIN16) || defined(_WIN32) || defined(_WIN64)
	srand(GetTickCount());
#else
	timeval tv;
	gettimeofday(&tv, NULL);

	srand(tv.tv_usec);
#endif

	// make the mean unpredictable, but close enough to the middle
	// so that timings are unaffected.
	unsigned int mean = rand() % 100 + 462;


	// init random number using GPU (thrust libray)
	// 		the range of number is in [0..numBins-1]
	thrust::minstd_rand rng;

	thrust::random::normal_distribution<float> normalDist((float)mean, stddev);

	// Generate the random values
	for (std::size_t i = 0; i < numElems; ++i) {
		vals[i] = std::min(
			(unsigned int)std::max((int)normalDist(rng), 0), // lower bound = 0
			numBins - 1);					// upper bound = (numBins -1)

	}


	unsigned int* d_vals, * d_histo;

	GpuTimer timer;

	// alloc GPU memory
	// checkCudaErros() prints caller name and parse cudaError_t to string.
	checkCudaErrors(cudaMalloc(&d_vals, sizeof(unsigned int) * numElems));
	checkCudaErrors(cudaMalloc(&d_histo, sizeof(unsigned int) * numBins));

	// set d_histo to all zeros
	checkCudaErrors(cudaMemset(d_histo, 0, sizeof(unsigned int) * numBins));

	// move raw data from host to device
	checkCudaErrors(cudaMemcpy(d_vals, vals, sizeof(unsigned int) * numElems, cudaMemcpyHostToDevice));

	timer.Start();
	computeHistogram(d_vals, d_histo, numBins, numElems);
	timer.Stop();
	int err = printf("Your code ran in: %f msecs.\n", timer.Elapsed());

	if (err < 0) {
		// Couldn't print! probably the student closed stdout - bad news
		std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
		exit(1);
	}
	// move histogram in GPU to CPU
	checkCudaErrors(cudaMemcpy(h_histo, d_histo, sizeof(unsigned int) * numBins, cudaMemcpyDeviceToHost));

	// generate reference for the given mean
	reference_calculation(vals, h_refHisto, numBins, numElems);

	// check equivalence of CPU and GPU versions.
	checkResultsExact(h_refHisto, h_histo, numBins);


	// free CPU heap memory and GPU memory.
	if (h_vals) delete[] h_vals;
	if (h_refHisto) delete[] h_refHisto;
	if (h_histo) delete[] h_histo;
	if (d_vals) cudaFree(d_vals);
	if (d_histo) cudaFree(d_histo);

	return 0;
}
