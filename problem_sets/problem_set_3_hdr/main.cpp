
#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <algorithm>
#include "timer.h"
#include "utils.h"
#include <algorithm>
#include "compare.h"
#include "cdf_cpu.h"
#include "cdf_gpu.cuh"
#include "processing.cuh"
int main (int argc, char **argv)
{
	float *d_luminance;
	unsigned int *d_cdf;
	
	std::size_t numRows, numCols;
	unsigned int numBins;
	
	std::string input_file;
	std::string output_file;
	std::string reference_file;
	double perPixelError = 0.0f;
	double globalError = 0.0f;
	bool useEpsCheck = false;
	if (argc == 1) {
		input_file = "memorial_raw_large.png";
	}
	else {
		input_file = argv[1];
	}
	
	output_file = "output_gpu.png";
	reference_file = "output_cpu.png";
	

	
	// load the imge and give us input and output pointers
	// read image then allocate GPU and CPU memory
	preProcessGPU(&d_luminance, &d_cdf,
				&numRows, &numCols, &numBins, input_file);

	GpuTimer timer;
	float min_logLum, max_logLum;
	min_logLum = 0.f;
	max_logLum = 1.f;
	timer.Start();
	
	cdf_gpu(d_luminance, d_cdf, min_logLum, max_logLum,
			numRows, numCols, numBins);
	timer.Stop();
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	printf("Code ran in: %f msecs. \n", timer.Elapsed());
	// generate un-normalized CDF

	float *h_luminance = (float *) malloc(sizeof(float) * numRows * numCols);
	unsigned int *h_cdf = (unsigned int *) malloc(sizeof(unsigned int) * numBins);
	
	
  	checkCudaErrors(cudaMemcpy(h_luminance, d_luminance, numRows*numCols*sizeof(float), cudaMemcpyDeviceToHost));
	
	// normalize CDF and perform tone-mapping
	postProcessGPU(output_file, numRows, numCols, min_logLum, max_logLum);

	for (std::size_t i = 1; i < numCols * numRows; ++i) {
		min_logLum = std::min(h_luminance[i], min_logLum);
		max_logLum = std::max(h_luminance[i], max_logLum);
	}
	
	// generate unnormalized CDF
	cdf_cpu(h_luminance, h_cdf, numRows, numCols, numBins, min_logLum, max_logLum);

  	checkCudaErrors(cudaMemcpy(d_cdf, h_cdf, sizeof(unsigned int) * numBins, cudaMemcpyHostToDevice));

	// normalize CDF and perform tone-mapping
	postProcessGPU(reference_file, numRows, numCols, min_logLum, max_logLum);

	cleanupGlobalMemory();
	
	compareImages(reference_file, output_file, useEpsCheck, perPixelError, globalError);
	return 0;
}
