#include <iostream>
#include "timer.h"
#include "utils.h"
#include <string>
#include <stdio.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "compare.h"
#include "processing.cuh"
#include "sort_gpu.cuh"
#include "sort_cpu.h"

int main (int argc, char **argv)
{
	unsigned int *inputVals;
	unsigned int *inputPos;
	unsigned int *outputVals;
	unsigned int *outputPos;
	
	std::size_t numElems;
	
	std::string input_file;
	std::string template_file;
	std::string output_file;
	std::string reference_file;
	double perPixelError = 0.0;
	double globalError = 0.0;
	bool useEpsCheck = false;
	
	if (argc == 1) {
		input_file = "red_eye_effect_5.jpg";
		template_file = "red_eye_effect_template_5.jpg";
		output_file = "output.png";
	} else if (argc == 3) {
		input_file  = std::string(argv[1]);
      	template_file = std::string(argv[2]);
	  	output_file = "HW4_output.png";
	} else {
		std::cerr << "Usage: ./HW4 input_file template_file [output_filename]" << std::endl;
          exit(EXIT_FAILURE);
	}

	// load the image and give us our input and output pointers
	preProcess(&inputVals, &inputPos, &outputVals, &outputPos, numElems, input_file, template_file);

  GpuTimer timer;
  timer.Start();


	sort_cuda(inputVals, inputPos, outputVals, outputPos, numElems);

  timer.Stop();
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  printf("\n");
  printf("Your code ran in: %f msecs.\n", timer.Elapsed());


	// check results and output the red-eye corrected image
	postProcess(outputVals, outputPos, numElems, output_file);
	
	thrust::device_ptr<unsigned int> d_inputVals(inputVals);
	thrust::device_ptr<unsigned int> d_inputPos(inputPos);

	// copy device_vector to host_vector
	thrust::host_vector<unsigned int> h_inputVals(d_inputVals,
													d_inputVals + numElems);
	thrust::host_vector<unsigned int> h_inputPos(d_inputPos,
													d_inputPos + numElems);

	thrust::host_vector<unsigned int> h_outputVals(numElems);
	thrust::host_vector<unsigned int> h_outputPos(numElems);
	
	sort_cpu(&h_inputVals[0], &h_inputPos[0],
							&h_outputVals[0], &h_outputPos[0],
							numElems);


  //postProcess(valsPtr, posPtr, numElems, reference_file);

  //compareImages(reference_file, output_file, useEpsCheck, perPixelError, globalError);

	thrust::device_ptr<unsigned int> d_outputVals(outputVals);
	thrust::device_ptr<unsigned int> d_outputPos(outputPos);
	
	thrust::host_vector<unsigned int> h_yourOutputVals(d_outputVals,
													d_outputVals + numElems);
	thrust::host_vector<unsigned int> h_yourOutputPos(d_outputPos,
													d_outputPos + numElems);

	checkResultsExact(&h_outputVals[0], &h_yourOutputVals[0], numElems);
	checkResultsExact(&h_outputPos[0], &h_yourOutputPos[0], numElems);

	checkCudaErrors(cudaFree(inputVals));
  	checkCudaErrors(cudaFree(inputPos));
  	checkCudaErrors(cudaFree(outputVals));
  	checkCudaErrors(cudaFree(outputPos));

	return 0;
}