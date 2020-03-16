#include <string>
#include <stdio.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "compare.h"
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
		template_fie = "red_eye_effect_template_5.jpg";
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
	thrust::device_ptr<unsigned int> d_inputVals(inputVals);
	thrust::device_ptr<unsigned int> d_inputPos(inputPos);

	// copy device_vector to host_vector
	thrust::host_vector<unsigned int> h_inputVals(d_inputVals,
													d_inputVals + numElems);
	thrust::host_vector<unsigned int> h_inputPos(d_inputPos,
													d_inputPos + numElems);
	checkCudaErrors(cudaFree(inputVals));
  	checkCudaErrors(cudaFree(inputPos));
  	checkCudaErrors(cudaFree(outputVals));
  	checkCudaErrors(cudaFree(outputPos));

	return 0;
}