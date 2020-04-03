#include <iostream>
#include "timer.h"
#include "utils.h"
#include <string>
#include <stdio.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "blend_cpu.h"
#include "blend_gpu.cuh"
#include "compare.h"
#include "processing.h"

int main(int argc, char** argv)
{
	uchar4 *h_sourceImg, *h_destImg, *h_blendedImg;
	std::size_t numRowsSource, numColsSource;

	std::string input_source_file;
	std::string input_dest_file;
	std::string output_file;
	
	std::string reference_file;
	double perPixelError = 0.0;
	double globalError = 0.0;
	bool useEpsCheck = false;
	switch (argc) {
	case 1:
		input_source_file = "source.png";
		input_dest_file = "destination.png";
		output_file = "output.png";
		reference_file = "output_cpu.png";
		break;
	case 3:
		input_source_file = std::string(argv[1]);
		input_dest_file = std::string(argv[2]);
		output_file = "output.png";
		reference_file = "output_cpu.png";
		break;
  	default:
        std::cerr << "Usage: ./problem_set_6_seamless_clone input_source_file input_dest_filename" << std::endl;
        exit(1);
    }

	// load the image and give us our input and output pointers
	preProcess(&h_sourceImg, numRowsSource, numColsSource,
				&h_destImg,
				&h_blendedImg, input_source_file, input_dest_file);
				
	GpuTimer timer;
	timer.Start();

	// using GPU to blend images.
	blend_gpu(h_sourceImg, numRowsSource, numColsSource,
				h_destImg,
				h_blendedImg);

	timer.Stop();
  	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  	printf("Your code ran in: %f msecs.\n", timer.Elapsed());
	// check results and output the tone-mapped image
	postProcess(h_blendedImg, numRowsSource, numColsSource, output_file);
	
	// calculate the reference image
	uchar4* h_reference = new uchar4[numRowsSource * numColsSource];
	blend_cpu(h_sourceImg, numRowsSource, numColsSource,
				h_destImg, h_reference);
				
	// save the blended image using CPU.
	postProcess(h_reference, numRowsSource, numColsSource, reference_file);
	
	compareImages(reference_file, output_file, useEpsCheck, perPixelError, globalError);
	
	delete[] h_reference;
	delete[] h_destImg;
	delete[] h_sourceImg;
	delete[] h_blendedImg;
	return 0;
}