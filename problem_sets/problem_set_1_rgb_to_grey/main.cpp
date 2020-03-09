#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "utils.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <iostream>
#include "timer.h"
#include <stdio.h>
#include "rgb_to_grey.cuh"
#include "reference_cal.h"
#include "compare.h"

void rgba_to_greyscale_gpu(const uchar4* const h_rgbaImage,
	uchar4* const d_rgbaImage,
	unsigned char* const d_greyImage,
	std::size_t numRows, std::size_t numCols);

cv::Mat imageRGBA;
cv::Mat imageGrey;

uchar4* d_rgbaImage__;
unsigned char* d_greyImage__;

std::size_t numRows() { return imageRGBA.rows; }
std::size_t numCols() { return imageRGBA.cols; }

// return types are void since any internal error will be handled by quitting
// no point in returning error codes...
// returns a pointer to an RGBA version of the input image
// and a pointer to the single channel grey-scale output
// on both the host and device
void preProcess(uchar4** inputImage, unsigned char** greyImage,
	uchar4** d_rgbaImage, unsigned char** d_greyImage,
	const std::string& filename)
{
	// make sure the context initializes ok
	checkCudaErrors(cudaFree(0));

	cv::Mat image;
	image = cv::imread(filename.c_str(), cv::IMREAD_COLOR);
	if (image.empty()) {
		std::cerr << "Couldn't open file: " << filename << std::endl;
		exit(1);
	}

	cv::cvtColor(image, imageRGBA, cv::COLOR_BGR2RGBA);

	// allocate memory for the output
	imageGrey.create(image.rows, image.cols, CV_8UC1);

	// This shouldn't ever happen given the way the images are created
	// at least based upon my limited understanding of OpenCV, but better to check
	if (!imageRGBA.isContinuous() || !imageGrey.isContinuous()) {
		std::cerr << "Images aren't continuous!! Exiting." << std::endl;
		exit(1);
	}

	// Get pointer of 1st row
	*inputImage = (uchar4*)imageRGBA.ptr<unsigned char>(0);
	// Get pointer of 1st row
	*greyImage = imageGrey.ptr<unsigned char>(0);

	const std::size_t numPixels = numRows() * numCols();

	// allocate memory on the device for both input and output
	checkCudaErrors(cudaMalloc(d_rgbaImage, sizeof(uchar4) * numPixels));
	checkCudaErrors(cudaMalloc(d_greyImage, sizeof(unsigned char) * numPixels));

	// set gray image all zeros.
	checkCudaErrors(cudaMemset(*d_greyImage, 0, numPixels * sizeof(unsigned char))); // make sure no memory is left laying around

	// copy input array to the GPU
	checkCudaErrors(cudaMemcpy(*d_rgbaImage, *inputImage, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));

	d_rgbaImage__ = *d_rgbaImage;
	d_greyImage__ = *d_greyImage;
}

void save_image(const std::string& output_file, unsigned char* data_ptr)
{
	cv::Mat output(numRows(), numCols(), CV_8UC1, data_ptr);

	// save image
	cv::imwrite(output_file.c_str(), output);
}


void cleanup()
{
	// free GPU mem
	cudaFree(d_rgbaImage__);
	cudaFree(d_greyImage__);
}


int main(int argc, char** argv)
{
	uchar4* h_rgbaImage, * d_rgbaImage;
	unsigned char* h_greyImage, * d_greyImage;

	std::string input_file;
	std::string output_file;
	std::string reference_file;

	double perPixelError = 1.0f;
	double globalError = 0.0f;
	bool useEpsCheck = true;

	if (argc == 1) {
		input_file = "cinque_terre_small.jpg";
	}
	else {
		input_file = std::string(argv[1]);
	}

	output_file = "output_gpu.png";
	reference_file = "output_cpu.png";

	// read bgr image and convert to rgb image
	preProcess(&h_rgbaImage, &h_greyImage, &d_rgbaImage, &d_greyImage, input_file);


	GpuTimer timer;
	timer.Start();

	rgba_to_greyscale_gpu(h_rgbaImage, d_rgbaImage, d_greyImage, numRows(), numCols());
	timer.Stop();

	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	printf("Code ran in: %f msecs.\n", timer.Elapsed());

	std::size_t numPixels = numRows() * numCols();
	checkCudaErrors(cudaMemcpy(h_greyImage, d_greyImage, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost));

	// write grey image
	save_image(output_file, h_greyImage);

	rgba_to_greyscale_cpu(h_rgbaImage, h_greyImage, numRows(), numCols());

	save_image(reference_file, h_greyImage);

	compareImages(reference_file, output_file, useEpsCheck, perPixelError,
		globalError);


	cleanup();

	return 0;
}