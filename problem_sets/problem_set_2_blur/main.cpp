
#include <opencv2/opencv.hpp>
#include "utils.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include "timer.h"
#include "blur_gpu.cuh"
#include "reference_cal.h"
#include "compare.h"

uchar4* d_image_in__, *d_image_out__, *h_image_in__, *h_image_out__;
cv::Mat imageInputRGBA;
cv::Mat imageOutputRGBA;

uchar4 *d_inputImageRGBA__;
uchar4 *d_outputImageRGBA__;

float *h_filter__;

std::size_t numRows() { return imageInputRGBA.rows; }
std::size_t numCols() { return imageInputRGBA.cols; }

//return types are void since any internal error will be handled by quitting
//no point in returning error codes...
//returns a pointer to an RGBA version of the input image
//and a pointer to the single channel grey-scale output
//on both the host and device
void preProcess(uchar4 **h_inputImageRGBA, uchar4 **h_outputImageRGBA,
                uchar4 **d_inputImageRGBA, uchar4 **d_outputImageRGBA,
                unsigned char **d_redBlurred,
                unsigned char **d_greenBlurred,
                unsigned char **d_blueBlurred,
                float **h_filter, int *filterWidth,
                const std::string &filename) 
{
  //make sure the context initializes ok
  checkCudaErrors(cudaFree(0));
  
	// read image file to CPU memory
	cv::Mat image = cv::imread(filename.c_str(), cv::IMREAD_COLOR);

	if (image.empty()) {
    	std::cerr << "Couldn't open file: " << filename << std::endl;
		exit(EXIT_FAILURE);
	}

	// convert BGR TO RGBA
	cv::cvtColor(image, imageInputRGBA, cv::COLOR_BGR2RGBA);
	// allocate memory for the output
	imageOutputRGBA.create(image.rows, image.cols, CV_8UC4);
	if (!imageInputRGBA.isContinuous() || !imageOutputRGBA.isContinuous()) {
    	std::cerr << "Images aren't continuous!! Exiting." << std::endl;
    	exit(EXIT_FAILURE);
  	}
	
	// alloc CPU mem
	*h_inputImageRGBA = (uchar4 *)imageInputRGBA.ptr<unsigned char>(0);
	*h_outputImageRGBA = (uchar4 *)imageOutputRGBA.ptr<unsigned char>(0);
	// alloc GPU mem
	const std::size_t numPixels = numRows() * numCols();
  //allocate memory on the device for both input and output
  checkCudaErrors(cudaMalloc(d_inputImageRGBA, sizeof(uchar4) * numPixels));
  checkCudaErrors(cudaMalloc(d_outputImageRGBA, sizeof(uchar4) * numPixels));
  checkCudaErrors(cudaMemset(*d_outputImageRGBA, 0, numPixels * sizeof(uchar4))); //make sure no memory is left laying around

  //copy input array to the GPU
  checkCudaErrors(cudaMemcpy(*d_inputImageRGBA, *h_inputImageRGBA, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));

  d_inputImageRGBA__  = *d_inputImageRGBA;
  d_outputImageRGBA__ = *d_outputImageRGBA;

  //now create the filter that they will use
  const int blurKernelWidth = 9;
  const float blurKernelSigma = 2.;

  *filterWidth = blurKernelWidth;

	// create and fill the filter we will convolve with
	*h_filter = new float[blurKernelWidth * blurKernelWidth];
	h_filter__ = *h_filter;

	float filterSum = 0.f; // for normalization

	// init filter
	for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) {
		for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) {
	      float filterValue = expf( -(float)(c * c + r * r) / (2.f * blurKernelSigma * blurKernelSigma));
	      (*h_filter)[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] = filterValue;
	      filterSum += filterValue;
	    }
	  }

	float normalizationFactor = 1.f / filterSum;

	// normalize filter
	for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) {
		for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) {
	  		(*h_filter)[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] *= normalizationFactor;
		}
	}
  //blurred
  checkCudaErrors(cudaMalloc(d_redBlurred,    sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMalloc(d_greenBlurred,  sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMalloc(d_blueBlurred,   sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMemset(*d_redBlurred,   0, sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMemset(*d_greenBlurred, 0, sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMemset(*d_blueBlurred,  0, sizeof(unsigned char) * numPixels));
}

void save_image (const std::string& output_file, uchar4* data_ptr)
{
	cv::Mat output(numRows(), numCols(), CV_8UC4, (void*) data_ptr);

	cv::Mat imageOutputBGR;
	cv::cvtColor(output, imageOutputBGR, cv::COLOR_RGBA2BGR);
	// output the image
	cv::imwrite(output_file.c_str(), imageOutputBGR);
}
void cleanUp(void)
{
  cudaFree(d_inputImageRGBA__);
  cudaFree(d_outputImageRGBA__);
  delete[] h_filter__;
}

int main(int argc, char **argv)
{
  uchar4 *h_inputImageRGBA,  *d_inputImageRGBA;
  uchar4 *h_outputImageRGBA, *d_outputImageRGBA;
  unsigned char *d_redBlurred, *d_greenBlurred, *d_blueBlurred;

  float *h_filter;
  int    filterWidth;

  std::string input_file;
  std::string output_file;
  std::string reference_file;
  double perPixelError = 1.0f;
  double globalError   = 0.0;
  bool useEpsCheck = false;
  
	if (argc == 1) {
		input_file = "cinque_terre_small.jpg";
	}
	else {
		input_file = std::string(argv[1]);
	}

	output_file = "output_gpu.png";
	reference_file = "output_cpu.png";
	
  //load the image and give us our input and output pointers
  preProcess(&h_inputImageRGBA, &h_outputImageRGBA, &d_inputImageRGBA, &d_outputImageRGBA,
             &d_redBlurred, &d_greenBlurred, &d_blueBlurred,
             &h_filter, &filterWidth, input_file);
	
  allocateMemoryAndCopyToGPU(numRows(), numCols(), h_filter, filterWidth);
  GpuTimer timer;
  timer.Start();
  blur_gpu(h_inputImageRGBA, d_inputImageRGBA, d_outputImageRGBA, numRows(), numCols(),
                     d_redBlurred, d_greenBlurred, d_blueBlurred, filterWidth);
  timer.Stop();
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  int err = printf("Your code ran in: %f msecs.\n", timer.Elapsed());

	// move image from device to host
  size_t numPixels = numRows()*numCols();
  //copy the output back to the host
  checkCudaErrors(cudaMemcpy(h_outputImageRGBA, d_outputImageRGBA__, sizeof(uchar4) * numPixels, cudaMemcpyDeviceToHost));

	// save GPU image
	save_image(output_file, h_outputImageRGBA);

  blur_cpu(h_inputImageRGBA, h_outputImageRGBA,
                       numRows(), numCols(),
                       h_filter, filterWidth);
	// save CPU image
	save_image(reference_file, h_outputImageRGBA);

  compareImages(reference_file, output_file, useEpsCheck, perPixelError, globalError);

  checkCudaErrors(cudaFree(d_redBlurred));
  checkCudaErrors(cudaFree(d_greenBlurred));
  checkCudaErrors(cudaFree(d_blueBlurred));

  cleanUp();

	return 0;
}
