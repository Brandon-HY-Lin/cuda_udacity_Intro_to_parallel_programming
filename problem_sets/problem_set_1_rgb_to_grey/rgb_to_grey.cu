
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "utils.h"

const unsigned int BLOCKSIZE = 1024;

__global__ void nvaive_rgba_to_greyscale(uchar4* d_rgbaImage,
	unsigned char* d_grayImage)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

    
	//uchar4* rgba = &(d_rgbaImage[index]);
	//float channelSum = .299f * rgba->x + .587f * rgba->y + .114f * rgba->z;
    
    uchar4 rgba = (d_rgbaImage[index]);
    float channelSum = .299f * rgba.x + .587f * rgba.y + .114f * rgba.z;

	d_grayImage[index] = channelSum;
}


void rgba_to_greyscale_gpu(const uchar4* const h_rgbaImage,
	uchar4* const d_rgbaImage,
	unsigned char* const d_greyImage,
	std::size_t numRows, std::size_t numCols)
{
	nvaive_rgba_to_greyscale <<<(numCols * numRows - 1) / BLOCKSIZE + 1, BLOCKSIZE >>> (d_rgbaImage, d_greyImage);
}