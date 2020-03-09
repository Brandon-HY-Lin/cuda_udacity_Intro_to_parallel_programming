// for uchar4 struct
#include <iostream>
#include <cuda_runtime.h>

void rgba_to_greyscale_cpu(const uchar4* const rgbaImage,
	unsigned char* const greyImage,
	size_t numRows,
	size_t numCols)
{
	for (std::size_t r = 0; r < numRows; ++r) {
		for (std::size_t c = 0; c < numCols; ++c) {
			const uchar4* const rgba = &(rgbaImage[r * numCols + c]);
			float channelSum = .299f * rgba->x + .587f * rgba->y + .114f * rgba->z;
			greyImage[r * numCols + c] = channelSum;
		}
	}
}