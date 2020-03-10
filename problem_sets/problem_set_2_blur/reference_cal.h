#ifndef REFERENCE_H__
#define REFERENCE_H__

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

void blur_cpu(const uchar4* const rgbaImage, uchar4* const outputImage,
    std::size_t numRows, std::size_t numCols,
    const float* const filter, const int filterWidth);

#endif