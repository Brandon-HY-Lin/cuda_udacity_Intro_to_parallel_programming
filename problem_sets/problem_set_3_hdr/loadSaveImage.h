#ifndef LOADSAVEIMAGE_H__
#define LOADSAVEIMAGE_H__

#include <string>
#include <cuda_runtime.h> //for uchar4

void loadImageHDR(const std::string& filename,
    float** imagePtr,
    size_t* numRows, size_t* numCols);

#endif
