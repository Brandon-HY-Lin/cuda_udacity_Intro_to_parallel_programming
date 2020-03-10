#ifndef BLUR_GPU_CUH__
#define BLUR_GPU_CUH__

void blur_gpu(const uchar4* const h_inputImageRGBA, uchar4* const d_inputImageRGBA,
    uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
    unsigned char* d_redBlurred,
    unsigned char* d_greenBlurred,
    unsigned char* d_blueBlurred,
    const int filterWidth);

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
    const float* const h_filter, const size_t filterWidth);

#endif /*BLUR_GPU_CUH__*/