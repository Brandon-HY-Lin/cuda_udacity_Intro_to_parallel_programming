#ifndef CDF_GPU_CUH__
#define CDF_GPU_CUH__

void cdf_gpu(const float* const d_logLuminance,
    unsigned int* const d_cdf,
    float& min_logLum,
    float& max_logLum,
    const size_t numRows,
    const size_t numCols,
    const size_t numBins);

#endif