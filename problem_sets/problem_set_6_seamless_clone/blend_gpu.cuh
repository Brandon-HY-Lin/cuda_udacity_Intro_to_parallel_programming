#ifndef BLEND_GPU_CUH__
#define BLEND_GPU_CUH__

void blend_gpu(const uchar4* const h_sourceImg,  //IN
    const size_t numRowsSource, const size_t numColsSource,
    const uchar4* const h_destImg, //IN
    uchar4* const h_blendedImg);

#endif /*BLEND_GPU_CUH__*/