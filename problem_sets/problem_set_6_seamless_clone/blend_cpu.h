#ifndef BLEND_CPU_H__
#define BLEND_CPU_H__

void blend_cpu(const uchar4* const h_sourceImg,
    const size_t numRowsSource, const size_t numColsSource,
    const uchar4* const h_destImg,
    uchar4* const h_blendedImg);

#endif