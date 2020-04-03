#ifndef PROCESSING_H__
#define PROCESSING_H__

void preProcess(uchar4** sourceImg,
    size_t& numRows, size_t& numCols,
    uchar4** destImg,
    uchar4** blendedImg, const std::string& source_filename,
    const std::string& dest_filename);


void postProcess(const uchar4* const blendedImg,
    const size_t numRowsDest, const size_t numColsDest,
    const std::string& output_file);

#endif /*PROCESSING_H__*/