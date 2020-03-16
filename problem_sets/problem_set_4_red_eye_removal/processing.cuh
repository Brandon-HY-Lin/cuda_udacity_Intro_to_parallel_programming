#ifndef PROCESSING_CUH__
#define PROCESSING_CUH__

void preProcess(unsigned int** inputVals,
    unsigned int** inputPos,
    unsigned int** outputVals,
    unsigned int** outputPos,
    size_t& numElems,
    const std::string& filename,
    const std::string& template_file);

void postProcess(const unsigned int* const outputVals,
    const unsigned int* const outputPos,
    const size_t numElems,
    const std::string& output_file);

#endif /*PROCESSING_CUH__*/