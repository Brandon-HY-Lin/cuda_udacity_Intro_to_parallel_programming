#ifndef SORT_GPU_H__
#define SORT_GPU_H__

void sort_cuda(unsigned int* const d_inputVals,
    unsigned int* const d_inputPos,
    unsigned int* const d_outputVals,
    unsigned int* const d_outputPos,
    const size_t numElems);

#endif /*SORT_GPU_H__*/