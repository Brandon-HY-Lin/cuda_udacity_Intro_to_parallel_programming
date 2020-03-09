#ifndef REFERENCE_CAL_H__
#define REFERENCE_CAL_H__
void rgba_to_greyscale_cpu(const uchar4* const rgbaImage,
	unsigned char* const greyImage,
	size_t numRows,
	size_t numCols);

#endif /*REFERENCE_CAL_H__*/