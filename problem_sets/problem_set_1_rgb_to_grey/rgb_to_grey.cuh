#ifndef RGB_TO_GREY__
#define RGB_TO_GREY__

void rgba_to_greyscale_gpu(const uchar4* const h_rgbaImage,
	uchar4* const d_rgbaImage,
	unsigned char* const d_greyImage,
	std::size_t numRows, std::size_t numCols);

#endif /*RGB_TO_GREY__*/