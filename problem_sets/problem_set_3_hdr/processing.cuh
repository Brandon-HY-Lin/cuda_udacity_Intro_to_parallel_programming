#ifndef PROCESSING_H__
#define PROCESSING_H__

void preProcessGPU(float** d_luminance, unsigned int** d_cdf,
	std::size_t* numRows, std::size_t* numCols,
	unsigned int* numberOfBins,
	const std::string& filename);

void postProcessGPU(const std::string& output_file,
	size_t numRows, size_t numCols,
	float min_log_Y, float max_log_Y);

void cleanupGlobalMemory(void);

#endif /*PROCESSING_H__*/