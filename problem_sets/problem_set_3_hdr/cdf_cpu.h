#ifndef CDF_CPU_H__
#define CDF_CPU_H__

void cdf_cpu(const float* const h_logLuminance, unsigned int* const h_cdf,
	const size_t numRows, const size_t numCols, const size_t numBins,
	float& logLumMin, float& logLumMax);

#endif