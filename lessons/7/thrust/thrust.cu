#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cstdlib>
#include <vector>

struct GpuTimer
{
	cudaEvent_t start;
	cudaEvent_t stop;

	GpuTimer ()
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer ()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start ()
	{
		cudaEventRecord(start, 0);
	}

	void Stop ()
	{
		cudaEventRecord(stop, 0);
	}

	float Elapsed ()
	{
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
	}
};


int main (int argc, char **argv)
{
	// generate N random numbers serially
	// std::vector<int> sizes = {100000, 1000000, 10000000};
	std::vector<int> sizes;
	sizes.push_back(100000);
	sizes.push_back(1000000);
	sizes.push_back(10000000);

	for (std::vector<int>::iterator it=sizes.begin(); it != sizes.end(); it++) {
		int N = *it;

		thrust::host_vector<char> h_vec(N);
		std::generate(h_vec.begin(), h_vec.end(), rand);

		// transfer data to the device
		thrust::device_vector<char> d_vec = h_vec;

		// sort data on the device
		GpuTimer timer;
		timer.Start();
		thrust::sort(d_vec.begin(), d_vec.end());
		timer.Stop();

		// move data from device to host
		thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

	  	printf("Thrust sorted %d keys in %g ms\n", N, timer.Elapsed());
  	}
  	return 0;
}
