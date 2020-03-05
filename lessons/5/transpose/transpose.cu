#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

const int N= 1024;		// matrix size is NxN
const int K= 32;				// tile size is KxK


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
		cudaEventSynchronize(stop);
		float elapsed;
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
	}
};


int compare_matrices (float *m1, float *m2)
{
	for (int y = 0; y < N; y++) {
		for (int x = 0; x < N; x++) {
			if (m1[x + y*N] != m2[x + y*N])
				return 1;
		}
	}

	// if m1 == m2: return 0
	return 0;
}

// fill a matrix with sequential numbers in the range 0..N-1
void fill_matrix (float *mat)
{
	for (int i = 0; i < N*N; i++) {
		mat[i] = (float)i;
	}
}

void 
transpose_CPU (float in[], float out[])
{
	for (int y = 0; y < N; y++) {
		for (int x = 0; x < N; x++) {
			out[y + x*N] = in[x + y*N];	// out(y, x) = in(x, y)
		}
	}
}

// to be launched on a single thread
__global__ void 
transpose_serial (float in[], float out[])
{
	for (int y = 0; y < N; y++) {
		for (int x = 0; x < N; x++) {
			out[y + x*N] = in[x + y*N];	// out(y, x) = in(x, y)
		}
	}
}

// to be launched with one thread per row of output matrix
__global__ void
transpose_parallel_per_row (float in[], float out[])
{
	int x = threadIdx.x;

	for (int y = 0; y < N; y++) {
		out[y + x*N] = in[x + y*N]; // out(y, x) = in(x, y)
	}
}


// to be launched with one thread per element, in KxK threadblocks
// thread (x,y) in grid writes element (i,j) of output matrix 
__global__ void 
transpose_parallel_per_element (float in[], float out[])
{
	int x = blockIdx.x * K + threadIdx.x;
	int y = blockIdx.y * K + threadIdx.y;

	out[y + x*N] = in[x + y*N]; // out(y, x) = in(x, y)
}


// to be launched with one thread per element, in (tilesize)x(tilesize) threadblocks
// thread blocks read & write tiles, in coalesced fashion
// adjacent threads read adjacent input elements, write adjacent output elements
__global__ void
transpose_parallel_per_element_tiled (float in[], float out[])
{
	// (x, y) locations of the tile corners for input & output matrices:
	int in_corner_x = blockIdx.x * K, in_corner_y = blockIdx.y * K;
	
	// transpose coordinates
	int out_corner_x = in_corner_y, out_corner_y = in_corner_x;

	int x = threadIdx.x, y = threadIdx.y;

	__shared__ float tile[K][K];

	// coalesced read from global mem, TRANSPOSED write into shared mem:
	tile[y][x] = in[(in_corner_x + x) + (in_corner_y + y)*N];

	__syncthreads();
	// read from shared mem, coalesced write to global mem:

	out[(out_corner_x + x) + (out_corner_y + y)*N] = tile[x][y];;
}


// to be launched with one thread per element, in (tilesize)x(tilesize) threadblocks
// thread blocks read & write tiles, in coalesced fashion
// adjacent threads read adjacent input elements, write adjacent output elements
__global__ void
transpose_parallel_per_element_tiled16 (float in[], float out[])
{
	// (x, y) locations of the tile corners for input & output matrices:
	int in_corner_x = blockIdx.x * 16, in_corner_y = blockIdx.y * 16;
	int out_corner_x = in_corner_y, out_corner_y = in_corner_x;

	int x = threadIdx.x, y = threadIdx.y;
	
	// move and transpose data to shared memory
	__shared__ float tile[16][16];
	// coalesced read from global mem, TRANSPOSED write into shared mem:
	tile[y][x] = in[(in_corner_x + x) + (in_corner_y + y) * N];

	__syncthreads();
	// read from shared mem, coalesced write to global mem:

	out[(out_corner_x + x) + (out_corner_y + y)*N] = tile[x][y];
}


// to be launched with one thread per element, in KxK threadblocks
// thread blocks read & write tiles, in coalesced fashion
// shared memory array padded to avoid bank conflicts
__global__ void
transpose_parallel_per_element_tiled_padded16 (float in[], float out[])
{
	// (x, y) locations of the tile corners for input & output matrices:
	int in_corner_x = blockIdx.x * 16, in_corner_y = blockIdx.y * 16;
	int out_corner_x = in_corner_y, out_corner_y = in_corner_x;

	int x = threadIdx.x, y = threadIdx.y;

	// declared shared memory with odd size
	__shared__ float tile[16][16+1];

	// coalesced read from global mem, TRANSPOSED write into shared mem:
	tile[y][x] = in[(in_corner_x + x) + (in_corner_y + y)*N];

	__syncthreads();
	// read from shared mem, coalesced write to global mem:
	out[(out_corner_x + x) + (out_corner_y + y) * N] = tile[x][y];
}


int main (int argc, char **argv)
{
	const int numbytes = N * N * sizeof(float);

	float *in = (float *) malloc(numbytes);
	float *out = (float *) malloc(numbytes);
	float *gold = (float *) malloc(numbytes);

	// init in-matrix
	fill_matrix(in);
	transpose_CPU(in, gold);

	float *d_in, *d_out;

	// Alloc GPU mem
	cudaMalloc(&d_in, numbytes);
	cudaMalloc(&d_out, numbytes);
	cudaMemcpy(d_in, in, numbytes, cudaMemcpyHostToDevice);

	GpuTimer timer;

/*
 * Now time each kernel and verify that it produces the correct result.
 *
 * To be really careful about benchmarking purposes, we should run every kernel once
 * to "warm" the system and avoid any compilation or code-caching effects, then run
 * every kernel 10 or 100 times and average the timings to smooth out any variance.
 * But this makes for messy code and our goal is teaching, not detailed benchmarking.
 */

	timer.Start();
	transpose_serial<<<1, 1>>>(d_in, d_out);
	timer.Stop();
	cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
	printf("transpose_serial: %g ms.\nVerifying transpose...%s\n",
			timer.Elapsed(), compare_matrices(out, gold) ? "Failed" : "Success");


	timer.Start();
	transpose_parallel_per_row<<<1, N>>>(d_in, d_out);
	timer.Stop();
	cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
	printf("transpose_parallel_per_row: %g ms.\nVerifying transpose...%s\n",
			timer.Elapsed(), compare_matrices(out, gold) ? "Failed" : "Success");


	dim3 blocks(N/K, N/K);	// blocks per grid
	dim3 threads(K, K); 		// threads per block

	timer.Start();
	transpose_parallel_per_element<<<blocks, threads>>>(d_in, d_out);
	timer.Stop();
	cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
	printf("transpose_parallel_per_element: %g ms.\nVerifying transpose...%s\n",
			timer.Elapsed(), compare_matrices(out, gold) ? "Failed" : "Success");

	
	dim3 blocksKxK(N/K, N/K);	// blocks per grid
	dim3 threadsKxK(K, K);		// threads per block

	timer.Start();
	transpose_parallel_per_element_tiled<<<blocksKxK, threadsKxK>>>(d_in, d_out);
	timer.Stop();
	cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
	printf("transpose_parallel_per_element_tiled %dx%d: %g ms.\nVerifying ...%s\n", 
		   K, K, timer.Elapsed(), compare_matrices(out, gold) ? "Failed" : "Success");


	dim3 blocks16x16(N/16,N/16); // blocks per grid
	dim3 threads16x16(16,16);	 // threads per block

	timer.Start();
	transpose_parallel_per_element_tiled16<<<blocks16x16, threads16x16>>>(d_in, d_out);
	timer.Stop();
	cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
	printf("transpose_parallel_per_element_tiled 16x16: %g ms.\nVerifying ...%s\n", 
			timer.Elapsed(), compare_matrices(out, gold) ? "Failed" : "Success");


	timer.Start();
	transpose_parallel_per_element_tiled_padded16<<<blocks16x16, threads16x16>>>(d_in, d_out);
	timer.Stop();
	cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
	printf("transpose_parallel_per_element_tiled_padded 16x16: %g ms.\nVerifying...%s\n", 
			timer.Elapsed(), compare_matrices(out, gold) ? "Failed" : "Success");

	// Free memory
	cudaFree(d_in);
	cudaFree(d_out);

	free(in);
	free(out);
	free(gold);

	return 0;
}

