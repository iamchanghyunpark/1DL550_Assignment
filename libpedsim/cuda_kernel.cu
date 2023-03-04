#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "heatmap_seq.h"

#include <stdio.h>

void kernel_addWithCuda(int *hm, int **heatmap, int size);

__global__ void kernel_add(int *d_hm, int **d_heatmap, int size)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (id < size) {
		// mul = SIZE*id
		// atomicAdd(&heatmap[id], hm);
		// atomicAdd(&heatmap[id], mul);
		d_heatmap[id] = d_hm + size*id;

	}
}

void kernel_setupHeatmap(int *hm, int *shm, int *bhm);
	kernel_addWithCuda(hm, heatmap, SIZE);
	kernel_addWithCuda(shm, scaled_heatmap, SCALED_SIZE);
	kernel_addWithCuda(bhm, blurred_heatmap, SCALED_SIZE);

// Helper function for using CUDA to add vectors in parallel.
void kernel_addWithCuda(int *hm, int **heatmap, int size)
{
	int *d_hm;
	int *d_heatmap;
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		fprintf(stderr, "%s.\n", cudaGetErrorString(cudaGetLastError()));
		goto Error;
	}

	// Allocate GPU buffers for three vector
	cudaStatus = cudaMalloc((void **) &d_hm, size*size*sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void **) &d_heatmap, size*size*sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vector from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(d_hm, hm, size*size*sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(d_heatmap, heatmap, size*size*sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	kernel_add<<<1,size>>>(d_hm, d_heatmap, size);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	else
	{
		//fprintf(stderr, "Cuda launch succeeded! \n");
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(d_hm, hm, size*size*sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(d_heatmap, heatmap, size*size*sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(d_hm);
	cudaFree(d_heatmap);
	if (cudaStatus != 0){
		fprintf(stderr, "Cuda does not seem to be working properly.\n"); // This is not a good thing
	}
	else{
		fprintf(stderr, "Cuda functionality test succeeded.\n"); // This is a good thing
	}

	return cudaStatus;
}
