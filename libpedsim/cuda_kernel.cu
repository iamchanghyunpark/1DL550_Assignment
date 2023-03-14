#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ped_model.h"

#include <cstdlib>
#include <iostream>
#include <cmath>
#include <typeinfo>
using namespace std;

// Memory leak check with msvc++
#include <stdlib.h>

/* ---------------------------
	SET HEATMAP FUNCTIONS
-----------------------------*/

void Ped::Model::setupHeatmapCuda()
{
	agentsSize = agents.size();

	// Allocate memory on GPU
	cudaMalloc(&d_desiredX, agentsSize*sizeof(int));
	cudaMalloc(&d_desiredY, agentsSize*sizeof(int));
	cudaMalloc(&d_heatmap, SIZE*SIZE*sizeof(int));
	cudaMalloc(&d_scaled_heatmap, SCALED_SIZE*SCALED_SIZE*sizeof(int));
	cudaMalloc(&d_blurred_heatmap, SCALED_SIZE*SCALED_SIZE*sizeof(int));

	// Copy from CPU to GPU
	cudaMemcpy(d_heatmap, heatmap[0], SIZE*SIZE*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_scaled_heatmap, scaled_heatmap[0], SCALED_SIZE*SCALED_SIZE*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_blurred_heatmap, blurred_heatmap[0], SCALED_SIZE*SCALED_SIZE*sizeof(int), cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
}

/* ---------------------------------
	UPPDATE HEATMAP KERNEL FUNCTIONS
  ---------------------------------*/ 

__global__ void kernel_fade(int *d_heatmap)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < SIZE*SIZE){
		d_heatmap[i] = (int)round(d_heatmap[i] * 0.8);
	}
}

__global__ void kernel_agents(int *d_heatmap, int agentSize, int *desiredX, int *desiredY)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < agentSize){
		int x = desiredX[i];
		int y = desiredY[i];
		if (x >= 0 && x < SIZE && y >= 0 && y < SIZE){
			atomicAdd(&d_heatmap[y * SIZE + x], 40); 
		}
	}
}

__global__ void kernel_clip(int *d_heatmap)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;	
	if (i < SIZE*SIZE){
		atomicMin(&d_heatmap[i], 255);
	}
}

__global__ void kernel_scale(int *d_heatmap, int *d_scaled_heatmap)
{
	// Shared memory
	__shared__ int shared_heatmap[SIZE];
	__shared__ int shared_scaled_heatmap[SCALED_SIZE];
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;	
	if(i < SIZE*SIZE){
		int x = threadIdx.x;
		int y = blockIdx.x;
		shared_heatmap[x] = d_heatmap[i];
		int value = shared_heatmap[x];
		for (int cellY = 0; cellY < CELLSIZE; cellY++)
		{
			for (int cellX = 0; cellX < CELLSIZE; cellX++)
			{
				int index = cellY + x * CELLSIZE + cellX;
				if (index < SCALED_SIZE){
					shared_scaled_heatmap[index] = value;
					int global_index = (x * CELLSIZE + cellX) + (y * CELLSIZE + cellY) * SCALED_SIZE;
					d_scaled_heatmap[global_index] = shared_scaled_heatmap[index];
				}
			}
		}
	}
}

__global__ void kernel_blur(int *d_blurred_heatmap, int *d_scaled_heatmap)
{
	// Weigths for blur
	const int w[5][5] = {
		{ 1, 4, 7, 4, 1 },
		{ 4, 16, 26, 16, 4 },
		{ 7, 26, 41, 26, 7 },
		{ 4, 16, 26, 16, 4 },
		{ 1, 4, 7, 4, 1 }
	};
#define WEIGHTSUM 273

	int x = threadIdx.x;
	int y = blockIdx.x;
	if(x >= 2 && x <= SCALED_SIZE - 2 && y >= 2 && y <= SCALED_SIZE - 2){
		int sum = 0;
		for (int k = -2; k < 3; k++)
		{
			for (int l = -2; l < 3; l++)
			{
				int weight = w[2 + k][2 + l];
				int index = (x + l) + (y + k) * SCALED_SIZE;
				sum += weight * d_scaled_heatmap[index];
			}
		}
		int value = sum / WEIGHTSUM;
		d_blurred_heatmap[y*SCALED_SIZE + x] = 0x00FF0000 | value << 24;
	}
}


void Ped::Model::updateHeatmapCuda() 
{
	// Copy memory of desired position from CPU to GPU
	cudaMemcpyAsync(d_desiredX, desiredX, agentsSize*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_desiredY, desiredY, agentsSize*sizeof(int), cudaMemcpyHostToDevice);

	cudaEvent_t start[5], stop[5];
	float time;
	for(int i = 0; i < 5; i++) {
		cudaEventCreate(&start[i]);
		cudaEventCreate(&stop[i]);
	}
	

	// Fade heatmap
	cudaEventRecord(start[0]);
	kernel_fade<<<SIZE, SIZE>>>(d_heatmap);
	cudaEventRecord(stop[0]);
	cudaEventSynchronize(stop[0]);
	cudaEventElapsedTime(&time, start[0], stop[0]);
	cout << "Kernel fade time: " << time << "\n";

	// Desired positions
	cudaEventRecord(start[1]);
	int BLOCKS = 1 + agentsSize / SIZE;
	kernel_agents<<<BLOCKS, SIZE>>>(d_heatmap, agentsSize, d_desiredX, d_desiredY);
	cudaEventRecord(stop[1]);
	cudaEventSynchronize(stop[1]);
	cudaEventElapsedTime(&time, start[1], stop[1]);
	cout << "Kernel agents time: " << time << "\n";

	// Desired positions

	// Clip heatmap
	cudaEventRecord(start[2]);
	kernel_clip<<<SIZE, SIZE>>>(d_heatmap);
	cudaEventRecord(stop[2]);
	cudaEventSynchronize(stop[2]);
	cudaEventElapsedTime(&time, start[2], stop[2]);
	cout << "Kernel clip time: " << time << "\n";

	// Scale heatmap

	cudaEventRecord(start[3]);
	kernel_scale<<<SIZE, SIZE>>>(d_heatmap, d_scaled_heatmap);
	cudaEventRecord(stop[3]);
	cudaEventSynchronize(stop[3]);
	cudaEventElapsedTime(&time, start[3], stop[3]);
	cout << "Kernel scale time: " << time << "\n";

	// Blur heatmap
	cudaEventRecord(start[4]);
	kernel_blur<<<SIZE,SIZE>>>(d_blurred_heatmap, d_scaled_heatmap);
	cudaEventRecord(stop[4]);
	cudaEventSynchronize(stop[4]);
	cudaEventElapsedTime(&time, start[4], stop[4]);
	cout << "Kernel blur time: " << time << "\n";

	// Copy memory from GPU back to CPU
	cudaMemcpyAsync(blurred_heatmap[0], d_blurred_heatmap, SCALED_SIZE * SCALED_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();
}
