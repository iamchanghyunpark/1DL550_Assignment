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
	//__shared__ int shared_heatmap[SIZE];
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < SIZE*SIZE){
		// shared_heatmap[threadIdx.x] = d_heatmap[i];
		// int value = (int)round(shared_heatmap[threadIdx.x] * 0.8);
		// shared_heatmap[threadIdx.x] = value;
		// d_heatmap[i] = shared_heatmap[threadIdx.x];
		d_heatmap[i] = (int)round(d_heatmap[i] * 0.8);
	}
}

__global__ void kernel_agents(int *d_heatmap, int agentSize, int *desiredX, int *desiredY)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < agentSize){
		int x = desiredX[i];
		int y = desiredY[i];
		// if (x < 0 || x >= SIZE || y < 0 || y >= SIZE)
		// {
		// 	return;
		// }
		// atomicAdd(&d_heatmap[y * SIZE + x], 40); 
		if (x >= 0 && x < SIZE && y >= 0 && y < SIZE){
			atomicAdd(&d_heatmap[y * SIZE + x], 40); 
		}
	}
}

__global__ void kernel_clip(int *d_heatmap)
{
	//__shared__ int shared_heatmap[SIZE];
	int i = blockIdx.x * blockDim.x + threadIdx.x;	
	if (i < SIZE*SIZE){
		// shared_heatmap[threadIdx.x] = d_heatmap[i];
		// shared_heatmap[threadIdx.x] = shared_heatmap[threadIdx.x] < 255 ? shared_heatmap[threadIdx.x] : 255;
		// d_heatmap[i] = shared_heatmap[threadIdx.x];
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
					// int global_x_index = x * CELLSIZE + cellX;
					// int global_y_index = y * CELLSIZE + cellY;
					// global_y_index *= SCALED_SIZE;
					// int global_index = global_x_index + global_y_index;
					int global_index = (x * CELLSIZE + cellX) + (y * CELLSIZE + cellY) * SCALED_SIZE;
					d_scaled_heatmap[global_index] = shared_scaled_heatmap[index];
				}
			}
		}
		// for (int cellY = 0; cellY < CELLSIZE; cellY++)
		// {
		// 	for (int cellX = 0; cellX < CELLSIZE; cellX++)
		// 	{
		// 	int index = cellY + x * CELLSIZE + cellX;
		// 	if (index >= SCALED_SIZE) continue;
		// 	int global_x_index = x * CELLSIZE + cellX;
		// 	int global_y_index = y * CELLSIZE + cellY;
        //     global_y_index *= SCALED_SIZE;
        //     int global_index = global_x_index + global_y_index;
		// 	//int value = shared_scaled_heatmap[index];
		// 	d_scaled_heatmap[global_index] = shared_scaled_heatmap[index];
		// 	}
		// }
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

	// int i = blockIdx.x * blockDim.x + threadIdx.x;	
	// if (i < SIZE*SIZE){
	// 	int thread_x = threadIdx.x;
	// 	int block_y = blockIdx.x;
	// 	for (int cellY = 0; cellY < CELLSIZE; cellY++)
	// 	{
	// 		for (int cellX = 0; cellX < CELLSIZE; cellX++)
	// 		{
	// 			int sum = 0;
	// 			int x = threadIdx.x + cellX;
	// 			int y = blockIdx.x + cellY;
	// 			if (x < 2 || x > SCALED_SIZE - 2) continue;
	// 			if (y < 2 || y > SCALED_SIZE - 2) continue;
	// 			for (int k = -2; k < 3; k++)
	// 			{
	// 				for (int l = -2; l < 3; l++)
	// 				{
	// 					int weight_val = w[2 + k][2 + l];
	// 					int cur_y = y + k;
	// 					int cur_x = x + l;
	// 					int global_cur_y = cur_y * SCALED_SIZE;
	// 					int global_cur_x = cur_x;
	// 					int global_index = global_cur_x + global_cur_y;
	// 					sum += weight_val * d_scaled_heatmap[global_index];
	// 				}
	// 			}
	// 			int value = sum / WEIGHTSUM;
	// 			d_blurred_heatmap[y*SCALED_SIZE + x] = 0x00FF0000 | value << 24;
	// 		}
	// 	}
	// }
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

	// Fade heatmap
	kernel_fade<<<SIZE, SIZE>>>(d_heatmap);

	// Desired positions
	int BLOCKS = 1 + agentsSize / SIZE;
	kernel_agents<<<BLOCKS, SIZE>>>(d_heatmap, agentsSize, d_desiredX, d_desiredY);

	// Clip heatmap
	kernel_clip<<<SIZE, SIZE>>>(d_heatmap);

	// Scale heatmap
	kernel_scale<<<SIZE, SIZE>>>(d_heatmap, d_scaled_heatmap);

	// Blur heatmap
	kernel_blur<<<SIZE,SIZE>>>(d_blurred_heatmap, d_scaled_heatmap);

	// Copy memory from GPU back to CPU
	cudaMemcpyAsync(blurred_heatmap[0], d_blurred_heatmap, SCALED_SIZE * SCALED_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();
}
