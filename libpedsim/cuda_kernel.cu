#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ped_model.h"

#include <cstdlib>
#include <iostream>
#include <cmath>
using namespace std;

// Memory leak check with msvc++
#include <stdlib.h>
#define SIZE 1024

/* ---------------------------
	SET HEATMAP FUNCTIONS
-----------------------------*/

void Ped::Model::setupHeatmapCuda()
{
	cout << "malloc";
	// Allocate memory on CPU
	int *hm = (int*)calloc(SIZE*SIZE, sizeof(int));
	int *shm = (int*)malloc(SCALED_SIZE*SCALED_SIZE*sizeof(int));
	int *bhm = (int*)malloc(SCALED_SIZE*SCALED_SIZE*sizeof(int));

	heatmap = (int**)malloc(SIZE*sizeof(int*));
	scaled_heatmap = (int**)malloc(SCALED_SIZE*sizeof(int*));
	blurred_heatmap = (int**)malloc(SCALED_SIZE*sizeof(int*));

	// Initialize values, point to right memory
	for (int i = 0; i < SIZE; i++)
	{
		heatmap[i] = hm + SIZE*i;
	}
	for (int i = 0; i < SCALED_SIZE; i++)
	{
		scaled_heatmap[i] = shm + SCALED_SIZE*i;
		blurred_heatmap[i] = bhm + SCALED_SIZE*i;
	}


	desiredX = new int[agents.size()];
	desiredY = new int[agents.size()];
	agents = agents.size();
	agentsSize = &agents;

	//cudaMallocHost(&desiredX, agents.size()*sizeof(int));
	//cudaMallocHost(&desiredY, agents.size()*sizeof(int));
	
	for (int i = 0; i < agents.size(); i++)
	{
		desiredX[i] = agents[i]->getDesiredX();
		desiredY[i] = agents[i]->getDesiredY();
	}

	cudaMalloc(&d_desiredX, agents.size()*sizeof(int));
	cudaMalloc(&d_desiredY, agents.size()*sizeof(int));

	cudaMemcpy(d_desiredX, desiredX, agents.size()*sizeof(int));
	cudaMemcpy(d_desiredY, desiredY, agents.size()*sizeof(int));

	cudaMalloc(&d_agentsSize, sizeof(int*));
	cudaMempy(d_agentsSize, agentsSize, sizeof(int*), cudaMemcpyHostToDevice);


	// Allocate memory on GPU
	cudaMalloc(&d_heatmap, SIZE*SIZE*sizeof(int));
	cudaMemcpy(d_heatmap, heatmap[0], SIZE*SIZE*sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc(&d_scaled_heatmap, SCALED_SIZE*SCALED_SIZE*sizeof(int));
	cudaMemcpy(d_scaled_heatmap, scaled_heatmap[0], SCALED_SIZE*SCALED_SIZE*sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc(&d_blurred_heatmap, SCALED_SIZE*SCALED_SIZE*sizeof(int*));
	cudaMemcpy(d_blurred_heatmap, blurred_heatmap[0], SCALED_SIZE*SCALED_SIZE*sizeof(int), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	
	// Copy memory from host to device
}

/* ---------------------------
	UPPDATE HEATMAP FUNCTIONS
  --------------------------*/ 

__global__ void kernel_fade(int *dev_heatmap)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;	
	// int y = blockIdx.y * blockDim.y + threadIdx.y;	
	if (x < SIZE)
	{
		dev_heatmap[x] = (int)round(dev_heatmap[x] * 0.80);
	}
}

__global__ void kernel_agents(int *dev_heatmap, int size_agents, int *desiredX, int *desiredY)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < size_agents){
		int x = desiredX[i];
		int y = desiredY[i];

		if(x>=0 && x<SIZE && y>=0 && y<SIZE)
			// intensify heat for better color results
			//&dev_heatmap[y][x] += 40;
			atomicAdd(&dev_heatmap[y*SIZE + x], 40);
	}
}

__global__ void kernel_clip(int *dev_heatmap)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;	
	if (x < SIZE ){
		dev_heatmap[x] = dev_heatmap[x] < 255 ? dev_heatmap[x] : 255;
	}
}

__global__ void kernel_scale(int *dev_heatmap, int *dev_scaled_heatmap)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;	
	int x = id / SIZE;
	int y = id % SIZE;
	if (id < SCALED_SIZE)
	{
		int value = dev_heatmap[id];
		for (int cellY = 0; cellY < CELLSIZE; cellY++)
		{
			for (int cellX = 0; cellX < CELLSIZE; cellX++)
			{
				int s_y = y * CELLSIZE + cellY;
                int s_x = x * CELLSIZE + cellX;
				dev_scaled_heatmap[s_y*SCALED_SIZE + s_x] = value;

			}
		}

	}
}

__global__ void kernel_blur(int *dev_heatmap, int *dev_blurred_heatmap, int *dev_scaled_heatmap)
{
	//weights for blur
	const int w[5][5] = {
		{ 1, 4, 7, 4, 1 },
		{ 4, 16, 26, 16, 4 },
		{ 7, 26, 41, 26, 7 },
		{ 4, 16, 26, 16, 4 },
		{ 1, 4, 7, 4, 1 }
	};
	int x = blockIdx.x * blockDim.x + threadIdx.x;	
	int y = blockIdx.y * blockDim.y + threadIdx.y;	
	if (x >= 2 && x < SCALED_SIZE && y >= 2 && y < SCALED_SIZE)
	{
		int sum = 0;
		for (int k = -2; k < 3; k++)
		{
			for (int l = -2; l < 3; l++)
			{
				sum += w[2 + k][2 + l] * dev_scaled_heatmap[y + k][x + l];
			}
		}
		int value = sum / WEIGHTSUM;
		dev_blurred_heatmap[y][x] = 0x00FF0000 | value << 24;
	}
}


void Ped::Model::updateHeatmapCuda() 
{



	for (int i = 0; i < agents.size(); i++)
	{
		desiredX[i] = agents[i]->getDesiredX();
		desiredY[i] = agents[i]->getDesiredY();
	}


	cudaMemcpy(d_desiredX, desiredX, agents.size()*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_desiredY, desiredY, agents.size()*sizeof(int), cudaMemcpyHostToDevice);

	//cout << d_desiredX[0] << "\n";

	// dim3 threads_per_block(32, 32);
    // dim3 num_blocks(SIZE / threads_per_block.x, SIZE / threads_per_block.y);
	// Fade heatmap
	kernel_fade<<<SIZE, SIZE>>>(d_heatmap);
	//heatmapFading<<<SIZE, SIZE, 0, stream>>>(heatmap_tmp);

    // int threads_per_blocki = 1024;
    // int num_blocksi = (agents.size() + threads_per_blocki - 1) / threads_per_blocki;
	kernel_agents<<<1, agents.size()>>>(d_heatmap, d_agentsSize, d_desiredX, d_desiredY);


	// int size_agents = agents.size();
	// Ped::Tagent *d_agents;
	// cudaMalloc((void **)&d_agents, size_agents*sizeof(Ped::Tagent));
	// cudaMemcpyAsync(d_agents, agents, size_agents*sizeof(Ped::Tagent), cudaMemcpyHostToDevice, stream2);
	// cudaStreamWaitEvent(stream1, ev1);
	// kernel_agents<<<1, size_agents, 0, stream1>>>(d_heatmap, d_agents, size_agents);
	// free(d_agents)
	// free(*d_agents)

	//Clip heatmap
	kernel_clip<<<SIZE, SIZE>>>(d_heatmap, agents.size(), d_desiredX, d_desiredY);

	//Scale heatmap
	kernel_scale<<<SIZE, SIZE>>>(d_heatmap, d_scaled_heatmap);

	// Blur heatmap
	kernel_blur<<<SIZE,SIZE >>>(d_heatmap, d_blurred_heatmap, d_scaled_heatmap);

	cudaMemcpy(blurred_heatmap, d_blurred_heatmap, SCALED_SIZE * SCALED_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

}

int Ped::Model::getHeatmapSize() const {
	return SCALED_SIZE;
}

