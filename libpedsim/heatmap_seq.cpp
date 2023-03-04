// Created for Low Level Parallel Programming 2017
//
// Implements the heatmap functionality. 
//
#include "ped_model.h"

#include <cstdlib>
#include <iostream>
#include <cmath>
using namespace std;

// Memory leak check with msvc++
#include <stdlib.h>

// Cuda
#include "cuda_kernel.h"

// Sets up the heatmap
/*
void Ped::Model::setupHeatmapSeq()
{
	int *hm = (int*)calloc(SIZE*SIZE, sizeof(int));
	int *shm = (int*)malloc(SCALED_SIZE*SCALED_SIZE*sizeof(int));
	int *bhm = (int*)malloc(SCALED_SIZE*SCALED_SIZE*sizeof(int));

	heatmap = (int**)malloc(SIZE*sizeof(int*));

	scaled_heatmap = (int**)malloc(SCALED_SIZE*sizeof(int*));
	blurred_heatmap = (int**)malloc(SCALED_SIZE*sizeof(int*));


	for (int i = 0; i < SIZE; i++)
	{
		heatmap[i] = hm + SIZE*i;
	}
	for (int i = 0; i < SCALED_SIZE; i++)
	{
		scaled_heatmap[i] = shm + SCALED_SIZE*i;
		blurred_heatmap[i] = bhm + SCALED_SIZE*i;
	}
}

// Updates the heatmap according to the agent positions
void Ped::Model::updateHeatmapSeq()
{
	for (int x = 0; x < SIZE; x++)
	{
		for (int y = 0; y < SIZE; y++)
		{
			// heat fades
			heatmap[y][x] = (int)round(heatmap[y][x] * 0.80);
		}
	}

	// Count how many agents want to go to each location
	for (int i = 0; i < agents.size(); i++)
	{
		Ped::Tagent* agent = agents[i];
		int x = agent->getDesiredX();
		int y = agent->getDesiredY();

		if (x < 0 || x >= SIZE || y < 0 || y >= SIZE)
		{
			continue;
		}

		// intensify heat for better color results
		heatmap[y][x] += 40;

	}

	for (int x = 0; x < SIZE; x++)
	{
		for (int y = 0; y < SIZE; y++)
		{
			heatmap[y][x] = heatmap[y][x] < 255 ? heatmap[y][x] : 255;
		}
	}

	// Scale the data for visual representation
	for (int y = 0; y < SIZE; y++)
	{
		for (int x = 0; x < SIZE; x++)
		{
			int value = heatmap[y][x];
			for (int cellY = 0; cellY < CELLSIZE; cellY++)
			{
				for (int cellX = 0; cellX < CELLSIZE; cellX++)
				{
					scaled_heatmap[y * CELLSIZE + cellY][x * CELLSIZE + cellX] = value;
				}
			}
		}
	}

	// Weights for blur filter
	const int w[5][5] = {
		{ 1, 4, 7, 4, 1 },
		{ 4, 16, 26, 16, 4 },
		{ 7, 26, 41, 26, 7 },
		{ 4, 16, 26, 16, 4 },
		{ 1, 4, 7, 4, 1 }
	};

#define WEIGHTSUM 273
	// Apply gaussian blurfilter		       
	for (int i = 2; i < SCALED_SIZE - 2; i++)
	{
		for (int j = 2; j < SCALED_SIZE - 2; j++)
		{
			int sum = 0;
			for (int k = -2; k < 3; k++)
			{
				for (int l = -2; l < 3; l++)
				{
					sum += w[2 + k][2 + l] * scaled_heatmap[i + k][j + l];
				}
			}
			int value = sum / WEIGHTSUM;
			blurred_heatmap[i][j] = 0x00FF0000 | value << 24;
		}
	}
}
*/

/*
void Ped::Model::setupHeatmapSeq()
{
	int *hm = (int*)calloc(SIZE*SIZE, sizeof(int));
	int *shm = (int*)malloc(SCALED_SIZE*SCALED_SIZE*sizeof(int));
	int *bhm = (int*)malloc(SCALED_SIZE*SCALED_SIZE*sizeof(int));

	cudaMalloc((void **) &d_hm, SIZE*SIZE*sizeof(int));
	cudaMalloc((void **) &d_shm, SCALED_SIZE*SCALED_SIZE*sizeof(int));
	cudaMalloc((void **) &d_bhm, SCALED_SIZE*SCALED_SIZE*sizeof(int));

	cudaMemcpy(d_hm, hm, SIZE*SIZE*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_shm, shm, SCALED_SIZE*SCALED_SIZE*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_bhm, bhm, SCALED_SIZE*SCALED_SIZE*sizeof(int), cudaMemcpyHostToDevice);

	heatmap = (int**)malloc(SIZE*sizeof(int*));

	cudaMalloc((void ***)&d_heatmap, SIZE*sizeof(int*));
	cudaMemcpy(d_heatmap, heatmap, SIZE*sizeof(int*), cudaMemcpyHostToDevice);

	scaled_heatmap = (int**)malloc(SCALED_SIZE*sizeof(int*));
	blurred_heatmap = (int**)malloc(SCALED_SIZE*sizeof(int*));

	cudaMalloc((void ***)&d_scaled_heatmap, SCALED_SIZE*sizeof(int*));
	cudaMalloc((void ***)&d_blurred_heatmap, SCALED_SIZE*sizeof(int*));
	cudaMemcpy(d_scaled_heatmap, scaled_heatmap, SCALED_SIZE*sizeof(int*), cudaMemcpyHostToDevice);
	cudaMemcpy(d_blurred_heatmap, blurred_heatmap, SCALED_SIZE*sizeof(int*), cudaMemcpyHostToDevice);

	kernel_setupHeatmap<<<1,256>>>(hm, heatmap, SIZE);
	kernel_setupScaledHeatmap<<<1,256>>>(shm, scaled_heatmap, SCALED_SIZE);
	kernel_setupBlurredHeatmap<<<2,256>>>(bhm, blurred_heatmap, SCALED_SIZE);

	// Copy back to host
	cudaMemcpy(d_hm, hm, SIZE*SIZE*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(d_shm, shm, SCALED_SIZE*SCALED_SIZE*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(d_bhm, bhm, SCALED_SIZE*SCALED_SIZE*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(d_heatmap, heatmap, SIZE*sizeof(int*), cudaMemcpyDeviceToHost);
	cudaMemcpy(d_scaled_heatmap, scaled_heatmap, SCALED_SIZE*sizeof(int*), cudaMemcpyDeviceToHost);
	cudaMemcpy(d_blurred_heatmap, blurred_heatmap, SCALED_SIZE*sizeof(int*), cudaMemcpyDeviceToHost);

	// cudaFree(d_hm);
	// cudaFree(d_shm);
	// cudaFree(d_bhm);
	
	// cudaFree(d_geatmap);
	// cudaFree(d_scaled_heatmap);
	// cudaFree(d_blurred_heatmap);
	
    // free(hm);
    // free(shm);
    // free(bhm);

	// free(heatmap);
	// free(scaled_heatmap);
	// free(blurred_heatmap);
}

__global__ void Ped::Model::kernel_setupHeatmap(int *hm, int **heatmap, int size) {
	
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (id < size) {
		// mul = SIZE*id
		// atomicAdd(&heatmap[id], hm);
		// atomicAdd(&heatmap[id], mul);
		heatmap[id] = hm + size*id;

	}

}
__global__ void Ped::Model::kernel_setupScaledHeatmap(int *shm, int **scaled_heatmap, int scaled_size) {

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id < scaled_size) {
		//MIGHT CAUSE DATARACE
		// mul = SCALED_SIZE*id
		// atomicAdd(&scaled_heatmap[id], shm);
		// atomicAdd(&scaled_heatmap[id], mul);
		heatmap[id] = shm + scaled_size*id;
	}
}

__global__ void Ped::Model::kernel_setupBlurredHeatmap(int *bhm, int **blurred_heatmap, int scaled_size) {

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id < scaled_size) {
		// mul = SCALED_SIZE*id
		// atomicAdd(&blurred_heatmap[id], bhm);
		// atomicAdd(&blurred_heatmap[id], mul);
		heatmap[id] = bhm + scaled_size*id;
	}
}

// Updates the heatmap according to the agent positions
void Ped::Model::updateHeatmapSeq()
{
	for (int x = 0; x < SIZE; x++)
	{
		for (int y = 0; y < SIZE; y++)
		{
			// heat fades
			heatmap[y][x] = (int)round(heatmap[y][x] * 0.80);
		}
	}

	// Count how many agents want to go to each location
	for (int i = 0; i < agents.size(); i++)
	{
		Ped::Tagent* agent = agents[i];
		int x = agent->getDesiredX();
		int y = agent->getDesiredY();

		if (x < 0 || x >= SIZE || y < 0 || y >= SIZE)
		{
			continue;
		}

		// intensify heat for better color results
		heatmap[y][x] += 40;

	}

	for (int x = 0; x < SIZE; x++)
	{
		for (int y = 0; y < SIZE; y++)
		{
			heatmap[y][x] = heatmap[y][x] < 255 ? heatmap[y][x] : 255;
		}
	}

	// Scale the data for visual representation
	for (int y = 0; y < SIZE; y++)
	{
		for (int x = 0; x < SIZE; x++)
		{
			int value = heatmap[y][x];
			for (int cellY = 0; cellY < CELLSIZE; cellY++)
			{
				for (int cellX = 0; cellX < CELLSIZE; cellX++)
				{
					scaled_heatmap[y * CELLSIZE + cellY][x * CELLSIZE + cellX] = value;
				}
			}
		}
	}

	// Weights for blur filter
	const int w[5][5] = {
		{ 1, 4, 7, 4, 1 },
		{ 4, 16, 26, 16, 4 },
		{ 7, 26, 41, 26, 7 },
		{ 4, 16, 26, 16, 4 },
		{ 1, 4, 7, 4, 1 }
	};

#define WEIGHTSUM 273
	// Apply gaussian blurfilter		       
	for (int i = 2; i < SCALED_SIZE - 2; i++)
	{
		for (int j = 2; j < SCALED_SIZE - 2; j++)
		{
			int sum = 0;
			for (int k = -2; k < 3; k++)
			{
				for (int l = -2; l < 3; l++)
				{
					sum += w[2 + k][2 + l] * scaled_heatmap[i + k][j + l];
				}
			}
			int value = sum / WEIGHTSUM;
			blurred_heatmap[i][j] = 0x00FF0000 | value << 24;
		}
	}
}


int Ped::Model::getHeatmapSize() const {
	return SCALED_SIZE;
}
*/
void Ped::Model::setupHeatmapSeq()
{
	int *hm = (int*)calloc(SIZE*SIZE, sizeof(int));
	int *shm = (int*)malloc(SCALED_SIZE*SCALED_SIZE*sizeof(int));
	int *bhm = (int*)malloc(SCALED_SIZE*SCALED_SIZE*sizeof(int));

	kernel_setupHeatmap(hm, shm, bhm);


	// for (int i = 0; i < SIZE; i++)
	// {
	// 	heatmap[i] = hm + SIZE*i;
	// }
	// for (int i = 0; i < SCALED_SIZE; i++)
	// {
	// 	scaled_heatmap[i] = shm + SCALED_SIZE*i;
	// 	blurred_heatmap[i] = bhm + SCALED_SIZE*i;
	// }
}

// Updates the heatmap according to the agent positions
void Ped::Model::updateHeatmapSeq()
{

	// parallellize
	for (int x = 0; x < SIZE; x++)
	{
		for (int y = 0; y < SIZE; y++)
		{
			// heat fades
			heatmap[y][x] = (int)round(heatmap[y][x] * 0.80);
		}
	}

	// parallellize
	// Count how many agents want to go to each location
	for (int i = 0; i < agents.size(); i++)
	{
		Ped::Tagent* agent = agents[i];
		int x = agent->getDesiredX();
		int y = agent->getDesiredY();

		if (x < 0 || x >= SIZE || y < 0 || y >= SIZE)
		{
			continue;
		}

		// intensify heat for better color results
		heatmap[y][x] += 40;

	}

	// parallellize
	for (int x = 0; x < SIZE; x++)
	{
		for (int y = 0; y < SIZE; y++)
		{
			heatmap[y][x] = heatmap[y][x] < 255 ? heatmap[y][x] : 255;
		}
	}

	// Scale the data for visual representation
	for (int y = 0; y < SIZE; y++)
	{
		for (int x = 0; x < SIZE; x++)
		{
			int value = heatmap[y][x];
			for (int cellY = 0; cellY < CELLSIZE; cellY++)
			{
				for (int cellX = 0; cellX < CELLSIZE; cellX++)
				{
					scaled_heatmap[y * CELLSIZE + cellY][x * CELLSIZE + cellX] = value;
				}
			}
		}
	}

	// Weights for blur filter
	const int w[5][5] = {
		{ 1, 4, 7, 4, 1 },
		{ 4, 16, 26, 16, 4 },
		{ 7, 26, 41, 26, 7 },
		{ 4, 16, 26, 16, 4 },
		{ 1, 4, 7, 4, 1 }
	};

#define WEIGHTSUM 273
	// Apply gaussian blurfilter		       
	for (int i = 2; i < SCALED_SIZE - 2; i++)
	{
		for (int j = 2; j < SCALED_SIZE - 2; j++)
		{
			int sum = 0;
			for (int k = -2; k < 3; k++)
			{
				for (int l = -2; l < 3; l++)
				{
					sum += w[2 + k][2 + l] * scaled_heatmap[i + k][j + l];
				}
			}
			int value = sum / WEIGHTSUM;
			blurred_heatmap[i][j] = 0x00FF0000 | value << 24;
		}
	}
}

