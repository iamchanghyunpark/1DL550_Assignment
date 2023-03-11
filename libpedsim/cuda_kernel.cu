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

	//cudaMemcpy(d_desiredX, desiredX, agents.size()*sizeof(int), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_desiredY, desiredY, agents.size()*sizeof(int), cudaMemcpyHostToDevice);

	// Copy from CPU to GPU
	cudaMemcpy(d_heatmap, heatmap[0], SIZE*SIZE*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_scaled_heatmap, scaled_heatmap[0], SCALED_SIZE*SCALED_SIZE*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_blurred_heatmap, blurred_heatmap[0], SCALED_SIZE*SCALED_SIZE*sizeof(int), cudaMemcpyHostToDevice);

	//cudaDeviceSynchronize();

}

/* ---------------------------
	UPPDATE HEATMAP FUNCTIONS
  --------------------------*/ 

__global__ void kernel_fade(int *dev_heatmap)
{
	__shared__ int hm[SIZE];
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < SIZE*SIZE) {
	hm[threadIdx.x] = dev_heatmap[i];
	int value = (int)round(hm[threadIdx.x] * 0.8);
	hm[threadIdx.x] = value;
	dev_heatmap[i] = hm[threadIdx.x];

	}
}

__global__ void kernel_agents(int *dev_heatmap, int agentSize, int *desiredX, int *desiredY)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < agentSize){
		int x = desiredX[i];
		int y = desiredY[i];

		if (x < 0 || x >= SIZE || y < 0 || y >= SIZE)
		{
			return;
		}
		atomicAdd(&dev_heatmap[y * SIZE + x], 40); //UNCLEAR FLOAT ??
	}
}

__global__ void kernel_clip(int *dev_heatmap)
{
	__shared__ int hm[SIZE];
	int i = blockIdx.x * blockDim.x + threadIdx.x;	
	if (i < SIZE*SIZE){
		hm[threadIdx.x] = dev_heatmap[i];
		hm[threadIdx.x] = hm[threadIdx.x] < 255 ? hm[threadIdx.x] : 255;
		dev_heatmap[i] = hm[threadIdx.x];
	}
}

__global__ void kernel_scale(int *dev_heatmap, int *dev_scaled_heatmap)
{
	__shared__ int heatmap[SIZE];
	__shared__ int scaled_heatmap[SCALED_SIZE];
	int i = blockIdx.x * blockDim.x + threadIdx.x;	
	if(i < SIZE*SIZE) {
		int x = threadIdx.x;
		int y = blockIdx.x;
		heatmap[x] = dev_heatmap[i];
		int value = heatmap[x];
		for (int cellY = 0; cellY < CELLSIZE; cellY++)
		{
			for (int cellX = 0; cellX < CELLSIZE; cellX++)
			{
				int index = cellY + x * CELLSIZE + cellX;
				if (index >= SCALED_SIZE) continue;
				scaled_heatmap[index] = value;
			}
		}

		for (int cellY = 0; cellY < CELLSIZE; cellY++)
		{
			for (int cellX = 0; cellX < CELLSIZE; cellX++)
			{
			int index = cellY + x * CELLSIZE + cellX;
			if (index >= SCALED_SIZE) continue;
			int global_x_index = x * CELLSIZE + cellX;
			int global_y_index = y * CELLSIZE + cellY;
            global_y_index *= SCALED_SIZE;
            int global_index = global_x_index + global_y_index;
			int value = shm[index];
			dev_scaled_heatmap[global_index] = shm[index];
			}
		}
	}
}

__global__ void kernel_blur(int *dev_blurred_heatmap, int *dev_scaled_heatmap)
{
	//weights for blur
	const int w[5][5] = {
		{ 1, 4, 7, 4, 1 },
		{ 4, 16, 26, 16, 4 },
		{ 7, 26, 41, 26, 7 },
		{ 4, 16, 26, 16, 4 },
		{ 1, 4, 7, 4, 1 }
	};
	int i = blockIdx.x * blockDim.x + threadIdx.x;	
	if (i == SIZE*SIZE) return;
	int thread_x = threadIdx.x;
	int block_y = blockIdx.x;
    for (int cellY = 0; cellY < CELLSIZE; cellY++)
    {
        for (int cellX = 0; cellX < CELLSIZE; cellX++)
        {
            int sum = 0;
            int x = thread_x + cellX;
            int y = block_y + cellY;
            if (x < 2 || x > SCALED_SIZE - 2) continue;
            if (y < 2 || y > SCALED_SIZE - 2) continue;
            for (int k = -2; k < 3; k++)
            {
                for (int l = -2; l < 3; l++)
                {
                    int weight_val = w[2 + k][2 + l];
                    int cur_y = y + k;
                    int cur_x = x + l;
                    int global_cur_y = cur_y * SCALED_SIZE;
                    int global_cur_x = cur_x;
                    int global_index = global_cur_x + global_cur_y;
                    sum += weight_val * dev_scaled_heatmap[global_index];
                }
            }
            int value = sum / 273;
            dev_blurred_heatmap[y*SCALED_SIZE + x] = 0x00FF0000 | value << 24;
        }
    }
}


void Ped::Model::updateHeatmapCuda() 
{
	int blocks = 1 + agents.size() / 1024;
	cout << "HEJSAN :) 6 " << '\n';
	cudaMemcpyAsync(d_desiredX, desiredX, agents.size()*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_desiredY, desiredY, agents.size()*sizeof(int), cudaMemcpyHostToDevice);

	// Fade heatmap
	kernel_fade<<<SIZE, SIZE>>>(d_heatmap);

	// Desired positions
	kernel_agents<<<blocks, SIZE>>>(d_heatmap, agentsSize, d_desiredX, d_desiredY);
	cout << "HEJSAN :) " << '\n';

	//Clip heatmap
	kernel_clip<<<SIZE, SIZE>>>(d_heatmap);
	cout << "HEJSAN :)2 " << '\n';

	//Scale heatmap
	kernel_scale<<<SIZE, SIZE>>>(d_heatmap, d_scaled_heatmap);
	cout << "HEJSAN :)3 " << '\n';

	// // Blur heatmap
	kernel_blur<<<SIZE,SIZE>>>(d_blurred_heatmap, d_scaled_heatmap);
	cout << "HEJSAN :) 4" << '\n';

	cudaError_t status;
	status = cudaMemcpyAsync(blurred_heatmap[0], d_blurred_heatmap, SCALED_SIZE * SCALED_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	if(status != cudaSuccess)
	{
		fprintf(stderr, "memcopy faillllll");
	} else {fprintf(stderr, "noo fail");}
	cout << "HEJSAN :) 5" << '\n';
}
