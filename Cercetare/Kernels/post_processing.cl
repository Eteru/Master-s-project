
#include "Structs.h"

const sampler_t srcSampler = CLK_NORMALIZED_COORDS_FALSE |
							CLK_ADDRESS_CLAMP_TO_EDGE |
							CLK_FILTER_NEAREST;

#define MAX_NEURONS 10
#define EPS 0.001

__kernel void noise_reduction(
	read_only image2d_t input,
	write_only image2d_t output,
	__global read_only struct Neuron * neurons,
	int neuron_count,
	int kernel_size)
{
	int HALF_FILTER_SIZE = kernel_size >> 1;
	const int2 imgCoords = (int2)(get_global_id(0), get_global_id(1));
	unsigned int hist[MAX_NEURONS] = {0,0,0,0,0,0,0,0,0,0};

	#pragma unroll 3
	for (int r = -HALF_FILTER_SIZE; r <= HALF_FILTER_SIZE; r++)
	{
		#pragma unroll 3
		for (int c = -HALF_FILTER_SIZE; c <= HALF_FILTER_SIZE; c++)
		{
			int2 coords = (int2)(imgCoords.x + r, imgCoords.y + c);
			uint4 rgba = read_imageui(input, srcSampler, coords);


			#pragma unroll 2
			for (int i = 0; i < neuron_count; ++i)
			{
				//printf("(%d, %d, %d) vs (%d, %d, %d)\n", rgba.x, rgba.y, rgba.z, neurons[i].x, neurons[i].y, neurons[i].z);
				if (rgba.x == neurons[i].x && rgba.y == neurons[i].y && rgba.z == neurons[i].z)
				{
					hist[i]++;
					break;
				}
			}
		}
	}

	int max_hist = 0;
	#pragma unroll
	for (int i = 0; i < neuron_count; ++i)
	{
		//printf("hist[%d]: %d\n", i, hist[i]);
		if (hist[i] > max_hist)
		{
			max_hist = i;
		}
	}

	write_imageui(output, imgCoords, (uint4)(neurons[max_hist].x, neurons[max_hist].y, neurons[max_hist].z, 1));
}