
#include "Structs.h"

const sampler_t srcSampler = CLK_NORMALIZED_COORDS_FALSE |
							CLK_ADDRESS_CLAMP_TO_EDGE |
							CLK_FILTER_NEAREST;

__kernel void find_bmu(
	read_only float3 value,
	__global read_only struct Neuron * neurons,
	__global read_write float * distances,
	__global write_only int * bmu_idx,
	int neuron_count
)
{
	int pos = get_global_id(0);
	float3 dist;

	if (pos < neuron_count) 
	{
		dist = value - (float3)(neurons[pos].value_x, neurons[pos].value_y, neurons[pos].value_z);
		dist *= dist;

		distances[pos] = dist.x + dist.y + dist.z;
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (0 == pos) 
	{
		int crt_idx = 0;
		float crt_value = HUGE_VALF;

		for (int i = 0; i < neuron_count; ++i)
		{
			if (distances[i] < crt_value)
			{
				crt_idx = i;
				crt_value = distances[i];
			}
		}

		bmu_idx[0] = crt_idx;
	}
}

__kernel void update_weights(
	read_only float3 value,
	__global read_only int * bmu_idx,
	__global read_write struct Neuron * neurons,
	int neuron_count,
	float neigh_distance,
	float learning_rate
)
{
	int pos = get_global_id(0);

	if (pos < neuron_count)
	{
		int dist = abs(pos - bmu_idx[0]);

		if (dist < neigh_distance) 
		{
			float influence = exp(-(dist * dist) / (2.f * (neigh_distance * neigh_distance)));

			neurons[pos].value_x += learning_rate * influence * (value.x - neurons[pos].value_x);
			neurons[pos].value_y += learning_rate * influence * (value.y - neurons[pos].value_y);
			neurons[pos].value_z += learning_rate * influence * (value.z - neurons[pos].value_z);
		}
	}
}

__kernel void som_draw(
	read_only image2d_t input,
	write_only image2d_t output,
	__global read_write struct Neuron * neurons,
	int neuron_count
)
{
	float dist = HUGE_VALF;
	int neuron_idx = -1;
	const int2 imgCoords = (int2)(get_global_id(0), get_global_id(1));

	float3 rgb = read_imagef(input, srcSampler, imgCoords).xyz;

	float d;
	float3 distance;

	for (int i = 0; i < neuron_count; ++i)
	{
		distance = rgb - (float3)(neurons[i].value_x, neurons[i].value_y, neurons[i].value_z);

		distance *= distance;

		d = distance.x  + distance.y  + distance.z;

		if (d < dist)
		{
			dist = d;
			neuron_idx = i;
		}
	}

	write_imagef(output, imgCoords, (float4)(neurons[neuron_idx].value_x, neurons[neuron_idx].value_y, neurons[neuron_idx].value_z, 1.f));
}
