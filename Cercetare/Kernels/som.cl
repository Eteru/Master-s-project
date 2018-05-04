
#include "Structs.h"

const sampler_t srcSampler = CLK_NORMALIZED_COORDS_FALSE |
							CLK_ADDRESS_CLAMP_TO_EDGE |
							CLK_FILTER_NEAREST;

__kernel void find_bmu(
	read_only uint3 value,
	__global read_only struct Neuron * neurons,
	__global read_write float * distances,
	__global write_only int * bmu_idx,
	int neuron_count
)
{
	int pos = get_global_id(0);

	if (pos < neuron_count) 
	{
		uint dist_X = value.x - neurons[pos].x;
		uint dist_Y = value.y - neurons[pos].y;
		uint dist_Z = value.z - neurons[pos].z;

		distances[pos] = sqrt((float)(dist_X * dist_X + dist_Y * dist_Y + dist_Z * dist_Z));
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
	read_only uint3 value,
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

			neurons[pos].x += learning_rate * influence * (int)(value.x - neurons[pos].x);
			neurons[pos].y += learning_rate * influence * (int)(value.y - neurons[pos].y);
			neurons[pos].z += learning_rate * influence * (int)(value.z - neurons[pos].z);
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

	uint4 rgba = convert_uint4_sat_rte(read_imagef(input, srcSampler, imgCoords) * 255);
	//printf("%f, %f, %f\n", neurons[0].x / 255.f, neurons[0].y / 255.f, neurons[0].z / 255.f);

	float d;
	uint dist_X;
	uint dist_Y;
	uint dist_Z;

	for (int i = 0; i < neuron_count; ++i)
	{
		dist_X = rgba.x - neurons[i].x;
		dist_Y = rgba.y - neurons[i].y;
		dist_Z = rgba.z - neurons[i].z;

		d = sqrt((float)(dist_X * dist_X + dist_Y * dist_Y + dist_Z * dist_Z));

		if (d < dist)
		{
			dist = d;
			neuron_idx = i;
		}
	}

	float3 res = { (float)neurons[neuron_idx].x / 255.f, (float)neurons[neuron_idx].y / 255.f, (float)neurons[neuron_idx].z / 255.f };

	write_imagef(output, imgCoords, (float4)(res, 1.f));
}
