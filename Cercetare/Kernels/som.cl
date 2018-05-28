
struct Neuron
{
	float3 value;
};

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
		dist = value - neurons[pos].value;
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

			neurons[pos].value += learning_rate * influence * (value - neurons[pos].value);
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
		distance = rgb - neurons[i].value;

		distance *= distance;

		d = distance.x  + distance.y  + distance.z;

		if (d < dist)
		{
			dist = d;
			neuron_idx = i;
		}
	}

	write_imagef(output, imgCoords, (float4)(neurons[neuron_idx].value, 1.f));
}
