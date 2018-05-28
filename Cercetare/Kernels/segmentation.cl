
//#include "Structs.h"
struct Centroid
{
	float3 value;
	float3 sum;
	int count;
};

const sampler_t srcSampler = CLK_NORMALIZED_COORDS_FALSE |
							CLK_ADDRESS_CLAMP_TO_EDGE |
							CLK_FILTER_NEAREST;

__kernel void kmeans(
	read_only image2d_t input,
	__global read_write struct Centroid * centroids,
	int centroids_no
)
{
	float dist = FLT_MAX;
	int centroid_idx = -1;
	const int2 imgCoords = (int2)(get_global_id(0), get_global_id(1));

	float d;
	float3 delta;
	float3 rgb = read_imagef(input, srcSampler, imgCoords).xyz;

	for (int i = 0; i < centroids_no; ++i) 
	{
		delta = centroids[i].value - rgb;
		delta *= delta;
		d = delta.x + delta.y + delta.z;

		if (d < dist)
		{
			dist = d;
			centroid_idx = i;
		}
	}

	// TEST -------------
	//uint lid = get_local_id(0);
	//uint binId = get_group_id(0);
	//
	//uint group_offset = binId * bin_size;
	//uint maxval = 0;
	//
	//int prefix_sum_val = work_group_scan_inclusive_add(rgb.x);
	//barrier(CLK_GLOBAL_MEM_FENCE);
	//
	//int prefix_sum_val = work_group_scan_inclusive_add(rgb.y);
	//barrier(CLK_GLOBAL_MEM_FENCE);
	//
	//int prefix_sum_val = work_group_scan_inclusive_add(rgb.z);
	//barrier(CLK_GLOBAL_MEM_FENCE);
	//
	//// todo: sum of all workgroups
	//
	//
	//// ------------------
	//
	//centroids[centroid_idx].sum_x += rgb.x;
	//centroids[centroid_idx].sum_y += rgb.y;
	//centroids[centroid_idx].sum_z += rgb.z;
	//centroids[centroid_idx].count++;
	//atomic_add(&centroids[centroid_idx].sum_x, rgb.x);
	//atomic_add(&centroids[centroid_idx].sum_y, rgb.y);
	//atomic_add(&centroids[centroid_idx].sum_z, rgb.z);
	//atomic_inc(&centroids[centroid_idx].count);
}

__kernel void kmeans_draw(
	read_only image2d_t input,
	write_only image2d_t output,
	__global read_write struct Centroid * centroids,
	int centroids_no
)
{
	float dist = FLT_MAX;
	int centroid_idx = -1;
	const int2 imgCoords = (int2)(get_global_id(0), get_global_id(1));

	float d;
	float3 delta;
	float3 rgb = read_imagef(input, srcSampler, imgCoords).xyz;

	for (int i = 0; i < centroids_no; ++i) 
	{
		delta = centroids[i].value - rgb;
		delta *= delta;
		d = delta.x + delta.y + delta.z;

		if (d < dist) 
		{
			dist = d;
			centroid_idx = i;
		}
	}

	write_imagef(output, imgCoords, (float4)(centroids[centroid_idx].value, 1.f));
}

__kernel void update_centroids(
	__global read_write struct Centroid * centroids,
	int centroids_no
)
{
	int pos = get_global_id(0);

	if (pos < centroids_no && centroids[pos].count > 0) 
	{
		centroids[pos].value = centroids[pos].sum / centroids[pos].count;

		centroids[pos].sum = (float3)(0.f, 0.f, 0.f);
		centroids[pos].count = 0;
	}
}

__kernel void threshold(
	read_only image2d_t input,
	write_only image2d_t output,
	float value
)
{
	const int2 imgCoords = (int2)(get_global_id(0), get_global_id(1));

	float3 rgb = read_imagef(input, srcSampler, imgCoords).xyz;
	float colorX = rgb.x >= value ? 1.0 : 0.0;
	float colorY = rgb.y >= value ? 1.0 : 0.0;
	float colorZ = rgb.z >= value ? 1.0 : 0.0;

	write_imagef(output, imgCoords, (float4)(colorX, colorY, colorZ, 1.0));
}