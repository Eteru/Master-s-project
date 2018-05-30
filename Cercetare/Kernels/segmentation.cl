
#include "Structs.h"

const sampler_t srcSampler = CLK_NORMALIZED_COORDS_FALSE |
							CLK_ADDRESS_CLAMP_TO_EDGE |
							CLK_FILTER_NEAREST;

__kernel void kmeans(
	read_only image2d_t input,
	__global read_write struct Centroid * centroids,
	__global write_only float3 *buckets,
	int width,
	int height,
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
		delta = (float3)(centroids[i].value_x, centroids[i].value_y, centroids[i].value_z) - rgb;
		delta *= delta;
		d = delta.x + delta.y + delta.z;

		if (d < dist)
		{
			dist = d;
			centroid_idx = i;
		}
	}

	//printf("centroid=%d, count=%d\n", centroid_idx, centroids[centroid_idx].count);

	uint idx = atomic_inc(&centroids[centroid_idx].count);
	//printf("[%d]: centroid=%d, count=%d\n", idx, centroid_idx, centroids[centroid_idx].count);
	buckets[centroid_idx * width * height + idx] = rgb;
}

__kernel void kmeans_draw(
	read_only image2d_t input,
	write_only image2d_t output,
	__constant struct Centroid * centroids,
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
		delta = (float3)(centroids[i].value_x, centroids[i].value_y, centroids[i].value_z) - rgb;
		delta *= delta;
		d = delta.x + delta.y + delta.z;

		if (d < dist) 
		{
			dist = d;
			centroid_idx = i;
		}
	}

	write_imagef(output, imgCoords, (float4)(centroids[centroid_idx].value_x, centroids[centroid_idx].value_y, centroids[centroid_idx].value_z, 1.f));
}

__kernel void update_centroids(
	__global read_write struct Centroid * centroids,
	__global read_only float3 *buckets,
	int width,
	int height,
	int centroids_no
)
{
	int pos = get_global_id(0);
	unsigned long offset = pos * width * height;

	if (centroids[pos].count == 0)
	{
		return;
	}
	
	centroids[pos].value_x = 0;
	centroids[pos].value_y = 0;
	centroids[pos].value_z = 0;

	for (int i = 0; i < centroids[pos].count; ++i)
	{
		centroids[pos].value_x += buckets[offset + i].x;
		centroids[pos].value_y += buckets[offset + i].y;
		centroids[pos].value_z += buckets[offset + i].z;
	}
	
	centroids[pos].value_x /= centroids[pos].count;
	centroids[pos].value_y /= centroids[pos].count;
	centroids[pos].value_z /= centroids[pos].count;
	
	//printf("\nSetting centroid[%d] count to 0\n\n", pos);
	centroids[pos].count = 0;
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