
#include "Structs.h"

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

	float4 rgba = read_imagef(input, srcSampler, imgCoords);


	for (int i = 0; i < centroids_no; ++i) 
	{
		float d = centroids[i].x * centroids[i].x + rgba.x * rgba.x - 2 * centroids[i].x * rgba.x +
			centroids[i].y * centroids[i].y + rgba.y * rgba.y - 2 * centroids[i].y * rgba.y +
			centroids[i].z * centroids[i].z + rgba.z * rgba.z - 2 * centroids[i].z * rgba.z;
			//sqrt(pow(centroids[i].x - rgba.x, 2) + pow(centroids[i].y - rgba.y, 2) + pow(centroids[i].z - rgba.z, 2));

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
	//int prefix_sum_val = work_group_scan_inclusive_add(rgba.x);
	//barrier(CLK_GLOBAL_MEM_FENCE);
	//
	//int prefix_sum_val = work_group_scan_inclusive_add(rgba.y);
	//barrier(CLK_GLOBAL_MEM_FENCE);
	//
	//int prefix_sum_val = work_group_scan_inclusive_add(rgba.z);
	//barrier(CLK_GLOBAL_MEM_FENCE);
	//
	//// todo: sum of all workgroups
	//
	//
	//// ------------------
	//
	//centroids[centroid_idx].sum_x += rgba.x;
	//centroids[centroid_idx].sum_y += rgba.y;
	//centroids[centroid_idx].sum_z += rgba.z;
	//centroids[centroid_idx].count++;
	//atomic_add(&centroids[centroid_idx].sum_x, rgba.x);
	//atomic_add(&centroids[centroid_idx].sum_y, rgba.y);
	//atomic_add(&centroids[centroid_idx].sum_z, rgba.z);
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

	float4 rgba = read_imagef(input, srcSampler, imgCoords);


	for (int i = 0; i < centroids_no; ++i) 
	{
		float d = centroids[i].x * centroids[i].x + rgba.x * rgba.x - 2 * centroids[i].x * rgba.x +
			centroids[i].y * centroids[i].y + rgba.y * rgba.y - 2 * centroids[i].y * rgba.y +
			centroids[i].z * centroids[i].z + rgba.z * rgba.z - 2 * centroids[i].z * rgba.z;

		if (d < dist) 
		{
			dist = d;
			centroid_idx = i;
		}
	}

	write_imagef(output, imgCoords, (float4)(centroids[centroid_idx].x, centroids[centroid_idx].y, centroids[centroid_idx].z, 1));
}

__kernel void update_centroids(
	__global read_write struct Centroid * centroids,
	int centroids_no
)
{
	int pos = get_global_id(0);

	if (pos < centroids_no && centroids[pos].count > 0) 
	{
		centroids[pos].x = centroids[pos].sum_x / centroids[pos].count;
		centroids[pos].y = centroids[pos].sum_y / centroids[pos].count;
		centroids[pos].z = centroids[pos].sum_z / centroids[pos].count;

		centroids[pos].sum_x = 0;
		centroids[pos].sum_y = 0;
		centroids[pos].sum_z = 0;
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

	uint4 rgba = read_imageui(input, srcSampler, imgCoords);
	float colorX = rgba.x >= value ? 1.0 : 0.0;
	float colorY = rgba.y >= value ? 1.0 : 0.0;
	float colorZ = rgba.z >= value ? 1.0 : 0.0;

	write_imageui(output, imgCoords, (uint4)(colorX, colorY, colorZ, 1.0));
}