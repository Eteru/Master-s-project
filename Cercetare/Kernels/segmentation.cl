
#include "Structs.h"

const sampler_t srcSampler = CLK_NORMALIZED_COORDS_FALSE |
CLK_ADDRESS_CLAMP_TO_EDGE |
CLK_FILTER_NEAREST;

inline void atomicAdd_g_f(volatile __global float *addr, float val)
{
	union {
		unsigned int u32;
		float f32;
	} next, expected, current;
	current.f32 = *addr;
	do {
		expected.f32 = current.f32;
		next.f32 = expected.f32 + val;
		current.u32 = atomic_cmpxchg((volatile __global unsigned int *)addr,
			expected.u32, next.u32);
	} while (current.u32 != expected.u32);
}

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

	uint sz = width * height;
	uint index = atomic_inc(&centroids[centroid_idx].count);

	buckets[centroid_idx * sz + index] = rgb;
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
	//printf("(%f, %f, %f)\n", centroids[centroid_idx].value_x, centroids[centroid_idx].value_y, centroids[centroid_idx].value_z);
	write_imagef(output, imgCoords, (float4)(centroids[centroid_idx].value_x, centroids[centroid_idx].value_y, centroids[centroid_idx].value_z, 1.f));
}

__kernel void update_centroids(
	__global read_write struct Centroid * centroids,
	__global read_write float3 *buckets,
	int width,
	int height,
	int pos
)
{
	const int2 ids = (int2)(get_global_id(0), get_global_id(1));
	const uint thread_lin = ids.y * width + ids.x;
	const uint id = get_global_id(0);
	const uint lid = get_local_id(0);
	unsigned long offset = pos * width * height + thread_lin;

	if (centroids[pos].count == 0 || thread_lin > centroids[pos].count)
	{
		return;
	}

	float resx = work_group_reduce_add(buckets[offset].x);
	float resy = work_group_reduce_add(buckets[offset].y);
	float resz = work_group_reduce_add(buckets[offset].z);

	if (0 == lid)
	{
		atomicAdd_g_f(&centroids[pos].sum_x, resx);
		atomicAdd_g_f(&centroids[pos].sum_y, resy);
		atomicAdd_g_f(&centroids[pos].sum_z, resz);
		//printf("Res: (%f, %f, %f) vs. Sum: (%f, %f, %f)\n", resx, resy, resz, centroids[pos].sum_x, centroids[pos].sum_y, centroids[pos].sum_z);
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (0 == ids.x && 0 == ids.y)
	{
		printf("[%d]: x(%f, %d), y(%f, %d), z(%f, %d)\n", pos, centroids[pos].sum_x, centroids[pos].count, centroids[pos].sum_y, centroids[pos].count, centroids[pos].sum_z, centroids[pos].count);
		centroids[pos].value_x = centroids[pos].sum_x / centroids[pos].count;
		centroids[pos].value_y = centroids[pos].sum_y / centroids[pos].count;
		centroids[pos].value_z = centroids[pos].sum_z / centroids[pos].count;

		//printf("Centroid[%d]: (%f, %f, %f) with count %d\n", pos, centroids[pos].value_x, centroids[pos].value_y, centroids[pos].value_z, centroids[pos].count);

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

	float3 rgb = read_imagef(input, srcSampler, imgCoords).xyz;
	float colorX = rgb.x >= value ? 1.0 : 0.0;
	float colorY = rgb.y >= value ? 1.0 : 0.0;
	float colorZ = rgb.z >= value ? 1.0 : 0.0;

	write_imagef(output, imgCoords, (float4)(colorX, colorY, colorZ, 1.0));
}