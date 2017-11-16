
#include "structs.hpp"

const sampler_t srcSampler = CLK_NORMALIZED_COORDS_FALSE |
							CLK_ADDRESS_CLAMP_TO_EDGE |
							CLK_FILTER_NEAREST;

void atomicAdd_g_f(volatile __global float *addr, float val)
{
	union {
		unsigned int u32;
		float        f32;
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
	int centroids_no
)
{
	float dist = FLT_MAX;
	int centroid_idx = -1;
	const int2 imgCoords = (int2)(get_global_id(0), get_global_id(1));

	float4 rgba = read_imagef(input, srcSampler, imgCoords);


	for (int i = 0; i < centroids_no; ++i) {
		float d = centroids[i].x * centroids[i].x + rgba.x * rgba.x - 2 * centroids[i].x * rgba.x +
			centroids[i].y * centroids[i].y + rgba.y * rgba.y - 2 * centroids[i].y * rgba.y +
			centroids[i].z * centroids[i].z + rgba.z * rgba.z - 2 * centroids[i].z * rgba.z;
			//sqrt(pow(centroids[i].x - rgba.x, 2) + pow(centroids[i].y - rgba.y, 2) + pow(centroids[i].z - rgba.z, 2));

		if (d < dist) {
			dist = d;
			centroid_idx = i;
		}
	}

	atomicAdd_g_f(&centroids[centroid_idx].sum_x, rgba.x);
	atomicAdd_g_f(&centroids[centroid_idx].sum_y, rgba.y);
	atomicAdd_g_f(&centroids[centroid_idx].sum_z, rgba.z);
	atomic_inc(&centroids[centroid_idx].count);
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


	for (int i = 0; i < centroids_no; ++i) {
		float d = centroids[i].x * centroids[i].x + rgba.x * rgba.x - 2 * centroids[i].x * rgba.x +
			centroids[i].y * centroids[i].y + rgba.y * rgba.y - 2 * centroids[i].y * rgba.y +
			centroids[i].z * centroids[i].z + rgba.z * rgba.z - 2 * centroids[i].z * rgba.z;

		if (d < dist) {
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

	if (pos < centroids_no && centroids[pos].count > 0) {
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

	float4 rgba = read_imagef(input, srcSampler, imgCoords);
	float color = rgba.x >= value ? 1.0 : 0.0;

	write_imagef(output, imgCoords, (float4)(color, color, color, 1.0));
}