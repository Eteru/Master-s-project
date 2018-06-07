
const sampler_t srcSampler = CLK_NORMALIZED_COORDS_FALSE |
							CLK_ADDRESS_CLAMP_TO_EDGE |
							CLK_FILTER_NEAREST;

__kernel void convolute(
	read_only image2d_t input,
	write_only image2d_t output,
	__constant read_only float * filter
)
{
	int HALF_FILTER_SIZE	= (int)(filter[0]) >> 1;
	const int2 imgCoords	= (int2)(get_global_id(0), get_global_id(1));

	int fIndex = 1;
	float3 sum = (float3)(0.f, 0.f, 0.f);

	//#pragma unroll 3
	for (int r = -HALF_FILTER_SIZE; r <= HALF_FILTER_SIZE; ++r) 
	{
		//#pragma unroll 3
		for (int c = -HALF_FILTER_SIZE; c <= HALF_FILTER_SIZE; ++c)
		{
			int2 coords = (int2)(imgCoords.x + r, imgCoords.y + c);
			float3 rgba = read_imagef(input, srcSampler, coords).xyz;

			sum += rgba * filter[fIndex];
			++fIndex;
		}
	}

	//printf("blurred: %f\n", sum.x);
	write_imagef(output, imgCoords, (float4)(sum, 1.f));
}

__kernel void color_smooth_filter(
	read_only image2d_t input,
	write_only image2d_t output,
	__constant read_only float * filter
)
{
	int HALF_FILTER_SIZE = (int)(filter[0]) >> 1;
	const int2 imgCoords = (int2)(get_global_id(0), get_global_id(1));

	int fIndex = 1;
	float3 sum = (float3)(0.f, 0.f, 0.f);

	#pragma unroll 3
	for (int r = -HALF_FILTER_SIZE; r <= HALF_FILTER_SIZE; ++r)
	{
		#pragma unroll 3
		for (int c = -HALF_FILTER_SIZE; c <= HALF_FILTER_SIZE; ++c) 
		{
			int2 coords = (int2)(imgCoords.x + r, imgCoords.y + c);
			float4 rgba = read_imagef(input, srcSampler, coords);

			sum += (float3)(rgba.x * filter[fIndex], rgba.y * filter[fIndex], rgba.z * filter[fIndex]);
			++fIndex;
		}
	}
	int sz = (filter[0] * filter[0]);
	sum /= sz;

	write_imagef(output, imgCoords, (float4)(sum.x, sum.y, sum.z, 1.f));
}

__kernel void sharpness_filter(
	read_only image2d_t input,
	write_only image2d_t output,
	__constant read_only float * filter
)
{
	int HALF_FILTER_SIZE = (int)(filter[0]) >> 1;
	const int2 imgCoords = (int2)(get_global_id(0), get_global_id(1));

	int fIndex = 1;
	float3 sum = (float3)(0.f, 0.f, 0.f);

	#pragma unroll 3
	for (int r = -HALF_FILTER_SIZE; r <= HALF_FILTER_SIZE; ++r) 
	{
		#pragma unroll 3
		for (int c = -HALF_FILTER_SIZE; c <= HALF_FILTER_SIZE; ++c) 
		{
			int2 coords = (int2)(imgCoords.x + r, imgCoords.y + c);
			float4 rgba = read_imagef(input, srcSampler, coords);

			sum += (float3)(rgba.x * filter[fIndex], rgba.y * filter[fIndex], rgba.z * filter[fIndex]);
			++fIndex;
		}
	}

	sum += read_imagef(input, srcSampler, imgCoords).xyz;
	write_imagef(output, imgCoords, (float4)(sum.xyz, 1.f));
}
