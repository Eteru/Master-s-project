
const sampler_t srcSampler = CLK_NORMALIZED_COORDS_FALSE |
							CLK_ADDRESS_CLAMP_TO_EDGE |
							CLK_FILTER_NEAREST;

__kernel void grayscale(read_only image2d_t input, write_only image2d_t output)
{
	int2 imgCoords = (int2)(get_global_id(0), get_global_id(1));

	uint4 imgVal = read_imageui(input, srcSampler, imgCoords);

	write_imageui(output, imgCoords, (uint4)(0.21 * imgVal.x + 0.72 * imgVal.y + 0.07 * imgVal.z, 0.0, 0.0, 1.0));
}

__kernel void convolute(
	read_only image2d_t input,
	write_only image2d_t output,
	__constant read_only float * filter
)
{
	int HALF_FILTER_SIZE	= (int)(filter[0]) >> 1;
	const int2 imgCoords	= (int2)(get_global_id(0), get_global_id(1));

	int fIndex = 1;
	int3 sum = (int3)(0, 0, 0);

	#pragma unroll 3
	for (int r = -HALF_FILTER_SIZE; r <= HALF_FILTER_SIZE; r++) 
	{
		#pragma unroll 3
		for (int c = -HALF_FILTER_SIZE; c <= HALF_FILTER_SIZE; c++) 
		{
			int2 coords = (int2)(imgCoords.x + r, imgCoords.y + c);
			uint4 rgba = read_imageui(input, srcSampler, coords);

			sum += (int3)(rgba.x * filter[fIndex], rgba.y * filter[fIndex], rgba.z * filter[fIndex]);
			++fIndex;
		}
	}

	sum = clamp(sum, 0, 255);
	write_imageui(output, imgCoords, (uint4)(sum.x, sum.y, sum.z, 1));
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
	int3 sum = (int3)(0, 0, 0);

	#pragma unroll 3
	for (int r = -HALF_FILTER_SIZE; r <= HALF_FILTER_SIZE; r++)
	{
		#pragma unroll 3
		for (int c = -HALF_FILTER_SIZE; c <= HALF_FILTER_SIZE; c++) 
		{
			int2 coords = (int2)(imgCoords.x + r, imgCoords.y + c);
			uint4 rgba = read_imageui(input, srcSampler, coords);

			sum += (int3)(rgba.x * filter[fIndex], rgba.y * filter[fIndex], rgba.z * filter[fIndex]);
			++fIndex;
		}
	}
	int sz = (filter[0] * filter[0]);
	sum /= sz;

	sum = clamp(sum, 0, 255);
	write_imageui(output, imgCoords, (uint4)(sum.x, sum.y, sum.z, 1));
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
	int3 sum = (int3)(0, 0, 0);

	#pragma unroll 3
	for (int r = -HALF_FILTER_SIZE; r <= HALF_FILTER_SIZE; r++) 
	{
		#pragma unroll 3
		for (int c = -HALF_FILTER_SIZE; c <= HALF_FILTER_SIZE; c++) 
		{
			int2 coords = (int2)(imgCoords.x + r, imgCoords.y + c);
			uint4 rgba = read_imageui(input, srcSampler, coords);

			sum += (int3)(rgba.x * filter[fIndex], rgba.y * filter[fIndex], rgba.z * filter[fIndex]);
			++fIndex;
		}
	}

	sum = clamp(sum, 0, 255);
	sum += convert_int3(read_imageui(input, srcSampler, imgCoords).xyz);
	write_imageui(output, imgCoords, (uint4)(sum.x, sum.y, sum.z, 1));
}
