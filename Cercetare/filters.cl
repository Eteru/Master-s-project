
const sampler_t srcSampler = CLK_NORMALIZED_COORDS_FALSE |
							CLK_ADDRESS_CLAMP_TO_EDGE |
							CLK_FILTER_NEAREST;

__kernel void grayscale(read_only image2d_t input, write_only image2d_t output)
{
	int2 imgCoords = (int2)(get_global_id(0), get_global_id(1));

	float4 imgVal = read_imagef(input, srcSampler, imgCoords);

	write_imagef(output, imgCoords, (float4)(0.21 * imgVal.x + 0.72 * imgVal.y + 0.07 * imgVal.z, 0.0, 0.0, 1.0));
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
	float4 sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

	#pragma unroll
	for (int r = -HALF_FILTER_SIZE; r <= HALF_FILTER_SIZE; r++) {
		#pragma unroll
		for (int c = -HALF_FILTER_SIZE; c <= HALF_FILTER_SIZE; c++) {
			int2 coords = (int2)(imgCoords[0] + r, imgCoords[1] + c);
			float4 rgba = read_imagef(input, srcSampler, coords);

			sum += rgba * filter[fIndex];
			++fIndex;
		}
	}

	sum.w = 1.0;
	write_imagef(output, imgCoords, sum);
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
	float4 sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

	#pragma unroll
	for (int r = -HALF_FILTER_SIZE; r <= HALF_FILTER_SIZE; r++) {
		#pragma unroll
		for (int c = -HALF_FILTER_SIZE; c <= HALF_FILTER_SIZE; c++) {
			int2 coords = (int2)(imgCoords[0] + r, imgCoords[1] + c);
			float4 rgba = read_imagef(input, srcSampler, coords);

			sum += rgba * filter[fIndex];
			++fIndex;
		}
	}
	sum /= (filter[0] * filter[0]);
	sum.w = 1.0;
	write_imagef(output, imgCoords, sum);
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
	float4 sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

	#pragma unroll
	for (int r = -HALF_FILTER_SIZE; r <= HALF_FILTER_SIZE; r++) {
		#pragma unroll
		for (int c = -HALF_FILTER_SIZE; c <= HALF_FILTER_SIZE; c++) {
			int2 coords = (int2)(imgCoords[0] + r, imgCoords[1] + c);
			float4 rgba = read_imagef(input, srcSampler, coords);

			sum += rgba * filter[fIndex];
			++fIndex;
		}
	}

	sum.w = 1.0;
	write_imagef(output, imgCoords, read_imagef(input, srcSampler, imgCoords) + sum);
}
