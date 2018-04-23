
const sampler_t srcSampler = 
	CLK_NORMALIZED_COORDS_FALSE |
	CLK_ADDRESS_CLAMP_TO_EDGE |
	CLK_FILTER_NEAREST;

const sampler_t outSampler =
	CLK_NORMALIZED_COORDS_FALSE |
	CLK_ADDRESS_CLAMP |
	CLK_FILTER_NEAREST;

__kernel void grayscale(read_only image2d_t input, write_only image2d_t output)
{
	int2 imgCoords = (int2)(get_global_id(0), get_global_id(1));

	float4 imgVal = read_imagef(input, srcSampler, imgCoords);
	float color = 0.21 * imgVal.x + 0.72 * imgVal.y + 0.07 * imgVal.z;

	write_imagef(output, imgCoords, (float4)(color, color, color, 1.0));
}

__kernel void resize_image(
	__read_only  image2d_t sourceImage,
	__write_only image2d_t targetImage,
	float reduce_by)
{
	int2 posOut = { get_global_id(0), get_global_id(1) };
	int2 posIn = { posOut.x * reduce_by, posOut.y * reduce_by };

	uint4 pixel = read_imageui(sourceImage, srcSampler, posIn);
	write_imageui(targetImage, posOut, pixel);
}