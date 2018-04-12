
const sampler_t srcSampler = CLK_NORMALIZED_COORDS_FALSE |
							CLK_ADDRESS_CLAMP_TO_EDGE |
							CLK_FILTER_NEAREST;

__kernel void grayscale(read_only image2d_t input, write_only image2d_t output)
{
	int2 imgCoords = (int2)(get_global_id(0), get_global_id(1));

	float4 imgVal = read_imagef(input, srcSampler, imgCoords);
	float color = 0.21 * imgVal.x + 0.72 * imgVal.y + 0.07 * imgVal.z;

	write_imagef(output, imgCoords, (float4)(color, color, color, 1.0));
}
