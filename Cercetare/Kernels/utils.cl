
const sampler_t srcSampler = 
	CLK_NORMALIZED_COORDS_FALSE |
	CLK_ADDRESS_CLAMP_TO_EDGE |
	CLK_FILTER_NEAREST;

const sampler_t outSampler =
	CLK_NORMALIZED_COORDS_FALSE |
	CLK_ADDRESS_CLAMP |
	CLK_FILTER_NEAREST;

const sampler_t scalingSampler =
	CLK_NORMALIZED_COORDS_TRUE |
	CLK_ADDRESS_NONE |
	CLK_FILTER_LINEAR;

__kernel void int_to_float(read_only image2d_t input, write_only image2d_t output)
{
	int2 imgCoords = (int2)(get_global_id(0), get_global_id(1));

	float4 imgVal = read_imagef(input, srcSampler, imgCoords);

	write_imagef(output, imgCoords, (float4)(imgVal.xyz, 1.0));
}

__kernel void grayscale(read_only image2d_t input, write_only image2d_t output)
{
	int2 imgCoords = (int2)(get_global_id(0), get_global_id(1));

	float4 imgVal = read_imagef(input, srcSampler, imgCoords);
	float color = 0.21 * imgVal.x + 0.72 * imgVal.y + 0.07 * imgVal.z;

	write_imagef(output, imgCoords, (float4)(color, color, color, 1.0));
}

__kernel void resize_image(
	read_only  image2d_t sourceImage,
	write_only image2d_t targetImage,
	uint w, uint h)
{
	int2 coords = { get_global_id(0), get_global_id(1) };
	float2 coords_normed = convert_float2(coords) / (float2)(w, h);

	float4 pixel = read_imagef(sourceImage, scalingSampler, coords_normed);
	//printf("%f\n", pixel.x);

	write_imagef(targetImage, coords, pixel);
}

__kernel void difference(
	read_only  image2d_t first,
	read_only  image2d_t second,
	write_only image2d_t output)
{
	int2 pos = { get_global_id(0), get_global_id(1) };

	float4 px1 = read_imagef(first, srcSampler, pos);
	float4 px2 = read_imagef(second, srcSampler, pos);
	float3 diff = (px1.xyz - px2.xyz);
	//diff = clamp(diff, 0, 255);
	//if (diff.x < 0)
	//{
	//	diff.y = -diff.x;
	//	diff.x = 0;
	//}
	//else
	//{
	//	diff.y = 0;
	//}

	//printf("(%d, %d, %d) - (%d, %d, %d) = (%d, %d, %d)\n",
	//	px1.x, px1.y, px1.z, px2.x, px2.y, px2.z, diff.x, diff.y, diff.z);

	write_imagef(output, pos, (float4)(diff, 1.f));
}

#define SURROUNDING_PIXELS 26
__kernel void find_extreme_points(
	read_only  image2d_t crt_dog,
	read_only  image2d_t prev_dog,
	read_only  image2d_t next_dog,
	write_only image2d_t output)
{
	int2 pos = { get_global_id(0), get_global_id(1) };
	float pixel = read_imagef(crt_dog, srcSampler, pos).x;

	//printf("%d\n", pixel);

	float4 max_no = { 0,0,0,1 };
	float4 max_yes = { 1,1,1,1 };

	float surrounding_values[SURROUNDING_PIXELS];

	uint index = 0;
	surrounding_values[index++] = read_imagef(prev_dog, srcSampler, pos).x;
	surrounding_values[index++] = read_imagef(next_dog, srcSampler, pos).x;

	for (int r = -1; r <= 1; ++r)
	{
		for (int c = -1; c <= 1; ++c)
		{
			int2 coords = (int2)(pos.x + r, pos.y + c);

			surrounding_values[index++] = read_imagef(crt_dog, srcSampler, coords).x;
			surrounding_values[index++] = read_imagef(prev_dog, srcSampler, coords).x;
			surrounding_values[index++] = read_imagef(next_dog, srcSampler, coords).x;
		}
	}

	if (fabs(pixel + 0.0491962) < 0.0001f)
	{
		printf("test: %f\n", read_imagef(prev_dog, srcSampler, (int2)(pos.x - 1, pos.y - 1)).x);
		for (int i = 0; i < SURROUNDING_PIXELS; ++i)
		{
			printf("(%f vs. %f)", pixel, surrounding_values[i]);
		}
	}

	bool is_max = false;
	bool is_min = false;

	if (pixel >= surrounding_values[0]  &&
		pixel >= surrounding_values[1]  &&
		pixel >= surrounding_values[2]  &&
		pixel >= surrounding_values[3]  &&
		pixel >= surrounding_values[4]  &&
		pixel >= surrounding_values[5]  &&
		pixel >= surrounding_values[6]  &&
		pixel >= surrounding_values[7]  &&
		pixel >= surrounding_values[8]  &&
		pixel >= surrounding_values[9]  &&
		pixel >= surrounding_values[10] &&
		pixel >= surrounding_values[11] &&
		pixel >= surrounding_values[12] &&
		pixel >= surrounding_values[13] &&
		pixel >= surrounding_values[14] &&
		pixel >= surrounding_values[15] &&
		pixel >= surrounding_values[16] &&
		pixel >= surrounding_values[17] &&
		pixel >= surrounding_values[18] &&
		pixel >= surrounding_values[19] &&
		pixel >= surrounding_values[20] &&
		pixel >= surrounding_values[21] &&
		pixel >= surrounding_values[22] &&
		pixel >= surrounding_values[23] &&
		pixel >= surrounding_values[24] &&
		pixel >= surrounding_values[25])
	{
		printf("max\n");
		is_max = true;
	}
	else if (pixel <= surrounding_values[0]  &&
			 pixel <= surrounding_values[1]  &&
			 pixel <= surrounding_values[2]  &&
			 pixel <= surrounding_values[3]  &&
			 pixel <= surrounding_values[4]  &&
			 pixel <= surrounding_values[5]  &&
			 pixel <= surrounding_values[6]  &&
			 pixel <= surrounding_values[7]  &&
			 pixel <= surrounding_values[8]  &&
			 pixel <= surrounding_values[9]  &&
			 pixel <= surrounding_values[10] &&
			 pixel <= surrounding_values[11] &&
			 pixel <= surrounding_values[12] &&
			 pixel <= surrounding_values[13] &&
			 pixel <= surrounding_values[14] &&
			 pixel <= surrounding_values[15] &&
			 pixel <= surrounding_values[16] &&
			 pixel <= surrounding_values[17] &&
			 pixel <= surrounding_values[18] &&
			 pixel <= surrounding_values[19] &&
			 pixel <= surrounding_values[20] &&
			 pixel <= surrounding_values[21] &&
			 pixel <= surrounding_values[22] &&
			 pixel <= surrounding_values[23] &&
			 pixel <= surrounding_values[24] &&
			 pixel <= surrounding_values[25])
	{
		printf("min\n");
		is_min = true;
	}

	if (is_min || is_max)
	{
		write_imagef(output, pos, max_yes);
		return;
	}

	write_imagef(output, pos, max_no);
}