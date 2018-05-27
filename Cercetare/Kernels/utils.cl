
#include "Structs.h"

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

#define FLOAT_CORRECTNESS 1000

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
	float3 diff = (px2.xyz - px1.xyz);

	//printf("(%d, %d, %d) - (%d, %d, %d) = (%d, %d, %d)\n",
	//	px1.x, px1.y, px1.z, px2.x, px2.y, px2.z, diff.x, diff.y, diff.z);
	//printf("%f, %f, %f\n", diff.x, diff.y, diff.z);
	write_imagef(output, pos, (float4)(diff, 1.f));
}

#define SURROUNDING_PIXELS 26
__kernel void find_extreme_points(
	read_only  image2d_t crt_dog,
	read_only  image2d_t prev_dog,
	read_only  image2d_t next_dog,
	write_only image2d_t output)
{
	const float CONTRAST_THRESHOLD = 0.0001f;
	const float CURVATURE_THRESHOLD = 5.f;
	//const float curvature_thresh = (CURVATURE_THRESHOLD + 1)*(CURVATURE_THRESHOLD + 1) / CURVATURE_THRESHOLD;
	const float curvature_thresh = 10.f;

	int2 pos = { get_global_id(0), get_global_id(1) };
	float pixel = read_imagef(crt_dog, srcSampler, pos).x;

	//printf("%f\n", pixel);

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
			if (r == 0 && c == 0)
			{
				continue;
			}

			int2 coords = (int2)(pos.x + r, pos.y + c);

			surrounding_values[index++] = read_imagef(crt_dog, srcSampler, coords).x;
			surrounding_values[index++] = read_imagef(prev_dog, srcSampler, coords).x;
			surrounding_values[index++] = read_imagef(next_dog, srcSampler, coords).x;
		}
	}

	//if (fabs(pixel + 0.0491962) < 0.0001f)
	//{
	//	printf("test: %f\n", read_imagef(prev_dog, srcSampler, (int2)(pos.x - 1, pos.y - 1)).x);
	//	for (int i = 0; i < SURROUNDING_PIXELS; ++i)
	//	{
	//		printf("(%f vs. %f)", pixel, surrounding_values[i]);
	//	}
	//}

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
		pixel >= surrounding_values[25] /*&&
		fabs(pixel) > CONTRAST_THRESHOLD*/)
	{
		//printf("max %f\n", pixel);
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
			 pixel <= surrounding_values[25] /*&&
			 fabs(pixel) > CONTRAST_THRESHOLD*/)
	{
		//printf("min %f\n", pixel);

		is_min = true;
	}

	if (is_min || is_max)
	{
		float dxx =
			read_imagef(crt_dog, srcSampler, (int2)(pos.x, pos.y - 1)).x +
			read_imagef(crt_dog, srcSampler, (int2)(pos.x, pos.y + 1)).x -
			2.f * read_imagef(crt_dog, srcSampler, pos).x;

		float dyy =
			read_imagef(crt_dog, srcSampler, (int2)(pos.x - 1, pos.y)).x +
			read_imagef(crt_dog, srcSampler, (int2)(pos.x + 1, pos.y)).x -
			2.f * read_imagef(crt_dog, srcSampler, pos).x;

		float dxy =
			(read_imagef(crt_dog, srcSampler, (int2)(pos.x - 1, pos.y - 1)).x +
			read_imagef(crt_dog, srcSampler, (int2)(pos.x + 1, pos.y + 1)).x -
			read_imagef(crt_dog, srcSampler, (int2)(pos.x - 1, pos.y + 1)).x -
			read_imagef(crt_dog, srcSampler, (int2)(pos.x + 1, pos.y - 1)).x)
			* 4.f;

		float trH = dxx + dyy;
		float detH = dxx * dyy - dxy*dxy;

		float curvature_ratio = trH * trH / detH;

		if (!signbit(detH) && curvature_ratio <= curvature_thresh)
		{
			write_imagef(output, pos, max_yes);
			return;
		}

		//printf("[%d, %d] vs [%f, %f]\n", pos.x, pos.y, dxx, dyy);
		//printf("Discarded point: %f, detH=%f, curvature_ratio=%f\n", pixel, detH, curvature_ratio);
	}

	write_imagef(output, pos, max_no);
}

__kernel void compute_magnitude_and_orientation(
	read_only  image2d_t input,
	write_only image2d_t magnitude,
	write_only image2d_t orientation)
{
	int2 pos = { get_global_id(0), get_global_id(1) };

	float dx = read_imagef(input, srcSampler, (int2)(pos.x + 1, pos.y)).x -
				read_imagef(input, srcSampler, (int2)(pos.x - 1, pos.y)).x;
	float dy = read_imagef(input, srcSampler, (int2)(pos.x, pos.y + 1)).x -
				read_imagef(input, srcSampler, (int2)(pos.x, pos.y - 1)).x;

	float magn = FLOAT_CORRECTNESS * sqrt(dx*dx + dy*dy);
	float orien = dx == 0.f ? 0.f : atan(dy / dx);

	//printf("dx=%f, dy=%f, magn=%f, orien=%f\n", dx, dy, magn, orien);

	write_imagef(magnitude, pos, (float4)(magn, magn, magn, 1.f));
	write_imagef(orientation, pos, (float4)(orien, orien, orien, 1.f));
}

__kernel void compute_magnitude_and_orientation_interp(
	read_only  image2d_t input,
	write_only image2d_t magnitude,
	write_only image2d_t orientation,
	const unsigned int width,
	const unsigned int height)
{
	int2 pos = { get_global_id(0), get_global_id(1) };
	float2 pos_f = (float2)(pos.x + 0.5f, pos.y + 0.5f) / (float2)(width, height);
	float2 pos_f_p_1 = (float2)(pos.x + 0.5f + 1, pos.y + 0.5f + 1) / (float2)(width, height);
	float2 pos_f_m_1 = (float2)(pos.x + 0.5f - 1, pos.y + 0.5f - 1) / (float2)(width, height);
	
	float dx = read_imagef(input, scalingSampler, (float2)(pos_f_p_1.x, pos_f.y)).x -
		read_imagef(input, scalingSampler, (float2)(pos_f_m_1.x, pos_f.y)).x;
	float dy = read_imagef(input, scalingSampler, (float2)(pos_f.x, pos_f_p_1.y)).x -
		read_imagef(input, scalingSampler, (float2)(pos_f.x, pos_f_m_1.y)).x;

	float magn = FLOAT_CORRECTNESS * sqrt(dx*dx + dy*dy);
	float orien = dx == 0.f ? 0.f : atan(dy / dx);

	//printf("dx=%f, dy=%f, magn=%f, orien=%f\n", dx, dy, magn, orien);

	write_imagef(magnitude, pos, (float4)(magn, magn, magn, 1.f));
	write_imagef(orientation, pos, (float4)(orien, orien, orien, 1.f));
}

#define NUM_BINS 36
__kernel void generate_feature_points(
	read_only image2d_t keypoints,
	read_only image2d_t magnitude,
	read_only image2d_t orientation,
	write_only image2d_t output,
	global write_only struct KeyPoint* kps,
	const unsigned int keypoints_count,
	const unsigned int kernel_size,
	const unsigned int scale,
	const unsigned int width,
	const unsigned int height,
	global unsigned int* count)
{
	int2 pos = { get_global_id(0), get_global_id(1) };

	float pixel = read_imagef(keypoints, srcSampler, pos).x;

	if (pixel == 0.f)
	{
		return;
	}

	int HALF_KERNEL_SIZE = kernel_size >> 1;
	float histogram[NUM_BINS];

	for (int r = -HALF_KERNEL_SIZE; r <= HALF_KERNEL_SIZE; ++r)
	{
		for (int c = -HALF_KERNEL_SIZE; c <= HALF_KERNEL_SIZE; ++c)
		{
			int2 coords = (int2)(pos.x + r, pos.y + c);
			float value = read_imagef(orientation, srcSampler, coords).x + M_PI;

			// convert to degrees
			unsigned orientationDegrees = value * 180 * M_1_PI;

			histogram[orientationDegrees / (360 / NUM_BINS)] += read_imagef(magnitude, srcSampler, coords).x;
		}
	}

	double max_peak = histogram[0];
	unsigned int max_peak_index = 0;

	for (int i = 1; i < NUM_BINS; ++i)
	{
		if (histogram[i] > max_peak)
		{
			max_peak = histogram[i];
			max_peak_index = i;
		}
	}

	//printf("Thread: (%d,%d)\n", pos.x, pos.y);

	int index = atomic_inc(&count[0]);
	if (index < keypoints_count)
	{
		kps[index].x_interp = (float)pos.x / (float)width;
		kps[index].y_interp = (float)pos.y / (float)height;
		kps[index].x = pos.x;
		kps[index].y = pos.y;
		kps[index].scale = scale;
		kps[index].magnitude = histogram[max_peak_index];
		kps[index].orientation = max_peak_index;
		//printf("[count=%d]: hist=%f, thresh=%f\n", index, max_peak, max_peak * 0.8f);

		float thresh = 0.8f * max_peak;
		for (int i = 0; i < max_peak_index; ++i)
		{
			if (histogram[i] > thresh)
			{
				index = atomic_inc(&count[0]);

				kps[index].x_interp = (float)pos.x / (float)width;
				kps[index].y_interp = (float)pos.y / (float)height;
				kps[index].x = pos.x;
				kps[index].y = pos.y;
				kps[index].scale = scale;
				kps[index].magnitude = histogram[i];
				kps[index].orientation = i;
				//printf("[count=%d]: hist=%f, thresh=%f\n", index, max_peak, max_peak * 0.8f);
			}
		}

		for (int i = max_peak_index + 1; i < NUM_BINS; ++i)
		{
			if (histogram[i] > thresh)
			{
				index = atomic_inc(&count[0]);

				kps[index].x_interp = (float)pos.x / (float)width;
				kps[index].y_interp = (float)pos.y / (float)height;
				kps[index].x = pos.x;
				kps[index].y = pos.y;
				kps[index].scale = scale;
				kps[index].magnitude = histogram[i];
				kps[index].orientation = i;
				//printf("[count=%d]: hist=%f, thresh=%f\n", index, max_peak, max_peak * 0.8f);
			}
		}
	}

	barrier(CLK_GLOBAL_MEM_FENCE);
	//printf("count=%d, kps_size=%d\n", count[0], keypoints_count);
}

#define DESC_NUM_BINS 8
__kernel void extract_feature_points(
	read_only image2d_t magnitude,
	read_only image2d_t orientation,
	global read_only float* weights,
	global write_only struct KeyPoint* kps,
	global read_write double* feature_vector,
	const unsigned int width,
	const unsigned int height)
{
	const int FV_SIZE = 128;
	const int WINDOW_SIZE = 16;
	const int SMALL_WINDOW_SIZE = 4;
	const int HALF_WINDOW_SIZE = WINDOW_SIZE >> 1;
	const float M_2X_PI = 2 * M_PI;
	const float FV_THRESHOLD = 0.2f;

	int pos = get_global_id(0);
	int fv_start_index = FV_SIZE * pos;
	struct KeyPoint kp = kps[pos];
	int x = kp.x;
	int y = kp.y;

	double histogram[DESC_NUM_BINS];

	uint w_index = 0;
	uint fv_index = 0;

	//printf("[%d]: %d, %d, magn=%f, ori=%f, start=%d, end=%d\n", pos, x, y, kp.magnitude, kp.orientation, x - HALF_WINDOW_SIZE, x + HALF_WINDOW_SIZE);
	for (int wx = x - HALF_WINDOW_SIZE; wx < x + HALF_WINDOW_SIZE; wx += SMALL_WINDOW_SIZE)
	{

		for (int wy = y - HALF_WINDOW_SIZE; wy < y + HALF_WINDOW_SIZE; wy += SMALL_WINDOW_SIZE)
		{
			for (int swx = wx; swx < wx + SMALL_WINDOW_SIZE; ++swx)
			{
				for (int swy = wy; swy < wy + SMALL_WINDOW_SIZE; ++swy)
				{
					if (swx < 0 || swx > width || swy < 0 || swy > height)
					{
						++w_index;
						continue;
					}

					// rotation invariance
					float value = read_imagef(orientation, srcSampler, (int2)(swx, swy)).x;
					value -= kp.orientation;

					while (value < 0)		{ value += M_2X_PI; }
					while (value > M_2X_PI) { value -= M_2X_PI; }

					// convert to degrees
					unsigned orientationDegrees = value * 180 * M_1_PI;

					histogram[orientationDegrees / (360 / DESC_NUM_BINS)] += read_imagef(magnitude, srcSampler, (int2)(swx, swy)).x * weights[w_index];

					++w_index;
				}
			}

			//printf("[%d]: %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n",
			//	pos,
			//	histogram[0],
			//	histogram[1],
			//	histogram[2],
			//	histogram[3],
			//	histogram[4],
			//	histogram[5],
			//	histogram[6],
			//	histogram[7]);

			for (int i = 0; i < DESC_NUM_BINS; ++i)
			{
				feature_vector[fv_start_index + fv_index] = histogram[i];

				histogram[i] = 0.0;

				++fv_index;
			}
		}
	}

	// NORMALIZE, THRESHOLD AND NORMALIZE AGAIN
	double norm = 0.f;
	for (int i = 0; i < FV_SIZE; ++i)
	{
		norm += pow(feature_vector[fv_start_index + i], 2);
		//printf("[%d]: %lf, norm=%lf\n", fv_start_index + i, feature_vector[fv_start_index + i], norm);
	}

	norm = rsqrt(norm);

	float norm_s = 0.f;
	for (int i = 0; i < FV_SIZE; ++i)
	{
		feature_vector[fv_start_index + i] *= norm;
		//printf("%f ", feature_vector[fv_start_index + i]);

		if (feature_vector[fv_start_index + i] > FV_THRESHOLD)
		{
			feature_vector[fv_start_index + i] = FV_THRESHOLD;
		}

		norm_s += pow(feature_vector[fv_start_index + i], 2);
	}

	//printf("[%d]: norm=%f, rsqsrt(norm)=%f\n", pos, norm_s, rsqrt(norm_s));
	norm_s = rsqrt(norm_s);


	for (int i = 0; i < FV_SIZE; ++i)
	{
		feature_vector[fv_start_index + i] *= norm;
	}
}