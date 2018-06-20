
#include "Octave.h"
#include "CLManager.h"
#include "Structs.h"

#include <iostream>

#define M_PI 3.14159265359
float Octave::SIGMA_INCREMENT = sqrtf(2.f);

Octave::Octave()
{
	std::cerr << "You should be not calling this\n";
}

Octave::Octave(cl::Image2D * image, float sigma, uint32_t w, uint32_t h, uint32_t size)
	: m_width(w), m_height(h), m_starting_sigma(sigma)
{
	m_context = *CLManager::GetInstance()->GetContext();
	m_queue = *CLManager::GetInstance()->GetQueue();
	m_range = cl::NDRange(w, h);

	m_default_image = image;
	

	cl::ImageFormat imf = cl::ImageFormat(CL_RGBA, CL_FLOAT);

	m_images.resize(size);
	m_DoGs.resize(size - 1);
	m_points.resize(size - 3);
	m_magnitudes.resize(size - 3);
	m_orientations.resize(size - 3);

	m_images[0] = image;

	try
	{
		for (uint32_t i = 1; i < m_images.size(); ++i)
		{
			m_images[i] = new cl::Image2D(m_context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, imf, w, h);
		}

		for (uint32_t i = 0; i < m_DoGs.size(); ++i)
		{
			m_DoGs[i] = new cl::Image2D(m_context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, imf, w, h);
		}

		for (uint32_t i = 0; i < m_points.size(); ++i)
		{
			m_points[i] = new cl::Image2D(m_context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, imf, w, h);
		}

		for (uint32_t i = 0; i < m_magnitudes.size(); ++i)
		{
			m_magnitudes[i] = new cl::Image2D(m_context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, imf, w, h);
		}

		for (uint32_t i = 0; i < m_orientations.size(); ++i)
		{
			m_orientations[i] = new cl::Image2D(m_context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, imf, w, h);
		}
	}
	catch (const cl::Error & err)
	{
		std::cerr << "Octave::ctr1: " << err.what() << " with error: " << err.err() << std::endl;
	}

	Blur();
}

Octave::Octave(Octave & octave, uint32_t w, uint32_t h)
	: m_width(w), m_height(h)
{
	m_context = *CLManager::GetInstance()->GetContext();
	m_queue = *CLManager::GetInstance()->GetQueue();
	m_range = cl::NDRange(w, h);
	m_starting_sigma = octave.GetMiddleSigma();
	m_default_image = nullptr;

	cl::ImageFormat imf = cl::ImageFormat(CL_RGBA, CL_FLOAT);

	try
	{
		cl_int res;

		size_t size = octave.GetScaleSpaceSize();
		m_images.resize(size);
		m_DoGs.resize(size - 1);
		m_points.resize(size - 3);
		m_magnitudes.resize(size - 3);
		m_orientations.resize(size - 3);

		m_default_image = new cl::Image2D(m_context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, imf, w, h);

		for (uint32_t i = 0; i < m_images.size(); ++i)
		{
			m_images[i] = new cl::Image2D(m_context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, imf, w, h);
		}

		for (uint32_t i = 0; i < m_DoGs.size(); ++i)
		{
			m_DoGs[i] = new cl::Image2D(m_context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, imf, w, h);
		}

		for (uint32_t i = 0; i < m_points.size(); ++i)
		{
			m_points[i] = new cl::Image2D(m_context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, imf, w, h);
		}

		for (uint32_t i = 0; i < m_magnitudes.size(); ++i)
		{
			m_magnitudes[i] = new cl::Image2D(m_context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, imf, w, h);
		}

		for (uint32_t i = 0; i < m_orientations.size(); ++i)
		{
			m_orientations[i] = new cl::Image2D(m_context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, imf, w, h);
		}

		// Set first image
		cl::Kernel & kernel = *CLManager::GetInstance()->GetKernel(Constants::KERNEL_RESIZE);

		res = kernel.setArg(0, *octave.GetDefaultImage());
		res = kernel.setArg(1, *m_default_image);
		res = kernel.setArg(2, w);
		res = kernel.setArg(3, h);

		cl::Event ev;
		res = m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, m_range, cl::NullRange, nullptr, &ev);

		m_queue.finish();

		//std::cout << "Octave resize: " << (ev.getProfilingInfo<CL_PROFILING_COMMAND_END>() - ev.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / 1000000.f << std::endl;


		Blur();
	}
	catch (const cl::Error & err)
	{
		std::cerr << "Octave::ctr2: " << err.what() << " with error: " << err.err() << std::endl;
	}
}

Octave::~Octave()
{
	// TODO: it deletes the original image, do stuff to fix
	//for (size_t i = 0; i < m_images.size(); ++i)
	//{
	//	if (nullptr != m_images[i])
	//	{
	//		delete m_images[i];
	//		m_images[i] = nullptr;
	//	}
	//}

	// TODO: no fucking idea whats going on
	//if (nullptr != m_default_image)
	//{
	//	delete m_default_image;
	//	m_default_image = nullptr;
	//}
}

size_t Octave::GetScaleSpaceSize()
{
	return m_images.size();
}

float Octave::GetMiddleSigma() const
{
	return m_starting_sigma * std::pow(SIGMA_INCREMENT, m_images.size() / 2);
}

cl::Image2D * Octave::GetDefaultImage()
{
	return m_default_image;
}

cl::Image2D * Octave::GetImage(uint32_t idx)
{
	return m_images[idx];
	//return m_DoGs[0];
}

cl::Image2D * Octave::GetLastImage()
{
	return m_images.back();
}

std::vector<cl::Image2D*>& Octave::GetImages()
{
	return m_images;
}

std::vector<cl::Image2D*>& Octave::GetDoGs()
{
	return m_DoGs;
}

std::vector<cl::Image2D*>& Octave::GetFeatures()
{
	return m_points;
}

std::vector<cl::Image2D*>& Octave::GetMagnitudes()
{
	return m_magnitudes;
}

std::vector<cl::Image2D*>& Octave::GetOrientations()
{
	return m_orientations;
}

uint32_t Octave::GetWidth() const
{
	return m_width;
}

uint32_t Octave::GetHeight() const
{
	return m_height;
}

void Octave::Blur()
{
	try
	{
		cl::Event ev;
		unsigned long ret = 0;

		cl_int res;
		cl::Kernel & kernel = *CLManager::GetInstance()->GetKernel(Constants::KERNEL_CONVOLUTE);
		cl::Buffer convCL = cl::Buffer(m_context, CL_MEM_READ_ONLY, (1 + BLUR_KERNEL_SIZE * BLUR_KERNEL_SIZE) * sizeof(float), 0, 0);

		float sigma = m_starting_sigma;
		for (size_t i = 0; i < m_images.size(); ++i)
		{
			std::vector<float> gaussian = GaussianKernel(BLUR_KERNEL_SIZE, sigma);
			m_queue.enqueueWriteBuffer(convCL, CL_TRUE, 0, gaussian.size() * sizeof(float), &gaussian[0], 0, NULL);

			// Set arguments to kernel
			res = kernel.setArg(0, i == 0 ? *m_default_image : *m_images[i-1]);
			res = kernel.setArg(1, *m_images[i]);
			res = kernel.setArg(2, convCL);

			res = m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, m_range, cl::NullRange, nullptr, &ev);

			m_queue.finish();
			ret += ev.getProfilingInfo<CL_PROFILING_COMMAND_END>() - ev.getProfilingInfo<CL_PROFILING_COMMAND_START>();

			sigma *= SIGMA_INCREMENT;
		}

		//std::cout << "Octave blur=" << ret / 1000000.f << std::endl;

	}
	catch (const cl::Error & err)
	{
		std::cerr << "Octave::Blur: " << err.what() << " with error: " << err.err() << std::endl;
	}
}

void Octave::DoG()
{
	try
	{
		cl::Event ev;
		unsigned long ret = 0;

		cl::ImageFormat imf = cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8);

		cl_int res;
		cl::Kernel & kernel = *CLManager::GetInstance()->GetKernel(Constants::KERNEL_IMAGE_DIFFERENCE);

		uint32_t crt_dog = 0;
		for (size_t i = 1; i < m_images.size(); ++i)
		{
			// Set arguments to kernel
			res = kernel.setArg(0, *m_images[i-1]);
			res = kernel.setArg(1, *m_images[i]);
			res = kernel.setArg(2, *m_DoGs[crt_dog++]);

			res = m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, m_range, cl::NullRange, nullptr, &ev);
			m_queue.finish();

			m_queue.finish();
			ret += ev.getProfilingInfo<CL_PROFILING_COMMAND_END>() - ev.getProfilingInfo<CL_PROFILING_COMMAND_START>();

		}

		//std::cout << "Octave DoG=" << ret / 1000000.f << std::endl;
	}
	catch (const cl::Error & err)
	{
		std::cerr << "Octave::DoG: " << err.what() << " with error: " << err.err() << std::endl;
	}
}

void Octave::ComputeLocalMaxima()
{
	try
	{
		cl::Event ev;
		unsigned long ret = 0;

		cl::ImageFormat imf = cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8);

		cl_int res;
		cl::Kernel & kernel = *CLManager::GetInstance()->GetKernel(Constants::KERNEL_FIND_EXTREME_POINTS);

		uint32_t crt_feature_i = 0;
		for (size_t i = 1; i < m_DoGs.size() - 1; ++i)
		{
			// Set arguments to kernel
			res = kernel.setArg(0, *m_DoGs[i]);
			res = kernel.setArg(1, *m_DoGs[i - 1]);
			res = kernel.setArg(2, *m_DoGs[i + 1]);
			res = kernel.setArg(3, *m_points[crt_feature_i++]);

			res = m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, m_range, cl::NullRange, nullptr, &ev);

			m_queue.finish();
			ret += ev.getProfilingInfo<CL_PROFILING_COMMAND_END>() - ev.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		}

		//std::cout << "Local maxima=" << ret / 1000000.f << std::endl;
	}
	catch (const cl::Error & err)
	{
		std::cerr << "Octave::ComputeLocalMaxima: " << err.what() << " with error: " << err.err() << std::endl;
	}
}

std::vector<FeaturePoint> Octave::ComputeOrientation()
{
	std::vector<FeaturePoint> fvps;
	try
	{
		cl::Event ev;
		unsigned long ret = 0;

		unsigned kps_size = m_width * m_height;

		cl::ImageFormat imf = cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8);

		cl_int res;
		cl::Kernel & kernel = *CLManager::GetInstance()->GetKernel(Constants::KERNEL_MAGN_AND_ORIEN);
		cl::Kernel & kernel_magn_ori_interp = *CLManager::GetInstance()->GetKernel(Constants::KERNEL_MAGN_AND_ORIEN_INTERP);
		cl::Kernel & kernel_blur = *CLManager::GetInstance()->GetKernel(Constants::KERNEL_CONVOLUTE);
		cl::Kernel & kernel_gen_feature_points = *CLManager::GetInstance()->GetKernel(Constants::KERNEL_GENERATE_FEATURE_POINTS);
		cl::Kernel & kernel_extract_feature_points = *CLManager::GetInstance()->GetKernel(Constants::KERNEL_EXTRACT_FEATURE_POINTS);
		cl::Kernel & kernel_draw_feature_points = *CLManager::GetInstance()->GetKernel(Constants::KERNEL_DRAW_FEATURE_POINTS);

		cl::Buffer convCL = cl::Buffer(m_context, CL_MEM_READ_ONLY, (1 + BLUR_KERNEL_SIZE * BLUR_KERNEL_SIZE) * sizeof(float), 0, 0);
		cl::Buffer keypointsCL = cl::Buffer(m_context, CL_MEM_READ_WRITE, kps_size * sizeof(KeyPoint), 0, 0);
		cl::Buffer countCL = cl::Buffer(m_context, CL_MEM_READ_WRITE, sizeof(unsigned), 0, 0);
		cl::Buffer weightsCL = cl::Buffer(m_context, CL_MEM_READ_WRITE, WEIGHT_KERNEL_SIZE * WEIGHT_KERNEL_SIZE * sizeof(float), 0, 0);

		std::vector<float> weights = GaussianKernel(WEIGHT_KERNEL_SIZE, WEIGHT_KERNEL_SIZE >> 1);
		m_queue.enqueueWriteBuffer(weightsCL, CL_TRUE, 0, (weights.size() - 1) * sizeof(float), &weights[1], 0, NULL);

		uint32_t crt_feature_i = 0;
		for (size_t i = 1; i < m_DoGs.size() - 1; ++i)
		{
			// kps
			unsigned fv_count = 0;
			std::vector<KeyPoint> keypoints(kps_size);
			//m_queue.enqueueWriteBuffer(keypointsCL, CL_TRUE, 0, keypoints.size() * sizeof(KeyPoint), &keypoints[0], 0, NULL);
			m_queue.enqueueWriteBuffer(countCL, CL_TRUE, 0, sizeof(unsigned), &fv_count, 0, NULL);

			// Set arguments to kernel
			res = kernel.setArg(0, *m_DoGs[i]);
			res = kernel.setArg(1, *m_magnitudes[i - 1]);
			res = kernel.setArg(2, *m_orientations[i - 1]);

			res = m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, m_range, cl::NullRange, nullptr, &ev);
			m_queue.finish();
			ret += ev.getProfilingInfo<CL_PROFILING_COMMAND_END>() - ev.getProfilingInfo<CL_PROFILING_COMMAND_START>();

			// blur magnitudes
			float sigma = m_starting_sigma * std::pow(SIGMA_INCREMENT, i);

			// does not work for some reason
			//cl::Image2D *blurred_magn = new cl::Image2D(m_context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, imf, m_width, m_height);
			//
			//std::vector<float> gaussian = GaussianKernel(1, 1.5*sigma);
			//m_queue.enqueueWriteBuffer(convCL, CL_TRUE, 0, gaussian.size() * sizeof(float), &gaussian[0], 0, NULL);
			//
			//res = kernel_blur.setArg(0, *m_magnitudes[i - 1]);
			//res = kernel_blur.setArg(1, *blurred_magn);
			//res = kernel_blur.setArg(2, convCL);
			//
			//res = m_queue.enqueueNDRangeKernel(kernel_blur, cl::NullRange, m_range, cl::NullRange);
			//m_queue.finish();

			//delete m_magnitudes[i - 1];
			//m_magnitudes[i - 1] = blurred_magn;

			// compute keypoints
			res = kernel_gen_feature_points.setArg(0, *m_points[i-1]);
			res = kernel_gen_feature_points.setArg(1, *m_magnitudes[i - 1]);
			res = kernel_gen_feature_points.setArg(2, *m_orientations[i - 1]);
			res = kernel_gen_feature_points.setArg(3, *m_default_image); // TODO: change this
			res = kernel_gen_feature_points.setArg(4, keypointsCL);
			res = kernel_gen_feature_points.setArg(5, kps_size);
			res = kernel_gen_feature_points.setArg(6, GetKernelSize(1.5f *sigma));
			res = kernel_gen_feature_points.setArg(7, static_cast<cl_uint>(i));
			res = kernel_gen_feature_points.setArg(8, m_width);
			res = kernel_gen_feature_points.setArg(9, m_height);
			res = kernel_gen_feature_points.setArg(10, countCL);

			res = m_queue.enqueueNDRangeKernel(kernel_gen_feature_points, cl::NullRange, m_range, cl::NullRange, nullptr, &ev);
			m_queue.finish();
			ret += ev.getProfilingInfo<CL_PROFILING_COMMAND_END>() - ev.getProfilingInfo<CL_PROFILING_COMMAND_START>();

			// compute inerp magnitude and orientation
			res = kernel_magn_ori_interp.setArg(0, *m_DoGs[i]);
			res = kernel_magn_ori_interp.setArg(1, *m_magnitudes[i - 1]);
			res = kernel_magn_ori_interp.setArg(2, *m_orientations[i - 1]);
			res = kernel_magn_ori_interp.setArg(3, m_width);
			res = kernel_magn_ori_interp.setArg(4, m_height);

			res = m_queue.enqueueNDRangeKernel(kernel_magn_ori_interp, cl::NullRange, m_range, cl::NullRange, nullptr, &ev);
			m_queue.finish();
			ret += ev.getProfilingInfo<CL_PROFILING_COMMAND_END>() - ev.getProfilingInfo<CL_PROFILING_COMMAND_START>();

			m_queue.enqueueReadBuffer(countCL, CL_TRUE, 0, sizeof(unsigned), &fv_count, 0, NULL);

			//std::cout << "Count=" << fv_count << std::endl;
			if (0 == fv_count)
			{
				continue;
			}

			std::vector<FeaturePoint> fv(fv_count);
			cl::Buffer fvCL = cl::Buffer(m_context, CL_MEM_READ_WRITE, fv_count * sizeof(FeaturePoint), 0, 0);

			res = kernel_extract_feature_points.setArg(0, *m_magnitudes[i - 1]);
			res = kernel_extract_feature_points.setArg(1, *m_orientations[i - 1]);
			res = kernel_extract_feature_points.setArg(2, weightsCL);
			res = kernel_extract_feature_points.setArg(3, keypointsCL);
			res = kernel_extract_feature_points.setArg(4, fvCL);
			res = kernel_extract_feature_points.setArg(5, m_width);
			res = kernel_extract_feature_points.setArg(6, m_height);

			res = m_queue.enqueueNDRangeKernel(kernel_extract_feature_points, cl::NullRange, cl::NDRange(fv_count, 1), cl::NullRange, nullptr, &ev);
			m_queue.finish();
			ret += ev.getProfilingInfo<CL_PROFILING_COMMAND_END>() - ev.getProfilingInfo<CL_PROFILING_COMMAND_START>();
			
			res = kernel_draw_feature_points.setArg(0, *m_default_image);
			res = kernel_draw_feature_points.setArg(1, fvCL);
			res = kernel_draw_feature_points.setArg(2, m_width);
			res = kernel_draw_feature_points.setArg(3, m_height);

			res = m_queue.enqueueNDRangeKernel(kernel_draw_feature_points, cl::NullRange, cl::NDRange(fv_count, 1), cl::NullRange);
			m_queue.finish();

			m_queue.enqueueReadBuffer(fvCL, CL_TRUE, 0, fv_count * sizeof(FeaturePoint), &fv[0], 0, NULL);
			m_queue.finish();

			fvps.insert(fvps.end(), fv.begin(), fv.end());
		}

		//std::cout << "Orientation=" << ret / 1000000.f << std::endl;
	}
	catch (const cl::Error & err)
	{
		std::cerr << "Octave::ComputeLocalMaxima: " << err.what() << " with error: " << err.err() << std::endl;
	}

	return fvps;
}

float Octave::Gaussian(const int x, const int y, const float sigma)
{
	float r = sqrtf(x*x + y*y);
	float s = 2.f * sigma * sigma;

	return expf(-(r*r) / s) / (M_PI * s);
}

std::vector<float> Octave::GaussianKernel(const uint32_t kernel_size, const float sigma)
{
	std::vector<float> kernel(kernel_size * kernel_size + 1);
	kernel[0] = kernel_size;

	size_t idx = 1;
	float sum = 0;
	int half_kernel_size = kernel_size * 0.5;
	int start, end;

	if (kernel_size % 2 == 0)
	{
		start = -(kernel_size >> 1);
		end = -start - 1;
	}
	else
	{
		end = kernel_size >> 1;
		start = -end;
	}

	// compute values
	for (int row = start; row <= end; ++row)
	{
		for (int col = start; col <= end; ++col)
		{
			double x = Gaussian(row, col, sigma);
			kernel[idx++] = x;
			sum += x;
		}
	}

	// normalize
	idx = 1;
	for (int row = 0; row < kernel_size; row++)
	{
		for (int col = 0; col < kernel_size; col++)
		{
			kernel[idx++] /= sum;
		}
	}

	return kernel;
}

unsigned int Octave::GetKernelSize(float sigma, float cut_off)
{
	const unsigned MAX_KERNEL_SIZE = 20;
	size_t i;
	for (i = 0; i < MAX_KERNEL_SIZE; i++)
	{
		if (exp(-((double)(i*i)) / (2.0*sigma*sigma)) < cut_off)
		{
			break;
		}
	}

	return 2 * i - 1;
}
