
#include "Octave.h"
#include "CLManager.h"

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

		// Set first image
		cl::Kernel & kernel = *CLManager::GetInstance()->GetKernel(Constants::KERNEL_RESIZE);

		res = kernel.setArg(0, *octave.GetDefaultImage());
		res = kernel.setArg(1, *m_default_image);
		res = kernel.setArg(2, w);
		res = kernel.setArg(3, h);

		res = m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, m_range, cl::NullRange);

		m_queue.finish();

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
		cl_int res;
		cl::Kernel & kernel = *CLManager::GetInstance()->GetKernel(Constants::KERNEL_CONVOLUTE);
		cl::Buffer convCL = cl::Buffer(m_context, CL_MEM_READ_ONLY, (1 + BLUR_KERNEL_SIZE * BLUR_KERNEL_SIZE) * sizeof(float), 0, 0);

		float sigma = m_starting_sigma;
		for (size_t i = 0; i < m_images.size(); ++i)
		{
			std::vector<float> gaussian = GaussianKernel(BLUR_KERNEL_SIZE, sigma);
			m_queue.enqueueWriteBuffer(convCL, CL_TRUE, 0, gaussian.size() * sizeof(float), &gaussian[0], 0, NULL);

			// Set arguments to kernel
			res = kernel.setArg(0, *m_default_image);
			res = kernel.setArg(1, *m_images[i]);
			res = kernel.setArg(2, convCL);

			res = m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, m_range, cl::NullRange);
			m_queue.finish();

			sigma *= SIGMA_INCREMENT;
		}

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

			res = m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, m_range, cl::NullRange);
			m_queue.finish();
		}

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

			res = m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, m_range, cl::NullRange);
			m_queue.finish();
		}
	}
	catch (const cl::Error & err)
	{
		std::cerr << "Octave::ComputeLocalMaxima: " << err.what() << " with error: " << err.err() << std::endl;
	}
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
	// compute values
	for (int row = -half_kernel_size; row <= half_kernel_size; ++row)
	{
		for (int col = -half_kernel_size; col <= half_kernel_size; ++col)
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
