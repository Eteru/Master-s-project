
#include "Octave.h"
#include "CLManager.h"

#include <iostream>

Octave::Octave()
{
	std::cerr << "You should be not calling this\n";
}

Octave::Octave(cl::Image2D * image, uint32_t w, uint32_t h, uint32_t size)
	: m_width(w), m_height(h)
{
	m_context = *CLManager::GetInstance()->GetContext();
	m_queue = *CLManager::GetInstance()->GetQueue();
	m_range = cl::NDRange(w, h);

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

	cl::ImageFormat imf = cl::ImageFormat(CL_RGBA, CL_FLOAT);

	try
	{
		cl_int res;

		size_t size = octave.GetScaleSpaceSize();
		m_images.resize(size);
		m_DoGs.resize(size - 1);
		m_points.resize(size - 3);

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

		res = kernel.setArg(0, *octave.GetLastImage());
		res = kernel.setArg(1, *m_images[0]);
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
}

size_t Octave::GetScaleSpaceSize()
{
	return m_images.size();
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
	// sigma = 1.6, k = sqrt 2
	//std::vector<std::vector<float>> gaussian_kernels =
	//{
	//	{ // sigma = 1.6
	//		7.f,
	//		0.002121f, 0.005461f, 0.009629f, 0.011633f, 0.009629f, 0.005461f, 0.002121f,
	//		0.005461f, 0.014059f, 0.024791f, 0.029949f, 0.024791f, 0.014059f, 0.005461f,
	//		0.009629f, 0.024791f, 0.043715f, 0.052812f, 0.043715f, 0.024791f, 0.009629f,
	//		0.011633f, 0.029949f, 0.052812f, 0.063802f, 0.052812f, 0.029949f, 0.011633f,
	//		0.009629f, 0.024791f, 0.043715f, 0.052812f, 0.043715f, 0.024791f, 0.009629f,
	//		0.005461f, 0.014059f, 0.024791f, 0.029949f, 0.024791f, 0.014059f, 0.005461f,
	//		0.002121f, 0.005461f, 0.009629f, 0.011633f, 0.009629f, 0.005461f, 0.002121f
	//
	//	},
	//	{ // sigma = 1.6 * sqrt 2
	//		7.f,
	//		0.007036f, 0.011376f, 0.015176f, 0.016706f, 0.015176f, 0.011376f, 0.007036f,
	//		0.011376f, 0.018391f, 0.024536f, 0.027010f, 0.024536f, 0.018391f, 0.011376f,
	//		0.015176f, 0.024536f, 0.032732f, 0.036033f, 0.032732f, 0.024536f, 0.015176f,
	//		0.016706f, 0.027010f, 0.036033f, 0.039667f, 0.036033f, 0.027010f, 0.016706f,
	//		0.015176f, 0.024536f, 0.032732f, 0.036033f, 0.032732f, 0.024536f, 0.015176f,
	//		0.011376f, 0.018391f, 0.024536f, 0.027010f, 0.024536f, 0.018391f, 0.011376f,
	//		0.007036f, 0.011376f, 0.015176f, 0.016706f, 0.015176f, 0.011376f, 0.007036f
	//	},
	//	{ // sigma = 1.6 * sqrt 2 * sqrt 2
	//		7.f,
	//		0.012235f, 0.015587f, 0.018024f, 0.018919f, 0.018024f, 0.015587f, 0.012235f,
	//		0.015587f, 0.019858f, 0.022963f, 0.024102f, 0.022963f, 0.019858f, 0.015587f,
	//		0.018024f, 0.022963f, 0.026554f, 0.027872f, 0.026554f, 0.022963f, 0.018024f,
	//		0.018919f, 0.024102f, 0.027872f, 0.029255f, 0.027872f, 0.024102f, 0.018919f,
	//		0.018024f, 0.022963f, 0.026554f, 0.027872f, 0.026554f, 0.022963f, 0.018024f,
	//		0.015587f, 0.019858f, 0.022963f, 0.024102f, 0.022963f, 0.019858f, 0.015587f,
	//		0.012235f, 0.015587f, 0.018024f, 0.018919f, 0.018024f, 0.015587f, 0.012235f
	//	},
	//	{ // sigma = 1.6 * sqrt 2 * sqrt 2 * sqrt 2
	//		7.f,
	//		0.015892f, 0.017946f, 0.019304f, 0.019779f, 0.019304f, 0.017946f, 0.015892f,
	//		0.017946f, 0.020266f, 0.021799f, 0.022336f, 0.021799f, 0.020266f, 0.017946f,
	//		0.019304f, 0.021799f, 0.023449f, 0.024026f, 0.023449f, 0.021799f, 0.019304f,
	//		0.019779f, 0.022336f, 0.024026f, 0.024617f, 0.024026f, 0.022336f, 0.019779f,
	//		0.019304f, 0.021799f, 0.023449f, 0.024026f, 0.023449f, 0.021799f, 0.019304f,
	//		0.017946f, 0.020266f, 0.021799f, 0.022336f, 0.021799f, 0.020266f, 0.017946f,
	//		0.015892f, 0.017946f, 0.019304f, 0.019779f, 0.019304f, 0.017946f, 0.015892f
	//	},
	//	{ // sigma = 1.6 * sqrt 2 * sqrt 2 * sqrt 2 * sqrt 2
	//		7.f,
	//		0.018035f, 0.019168f, 0.019882f, 0.020125f, 0.019882f, 0.019168f, 0.018035f,
	//		0.019168f, 0.020372f, 0.021130f, 0.021389f, 0.021130f, 0.020372f, 0.019168f,
	//		0.019882f, 0.021130f, 0.021917f, 0.022186f, 0.021917f, 0.021130f, 0.019882f,
	//		0.020125f, 0.021389f, 0.022186f, 0.022457f, 0.022186f, 0.021389f, 0.020125f,
	//		0.019882f, 0.021130f, 0.021917f, 0.022186f, 0.021917f, 0.021130f, 0.019882f,
	//		0.019168f, 0.020372f, 0.021130f, 0.021389f, 0.021130f, 0.020372f, 0.019168f,
	//		0.018035f, 0.019168f, 0.019882f, 0.020125f, 0.019882f, 0.019168f, 0.018035f
	//	}
	//};

	std::vector<float> gaussian = { // sigma = sqrt 2
		7.f,
		0.001044f, 0.003468f, 0.007121f, 0.009050f, 0.007121f, 0.003468f, 0.001044f,
		0.003468f, 0.011514f, 0.023644f, 0.030051f, 0.023644f, 0.011514f, 0.003468f,
		0.007121f, 0.023644f, 0.048555f, 0.061712f, 0.048555f, 0.023644f, 0.007121f,
		0.009050f, 0.030051f, 0.061712f, 0.078433f, 0.061712f, 0.030051f, 0.009050f,
		0.007121f, 0.023644f, 0.048555f, 0.061712f, 0.048555f, 0.023644f, 0.007121f,
		0.003468f, 0.011514f, 0.023644f, 0.030051f, 0.023644f, 0.011514f, 0.003468f,
		0.001044f, 0.003468f, 0.007121f, 0.009050f, 0.007121f, 0.003468f, 0.001044f
	};

	try
	{
		cl_int res;
		cl::Kernel & kernel = *CLManager::GetInstance()->GetKernel(Constants::KERNEL_CONVOLUTE);
		cl::Buffer convCL = cl::Buffer(m_context, CL_MEM_READ_ONLY, gaussian.size() * sizeof(float), 0, 0);
		m_queue.enqueueWriteBuffer(convCL, CL_TRUE, 0, gaussian.size() * sizeof(float), &gaussian[0], 0, NULL);

		for (size_t i = 1; i < m_images.size(); ++i)
		{

			// Set arguments to kernel
			res = kernel.setArg(0, *m_images[i - 1]);
			res = kernel.setArg(1, *m_images[i]);
			res = kernel.setArg(2, convCL);

			res = m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, m_range, cl::NullRange);
			m_queue.finish();
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
