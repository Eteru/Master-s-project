
#include "Octave.h"
#include "CLManager.h"

#include <iostream>

Octave::Octave(cl::Image2D * image, uint32_t w, uint32_t h, uint32_t size)
{
	m_context = *CLManager::GetInstance()->GetContext();
	m_queue = *CLManager::GetInstance()->GetQueue();
	m_range = cl::NDRange(w, h);


	cl::ImageFormat imf = cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8);

	m_images.push_back(image);

	try
	{
		for (uint32_t i = 1; i < size; ++i)
		{
			m_images.push_back(new cl::Image2D(m_context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, imf, w, h));
		}
	}
	catch (const cl::Error & err)
	{
		std::cerr << "Octave: " << err.what() << " with error: " << err.err() << std::endl;
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

void Octave::Blur()
{
	std::vector<std::vector<float>> gaussian_kernels =
	{
		{
			3.f,
			0.102059f, 0.115349f, 0.102059f,
			0.115349f, 0.130371f, 0.115349f,
			0.102059f, 0.115349f, 0.102059f
		},
		{
			3.f,
			0.106504f, 0.113341f, 0.106504f,
			0.113341f, 0.120617f, 0.113341f,
			0.106504f, 0.113341f, 0.106504f
		},
		{
			3.f,
			0.108785f, 0.112255f, 0.108785f,
			0.112255f, 0.115836f, 0.112255f,
			0.108785f, 0.112255f, 0.108785f
		},
		{
			3.f,
			0.109941f, 0.111691f, 0.109941f,
			0.111691f, 0.113469f, 0.111691f,
			0.109941f, 0.111691f, 0.109941f
		}
	};

	try
	{
		cl_int res;
		cl::Kernel & kernel = *CLManager::GetInstance()->GetKernel(Constants::KERNEL_CONVOLUTE);
		cl::Buffer convCL = cl::Buffer(m_context, CL_MEM_READ_ONLY, gaussian_kernels[0].size() * sizeof(float), 0, 0);

		for (size_t i = 1; i < m_images.size(); ++i)
		{
			m_queue.enqueueWriteBuffer(convCL, CL_TRUE, 0, gaussian_kernels[0].size() * sizeof(float), &gaussian_kernels[i - 1][0], 0, NULL);

			// Set arguments to kernel
			res = kernel.setArg(0, *m_images[0]);
			res = kernel.setArg(1, *m_images[i]);
			res = kernel.setArg(2, convCL);

			res = m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, m_range, cl::NullRange);
		}

		m_queue.finish();
	}
	catch (const cl::Error & err)
	{
		std::cerr << "Octave::Blur: " << err.what() << " with error: " << err.err() << std::endl;
	}
}

void Octave::DoG()
{
}
