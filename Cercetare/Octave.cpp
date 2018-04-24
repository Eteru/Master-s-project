
#include "Octave.h"
#include "CLManager.h"

#include <iostream>

Octave::Octave(cl::Image2D * image, uint32_t w, uint32_t h, uint32_t size)
	: m_width(w), m_height(h)
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

cl::Image2D * Octave::GetImage(uint32_t idx)
{
	return m_images[idx];
}

void Octave::Blur()
{
	std::vector<std::vector<float>> gaussian_kernels =
	{
		{
			5.f,
			0.023528f, 0.033969f, 0.038393f, 0.033969f, 0.023528f,
			0.033969f, 0.049045f, 0.055432f, 0.049045f, 0.033969f,
			0.038393f, 0.055432f, 0.062651f, 0.055432f, 0.038393f,
			0.033969f, 0.049045f, 0.055432f, 0.049045f, 0.033969f,
			0.023528f, 0.033969f, 0.038393f, 0.033969f, 0.023528f
		},
		{
			5.f,
			0.033565f, 0.038140f, 0.039799f, 0.038140f, 0.033565f,
			0.038140f, 0.043337f, 0.045223f, 0.043337f, 0.038140f,
			0.039799f, 0.045223f, 0.047190f, 0.045223f, 0.039799f,
			0.038140f, 0.043337f, 0.045223f, 0.043337f, 0.038140f,
			0.033565f, 0.038140f, 0.039799f, 0.038140f, 0.033565f
		},
		{
			5.f,
			0.036676f, 0.039104f, 0.039949f, 0.039104f, 0.036676f,
			0.039104f, 0.041694f, 0.042594f, 0.041694f, 0.039104f,
			0.039949f, 0.042594f, 0.043514f, 0.042594f, 0.039949f,
			0.039104f, 0.041694f, 0.042594f, 0.041694f, 0.039104f,
			0.036676f, 0.039104f, 0.039949f, 0.039104f, 0.036676f
		},
		{
			5.f,
			0.037986f, 0.039473f, 0.039982f, 0.039473f, 0.037986f,
			0.039473f, 0.041019f, 0.041547f, 0.041019f, 0.039473f,
			0.039982f, 0.041547f, 0.042082f, 0.041547f, 0.039982f,
			0.039473f, 0.041019f, 0.041547f, 0.041019f, 0.039473f,
			0.037986f, 0.039473f, 0.039982f, 0.039473f, 0.037986f
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
	try
	{
		cl::ImageFormat imf = cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8);

		cl_int res;
		cl::Kernel & kernel = *CLManager::GetInstance()->GetKernel(Constants::KERNEL_IMAGE_DIFFERENCE);

		for (size_t i = 1; i < m_images.size(); ++i)
		{
			cl::Image2D *img = new cl::Image2D(m_context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, imf, m_width, m_height);
			// Set arguments to kernel
			res = kernel.setArg(0, *m_images[i-1]);
			res = kernel.setArg(1, *m_images[i]);
			res = kernel.setArg(2, *img);

			res = m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, m_range, cl::NullRange);
			m_queue.finish();

			delete m_images[i - 1];
			m_images[i - 1] = img;
		}

	}
	catch (const cl::Error & err)
	{
		std::cerr << "Octave::DoG: " << err.what() << " with error: " << err.err() << std::endl;
	}
}
