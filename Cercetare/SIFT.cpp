
#include "SIFT.h"
#include "CLManager.h"

#include <iostream>

SIFT::SIFT()
{
}

SIFT::~SIFT()
{
}

cl::Image2D * SIFT::Run(cl::Image2D * image, uint32_t w, uint32_t h)
{
	std::vector<Octave> m_octaves;
	cl::ImageFormat imf = cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8);

	uint32_t ratio = 2;
	uint32_t crt_w = w;
	uint32_t crt_h = h;

	// first octave has the default image
	m_octaves.push_back(Octave(image, w, h, NUMBER_OF_BLURS));

	try
	{
		cl_int res;

		cl::Context context = *CLManager::GetInstance()->GetContext();
		cl::CommandQueue queue = *CLManager::GetInstance()->GetQueue();
		cl::Kernel & kernel = *CLManager::GetInstance()->GetKernel(Constants::KERNEL_RESIZE);

		for (uint32_t i = 1; i < NUMBER_OF_OCTAVES; ++i)
		{
			crt_w = w / ratio;
			crt_h = h / ratio;

			cl::Image2D *octave_image = new cl::Image2D(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, imf, crt_w, crt_h);
			res = kernel.setArg(0, *image);
			res = kernel.setArg(1, *octave_image);
			res = kernel.setArg(2, ratio);

			res = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(crt_w, crt_h), cl::NullRange);

			queue.finish();

			m_octaves.push_back(Octave(octave_image, crt_w, crt_h, NUMBER_OF_BLURS));

			ratio *= 2;
		}
	}
	catch (const cl::Error & err)
	{
		std::cerr << "SIFT::Run: " << err.what() << " with error: " << err.err() << std::endl;
	}

	for (Octave & o : m_octaves)
	{
		o.Blur();
		o.DoG();
	}

	return m_octaves[0].GetImage(3);
}
