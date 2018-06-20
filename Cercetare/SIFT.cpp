
#include "SIFT.h"
#include "CLManager.h"

#include <iostream>
#include <algorithm>
#include <iterator>
#include <QImage>
#include <vector>

uint32_t SIFT::m_current_img = 0;


SIFT::SIFT()
{
}

SIFT::~SIFT()
{
}

cl::Image2D * SIFT::Run(cl::Image2D * image, uint32_t w, uint32_t h)
{
	std::vector<Octave> m_octaves;

	m_fvps[image] = {};

	cl::Image2D * ref_img = SetupReferenceImage(image, w, h);

	uint32_t crt_w = 2 * w;
	uint32_t crt_h = 2 * h;

	float default_sigma = sqrtf(2.f) * 0.5f;

	m_octaves.push_back(Octave(ref_img, default_sigma, crt_w, crt_h, NUMBER_OF_BLURS));

	for (size_t i = 1; i < NUMBER_OF_OCTAVES; ++i)
	{
		crt_w *= 0.5;
		crt_h *= 0.5;
	
		m_octaves.push_back(Octave(m_octaves[i-1], crt_w, crt_h));
	}

	for(int i = 0; i < m_octaves.size(); ++i)
	{
		m_octaves[i].DoG();
		m_octaves[i].ComputeLocalMaxima();
		auto fvps = m_octaves[i].ComputeOrientation();
		m_fvps[image].insert(m_fvps[image].end(), fvps.begin(), fvps.end());
		WriteOctaveImagesOnDisk(m_octaves[i], i);
	}

	return m_octaves[1].GetImage(0);
}

std::vector<float> SIFT::FindImage(cl::Image2D * image, uint32_t w, uint32_t h, cl::Image2D * image_to_find, uint32_t w_m, uint32_t h_m)
{
	const double DISTANCE_THRESHOLD = 0.6;

	if (m_fvps.find(image) == m_fvps.end())
	{
		Run(image, w, h);
	}

	if (m_fvps.find(image_to_find) == m_fvps.end())
	{
		Run(image_to_find, w_m, h_m);
	}
	
	auto features = m_fvps[image];
	auto features_to_find = m_fvps[image_to_find];

	//std::vector<FeaturePoint> fp;
	// min_x, max_x, min_y, max_y
	std::vector<float> rect = { static_cast<float>(w), 0.f, static_cast<float>(h), 0.f };

	for (auto ftf : features_to_find)
	{
		for (auto feature : features)
		{
			double euc = 0.0;
			for (int i = 0; i < MAX_ORIS && euc < 1.0; ++i)
			{
				euc += (ftf.orientations[i] - feature.orientations[i]) * (ftf.orientations[i] - feature.orientations[i]);
			}

			//euc = sqrt(euc);

			//if (euc < 1.0)
			//	std::cout << "Distance: " << euc << std::endl;

			if (euc < DISTANCE_THRESHOLD * DISTANCE_THRESHOLD)
			{
				//fp.push_back(feature);

				rect[0] = std::min(rect[0], feature.x);
				rect[1] = std::max(rect[1], feature.x);

				rect[2] = std::min(rect[2], feature.y);
				rect[3] = std::max(rect[3], feature.y);
			}
		}
	}

	return rect;
}

cl::Image2D *  SIFT::SetupReferenceImage(cl::Image2D * image, uint32_t w, uint32_t h)
{
	cl::Image2D *output = nullptr;
	cl::ImageFormat imf = cl::ImageFormat(CL_RGBA, CL_FLOAT);

	std::vector<float> gaussian = { // sigma = 1.6
		7.f,
		0.002121f, 0.005461f, 0.009629f, 0.011633f, 0.009629f, 0.005461f, 0.002121f,
		0.005461f, 0.014059f, 0.024791f, 0.029949f, 0.024791f, 0.014059f, 0.005461f,
		0.009629f, 0.024791f, 0.043715f, 0.052812f, 0.043715f, 0.024791f, 0.009629f,
		0.011633f, 0.029949f, 0.052812f, 0.063802f, 0.052812f, 0.029949f, 0.011633f,
		0.009629f, 0.024791f, 0.043715f, 0.052812f, 0.043715f, 0.024791f, 0.009629f,
		0.005461f, 0.014059f, 0.024791f, 0.029949f, 0.024791f, 0.014059f, 0.005461f,
		0.002121f, 0.005461f, 0.009629f, 0.011633f, 0.009629f, 0.005461f, 0.002121f
	
	};

	try
	{
		cl_int res;

		cl::Context context = *CLManager::GetInstance()->GetContext();
		cl::CommandQueue queue = *CLManager::GetInstance()->GetQueue();

		// Initial blur with sigma = 1.6
		cl::Kernel kernel_blur = *CLManager::GetInstance()->GetKernel(Constants::KERNEL_CONVOLUTE);
		cl::Buffer convCL = cl::Buffer(context, CL_MEM_READ_ONLY, gaussian.size() * sizeof(float), 0, 0);
		cl::Image2D resized_img = cl::Image2D(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, imf, w, h);

		queue.enqueueWriteBuffer(convCL, CL_TRUE, 0, gaussian.size() * sizeof(float), &gaussian[0], 0, NULL);

		// Set arguments to kernel
		res = kernel_blur.setArg(0, *image);
		res = kernel_blur.setArg(1, resized_img);
		res = kernel_blur.setArg(2, convCL);

		res = queue.enqueueNDRangeKernel(kernel_blur, cl::NullRange, cl::NDRange(w, h), cl::NullRange);
		queue.finish();

		// Resize to double its size		
		output = new cl::Image2D(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, imf, 2 * w, 2 * h);
		cl::Kernel kernel_resize = *CLManager::GetInstance()->GetKernel(Constants::KERNEL_RESIZE);

		res = kernel_resize.setArg(0, resized_img);
		res = kernel_resize.setArg(1, *output);
		res = kernel_resize.setArg(2, w * 2);
		res = kernel_resize.setArg(3, h * 2);

		res = queue.enqueueNDRangeKernel(kernel_resize, cl::NullRange, cl::NDRange(2 * w, 2 * h), cl::NullRange);

		queue.finish();
	}
	catch (const cl::Error & err)
	{
		std::cerr << "SIFT::SetupReferenceImage: " << err.what() << " with error: " << err.err() << std::endl;
	}

	return output;
}

cl::Image2D *  SIFT::SetupReferenceImageOld(cl::Image2D * image, uint32_t w, uint32_t h)
{
	cl::Image2D *output = nullptr;
	cl::ImageFormat imf = cl::ImageFormat(CL_RGBA, CL_FLOAT);

	std::vector<float> gaussian_kernel =
	{
		5.f,
		0.000002f, 0.000212f, 0.000922f, 0.000212f, 0.000002f,
		0.000212f, 0.024745f, 0.107391f, 0.024745f, 0.000212f,
		0.000922f, 0.107391f, 0.466066f, 0.107391f, 0.000922f,
		0.000212f, 0.024745f, 0.107391f, 0.024745f, 0.000212f,
		0.000002f, 0.000212f, 0.000922f, 0.000212f, 0.000002f
	};

	try
	{
		cl_int res;

		cl::Context context = *CLManager::GetInstance()->GetContext();
		cl::CommandQueue queue = *CLManager::GetInstance()->GetQueue();

		// Initial blur
		cl::Kernel kernel_conv = *CLManager::GetInstance()->GetKernel(Constants::KERNEL_CONVOLUTE);
		cl::Buffer convCL = cl::Buffer(context, CL_MEM_READ_ONLY, gaussian_kernel.size() * sizeof(float), 0, 0);
		cl::Image2D *blurred_image = new cl::Image2D(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, imf, w, h);


		queue.enqueueWriteBuffer(convCL, CL_TRUE, 0, gaussian_kernel.size() * sizeof(float), &gaussian_kernel[0], 0, NULL);
		
		res = kernel_conv.setArg(0, *image);
		res = kernel_conv.setArg(1, *blurred_image);
		res = kernel_conv.setArg(2, convCL);

		res = queue.enqueueNDRangeKernel(kernel_conv, cl::NullRange, cl::NDRange(w, h), cl::NullRange);
		queue.finish();

		// Resize to double its size		
		output = new cl::Image2D(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, imf, 2 * w, 2 * h);
		cl::Kernel kernel_resize = *CLManager::GetInstance()->GetKernel(Constants::KERNEL_RESIZE);

		res = kernel_resize.setArg(0, *blurred_image);
		res = kernel_resize.setArg(1, *output);
		res = kernel_resize.setArg(2, w * 2);
		res = kernel_resize.setArg(3, h * 2);

		res = queue.enqueueNDRangeKernel(kernel_resize, cl::NullRange, cl::NDRange(2 * w, 2 * h), cl::NullRange);

		queue.finish();

		delete blurred_image;
	}
	catch (const cl::Error & err)
	{
		std::cerr << "SIFT::SetupReferenceImage: " << err.what() << " with error: " << err.err() << std::endl;
	}
	
	return output;
}

void SIFT::WriteOctaveImagesOnDisk(Octave & o, uint32_t o_idx) const
{
	auto imgs = o.GetImages();
	imgs.insert(imgs.end(), o.GetDoGs().begin(), o.GetDoGs().end());
	imgs.insert(imgs.end(), o.GetFeatures().begin(), o.GetFeatures().end());
	imgs.insert(imgs.end(), o.GetMagnitudes().begin(), o.GetMagnitudes().end());
	imgs.insert(imgs.end(), o.GetOrientations().begin(), o.GetOrientations().end());
	imgs.push_back(o.GetDefaultImage());

	cl::Context context = *CLManager::GetInstance()->GetContext();
	cl::CommandQueue queue = *CLManager::GetInstance()->GetQueue();

	cl::size_t<3> m_origin, m_region;
	m_origin[0] = 0; m_origin[1] = 0, m_origin[2] = 0;
	m_region[0] = o.GetWidth(); m_region[1] = o.GetHeight(); m_region[2] = 1;

	std::vector<float> valuesf(m_region[0] * m_region[1] * 4);
	std::vector<uchar> valuesui(m_region[0] * m_region[1] * 4);

	for (auto img : imgs)
	{
		QImage qimg = QImage(m_region[0], m_region[1], QImage::Format::Format_RGB32);

		cl_int res = queue.enqueueReadImage(*img, CL_TRUE, m_origin, m_region, 0, 0, &valuesf.front());

		queue.finish();

		//std::transform(valuesf.begin(), valuesf.end(), valuesui.begin(), [](float f) { return f * 255; });
		//auto mM = std::minmax_element(valuesf.begin(), valuesf.end());
		//std::cout << o_idx << ": min=" << *mM.first << "(" << std::distance(valuesf.begin(), mM.first) << ")" << ", max=" << *mM.second << "(" << std::distance(valuesf.begin(), mM.second) << ")" <<std::endl;
		for (size_t i = 0; i < valuesf.size(); ++i)
		{
			valuesui[i] = valuesf[i] < 0.f ? 0 : valuesf[i] * 255;
		}

		for (int i = 0, row = 0; row < m_region[1]; ++row, i += qimg.bytesPerLine())
		{
			memcpy(qimg.scanLine(row), &valuesui[i], qimg.bytesPerLine());
		}

		bool t = qimg.save(QString::fromStdString("D:\\workspace\\sift tests\\Octave"+ std::to_string(o_idx) + " - img" + std::to_string(m_current_img++)) + ".png", nullptr, 100);
	}
}

void SIFT::WriteImageOnDisk(cl::Image2D * img, uint32_t w, uint32_t h, std::string name) const
{
	cl::Context context = *CLManager::GetInstance()->GetContext();
	cl::CommandQueue queue = *CLManager::GetInstance()->GetQueue();

	cl::size_t<3> m_origin, m_region;
	m_origin[0] = 0; m_origin[1] = 0, m_origin[2] = 0;
	m_region[0] = w; m_region[1] = h; m_region[2] = 1;

	std::vector<float> valuesf(m_region[0] * m_region[1] * 4);
	std::vector<uchar> valuesui(m_region[0] * m_region[1] * 4);

	QImage qimg = QImage(m_region[0], m_region[1], QImage::Format::Format_RGB32);

	cl_int res = queue.enqueueReadImage(*img, CL_TRUE, m_origin, m_region, 0, 0, &valuesf.front());

	queue.finish();

	for (size_t i = 0; i < valuesf.size(); ++i)
	{
		valuesui[i] = valuesf[i] < 0.f ? 0 : valuesf[i] * 255;
	}

	for (int i = 0, row = 0; row < m_region[1]; ++row, i += qimg.bytesPerLine())
	{
		memcpy(qimg.scanLine(row), &valuesui[i], qimg.bytesPerLine());
	}

	bool t = qimg.save(QString::fromStdString("D:\\workspace\\sift tests\\" + name + ".png"), nullptr, 100);
}
