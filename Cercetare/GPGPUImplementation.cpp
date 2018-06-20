
#include "GPGPUImplementation.h"
#include "Constants.h"
#include "CLManager.h"

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <set>
#include <chrono>

GPGPUImplementation::GPGPUImplementation()
	: m_initialized(false), m_data_original(nullptr), m_data_front(nullptr), m_data_back(nullptr)
{
	std::time_t t;
	std::srand(static_cast <unsigned> (time(&t)));
	
	if (false == CLManager::GetInstance()->Init())
	{
		return;
	}

	m_contextCL = *CLManager::GetInstance()->GetContext();
	m_queue = *CLManager::GetInstance()->GetQueue();
	m_CLready = true;
}

GPGPUImplementation::~GPGPUImplementation()
{
	if (nullptr == m_data_original)
	{
		delete m_data_original;
		m_data_original = nullptr;
	}

	if (nullptr == m_data_front)
	{
		delete m_data_front;
		m_data_front = nullptr;
	}

	if (nullptr == m_data_back)
	{
		delete m_data_back;
		m_data_back = nullptr;
	}
}

void GPGPUImplementation::LoadData(QImage & img)
{
	cl_int res;

	m_width = img.width();
	m_height = img.height();

	m_origin[0] = 0; m_origin[1] = 0, m_origin[2] = 0;
	m_region[0] = m_width; m_region[1] = m_height; m_region[2] = 1;

	m_values.resize(m_width * m_height * 4);

	CopyImageToBuffer(img, m_values_orig);

	if (nullptr != m_data_original)
	{
		delete m_data_original;
		m_data_original = nullptr;
	}

	if (nullptr != m_data_front)
	{
		delete m_data_front;
		m_data_front = nullptr;
	}

	if (nullptr != m_data_back)
	{
		delete m_data_back;
		m_data_back = nullptr;
	}

	try
	{
		cl::ImageFormat imf = cl::ImageFormat(CL_RGBA, CL_UNORM_INT8);

		m_data_original = new cl::Image2D(m_contextCL, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, imf, m_width, m_height);
		m_data_front = new cl::Image2D(m_contextCL, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, imf, m_width, m_height);
		m_data_back = new cl::Image2D(m_contextCL, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, imf, m_width, m_height);

		m_globalRange = cl::NDRange(m_width, m_height);

		res = m_queue.enqueueWriteImage(*m_data_original, CL_TRUE, m_origin, m_region, 0, 0, img.bits());
		m_queue.finish();
	}
	catch (const cl::Error & err)
	{
		std::cerr << "LoadData: " << err.what() << " with error: " << err.err() << std::endl;
	}
}

void GPGPUImplementation::SetData(QImage & img)
{
	LoadData(img);

	m_initialized = true;
}

void GPGPUImplementation::CustomFilter(QImage & img, const std::vector<float>& kernel_values)
{
	cl_int res;

	try
	{
		cl::Kernel & kernel = *CLManager::GetInstance()->GetKernel(Constants::KERNEL_CONVOLUTE);
		cl::Buffer convCL = cl::Buffer(m_contextCL, CL_MEM_READ_ONLY, kernel_values.size() * sizeof(float), 0, 0);
		m_queue.enqueueWriteBuffer(convCL, CL_TRUE, 0, kernel_values.size() * sizeof(float), &kernel_values[0], 0, NULL);

		// Set arguments to kernel
		res = kernel.setArg(0, *m_data_original);
		res = kernel.setArg(1, *m_data_front);
		res = kernel.setArg(2, convCL);
		
		res = m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, m_globalRange, cl::NullRange);

		res = m_queue.enqueueReadImage(*m_data_front, CL_TRUE, m_origin, m_region, 0, 0, &m_values.front());

		m_queue.finish();

		CopyBufferToImage(m_values, img);
	}
	catch (const cl::Error & err)
	{
		std::cerr << "CustomFilter: " << err.what() << " with error: " << err.err() << std::endl;
	}

}

float GPGPUImplementation::Grayscale(QImage & img)
{
	float ret = 0.f;
	if (false == m_CLready)
	{
		return ret;
	}

	cl_int res;

	try
	{
		cl::Kernel & kernel = *CLManager::GetInstance()->GetKernel(Constants::KERNEL_GRAYSCALE);
		// Set arguments to kernel
		res = kernel.setArg(0, *m_data_original);
		res = kernel.setArg(1, *m_data_front);

		cl::Event event;
		res = m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, m_globalRange, cl::NullRange, nullptr, &event);

		res = m_queue.enqueueReadImage(*m_data_front, CL_TRUE, m_origin, m_region, 0, 0, &m_values.front());

		m_queue.finish();

		ret = event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>();

		CopyBufferToImage(m_values, img);
	}
	catch (const cl::Error & err)
	{
		std::cerr << "Grayscale: " << err.what() << " with error: " << err.err() << std::endl;
	}

	return ret;
}

void GPGPUImplementation::Resize(QImage & img)
{
	if (false == m_CLready)
	{
		return;
	}

	cl_int res;

	try
	{
		cl::Kernel & kernel = *CLManager::GetInstance()->GetKernel(Constants::KERNEL_RESIZE);

		cl::ImageFormat imf = cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8);

		cl::Image2D resized_img = cl::Image2D(m_contextCL, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, imf, m_width / 2, m_height / 2);
		// Set arguments to kernel
		res = kernel.setArg(0, *m_data_original);
		res = kernel.setArg(1, resized_img);
		res = kernel.setArg(2, 2.f);

		res = m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(m_width / 2, m_height / 2), cl::NullRange);

		cl::size_t<3> region;
		region[0] = m_width / 2;
		region[1] = m_height / 2;
		region[2] = 1;

		res = m_queue.enqueueReadImage(resized_img, CL_TRUE, m_origin, region, 0, 0, &m_values.front());

		m_queue.finish();

		CopyBufferToImage(m_values, img, m_height / 2, m_width / 2);
	}
	catch (const cl::Error & err)
	{
		std::cerr << "Resize: " << err.what() << " with error: " << err.err() << std::endl;
	}
}

void GPGPUImplementation::Sobel(QImage & img)
{
	if (false == m_CLready)
	{
		return;
	}

	cl_int res;

	try
	{
		cl::Kernel & kernel = *CLManager::GetInstance()->GetKernel(Constants::KERNEL_CONVOLUTE);
		// first is filter size
		std::vector<float> conv = { 3, -1.f, 0.f, 1.f, -2.f, 0, 2.f, -1.f, 0, 1.f };

		cl::Buffer convCL = cl::Buffer(m_contextCL, CL_MEM_READ_ONLY, conv.size() * sizeof(float), 0, 0);
		m_queue.enqueueWriteBuffer(convCL, CL_TRUE, 0, conv.size() * sizeof(float), &conv[0], 0, NULL);

		// Set arguments to kernel
		res = kernel.setArg(0, *m_data_original);
		res = kernel.setArg(1, *m_data_front);
		res = kernel.setArg(2, convCL);

		res = m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, m_globalRange, cl::NullRange);

		res = m_queue.enqueueReadImage(*m_data_front, CL_TRUE, m_origin, m_region, 0, 0, &m_values.front());

		m_queue.finish();

		CopyBufferToImage(m_values, img);

	}
	catch (const cl::Error & err)
	{
		std::cerr << "Sobel: " << err.what() << " with error: " << err.err() << std::endl;
	}
}

float GPGPUImplementation::GaussianBlur(QImage & img)
{
	float ret = 0.f;
	if (false == m_CLready)
	{
		return ret;
	}

	cl_int res;

	// first is filter size
	std::vector<float> gaussian_kernel = 
	{
		3.f,
		0.102059f, 0.115349f, 0.102059f,
		0.115349f, 0.130371f, 0.115349f,
		0.102059f, 0.115349f, 0.102059f
	};

	try
	{
		cl::Kernel & kernel = *CLManager::GetInstance()->GetKernel(Constants::KERNEL_CONVOLUTE);
		cl::Buffer convCL = cl::Buffer(m_contextCL, CL_MEM_READ_ONLY, gaussian_kernel.size() * sizeof(float), 0, 0);
		m_queue.enqueueWriteBuffer(convCL, CL_TRUE, 0, gaussian_kernel.size() * sizeof(float), &gaussian_kernel[0], 0, NULL);

		// Set arguments to kernel
		res = kernel.setArg(0, *m_data_original);
		res = kernel.setArg(1, *m_data_front);
		res = kernel.setArg(2, convCL);

		auto start = std::chrono::system_clock::now();

		cl::Event event;
		res = m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, m_globalRange, cl::NullRange, nullptr, &event);

		res = m_queue.enqueueReadImage(*m_data_front, CL_TRUE, m_origin, m_region, 0, 0, &m_values.front());

		m_queue.finish();

		ret = event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>();

		CopyBufferToImage(m_values, img);

	}
	catch (const cl::Error & err)
	{
		std::cerr << "GaussianBlur: " << err.what() << " with error: " << err.err() << std::endl;
	}

	return ret;
}

void GPGPUImplementation::Sharpening(QImage & img)
{
	if (false == m_CLready)
	{
		return;
	}

	GaussianBlur(img);

	cl_int res;

	// first is filter size
	std::vector<float> laplacian_kernel(10, -1.0f);
	laplacian_kernel[0] = 3.f;
	laplacian_kernel[5] = 8.f;

	try
	{
		cl::Kernel & kernel = *CLManager::GetInstance()->GetKernel(Constants::KERNEL_SHARPNESS);
		cl::Buffer convCL = cl::Buffer(m_contextCL, CL_MEM_READ_ONLY, laplacian_kernel.size() * sizeof(float), 0, 0);
		m_queue.enqueueWriteBuffer(convCL, CL_TRUE, 0, laplacian_kernel.size() * sizeof(float), &laplacian_kernel[0], 0, NULL);

		// Set arguments to kernel
		res = kernel.setArg(0, *m_data_original);
		res = kernel.setArg(1, *m_data_front);
		res = kernel.setArg(2, convCL);
	
		res = m_queue.enqueueNDRangeKernel(kernel, cl::NullRange,m_globalRange, cl::NullRange);

		res = m_queue.enqueueReadImage(*m_data_front, CL_TRUE, m_origin, m_region, 0, 0, &m_values.front());

		m_queue.finish();

		CopyBufferToImage(m_values, img);
	}
	catch (const cl::Error & err)
	{
		std::cerr << "Sharpening: " << err.what() << " with error: " << err.err() << std::endl;
	}
}

void GPGPUImplementation::ColorSmoothing(QImage & img)
{
	if (false == m_CLready)
	{
		return;
	}

	cl_int res;

	// first is filter size
	std::vector<float> conv(26, 1.f);
	conv[0] = 5.f;

	try
	{
		cl::Kernel & kernel = *CLManager::GetInstance()->GetKernel(Constants::KERNEL_COLOR_SMOOTHING);
		cl::Buffer convCL = cl::Buffer(m_contextCL, CL_MEM_READ_ONLY, conv.size() * sizeof(float), 0, 0);
		m_queue.enqueueWriteBuffer(convCL, CL_TRUE, 0, conv.size() * sizeof(float), &conv[0], 0, NULL);

		// Set arguments to kernel
		res = kernel.setArg(0, *m_data_original);
		res = kernel.setArg(1, *m_data_front);
		res = kernel.setArg(2, convCL);

		res = m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, m_globalRange, cl::NullRange);

		res = m_queue.enqueueReadImage(*m_data_front, CL_TRUE, m_origin, m_region, 0, 0, &m_values.front());

		m_queue.finish();

		CopyBufferToImage(m_values, img);
	}
	catch (const cl::Error & err)
	{
		std::cerr << "ColorSmoothing: " << err.what() << " with error: " << err.err() << std::endl;
	}
}

float GPGPUImplementation::KMeans(QImage & img, const int centroid_count)
{
	float ret = 0.f;
	if (false == m_CLready)
	{
		return ret;
	}

	cl_int res;

	std::vector<Centroid> centroids;

	GenerateCentroids(centroid_count, centroids);

	try
	{
		cl::Kernel & kmeans = *CLManager::GetInstance()->GetKernel(Constants::KERNEL_KMEANS);
		cl::Kernel & kmeans_draw = *CLManager::GetInstance()->GetKernel(Constants::KERNEL_KMEANS_DRAW);
		cl::Kernel & kmeans_update_centroids = *CLManager::GetInstance()->GetKernel(Constants::KERNEL_KMEANS_UPDATE_CENTROIDS);

		cl::Buffer centroidsCL = cl::Buffer(m_contextCL, CL_MEM_READ_WRITE, centroid_count * sizeof(Centroid), 0, 0);
		cl::Buffer bucketsCL = cl::Buffer(m_contextCL, CL_MEM_READ_WRITE, centroid_count * m_width * m_height * sizeof(cl_float3), 0, 0);
		res = m_queue.enqueueWriteBuffer(centroidsCL, CL_TRUE, 0, centroid_count * sizeof(Centroid), &centroids[0], 0, NULL);
		
		// Set arguments to kmeans kernel
		res = kmeans.setArg(0, *m_data_original);
		res = kmeans.setArg(1, centroidsCL);
		res = kmeans.setArg(2, bucketsCL);
		res = kmeans.setArg(3, m_width);
		res = kmeans.setArg(4, m_height);
		res = kmeans.setArg(5, centroid_count);

		// Set arguments to update centroids kernel
		res = kmeans_update_centroids.setArg(0, centroidsCL);
		res = kmeans_update_centroids.setArg(1, bucketsCL);
		res = kmeans_update_centroids.setArg(2, m_width);
		res = kmeans_update_centroids.setArg(3, m_height);

		cl::Event event;

		for (int i = 0; i < 20; ++i)
		{
			// Run the kmeans kernel
			res = m_queue.enqueueNDRangeKernel(kmeans, cl::NullRange, m_globalRange, cl::NullRange, nullptr, &event);
			m_queue.finish();

			ret += event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>();


			// Run the update centroid kernel
			for (int uc = 0; uc < centroid_count; ++uc)
			{
				res = kmeans_update_centroids.setArg(4, uc);

				res = m_queue.enqueueNDRangeKernel(kmeans_update_centroids, cl::NullRange, m_globalRange, cl::NullRange, nullptr, &event);
				m_queue.finish();
				ret += event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
			}
		}

		// Run a last iteration of K-Means that will also change de colors
		res = kmeans_draw.setArg(0, *m_data_original);
		res = kmeans_draw.setArg(1, *m_data_front);
		res = kmeans_draw.setArg(2, centroidsCL);
		res = kmeans_draw.setArg(3, centroid_count);

		res = m_queue.enqueueNDRangeKernel(kmeans_draw, cl::NullRange, m_globalRange, cl::NullRange, nullptr, &event);
		m_queue.finish();

		ret += event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>();

		
		res = m_queue.enqueueReadImage(*m_data_front, CL_TRUE, m_origin, m_region, 0, 0, &m_values.front());

		m_queue.finish();

		CopyBufferToImage(m_values, img);
	}
	catch (const cl::Error & err)
	{
		std::cerr << "KMeans: " << err.what() << " with error: " << err.err() << std::endl;
	}

	return ret;
}

float GPGPUImplementation::SOMSegmentation(QImage & img, QImage * ground_truth)
{
	float ret = 0.f;
	if (false == m_CLready)
	{
		return ret;
	}

	bool got_gt = false;
	std::pair<float, float> gt_values;
	if (nullptr != ground_truth)
	{
		got_gt = true;
		gt_values = ComputeVMAndDBIndices(ground_truth);
		m_log("[SOM segmentation Ground Truth]: VM = " + std::to_string(gt_values.first) + ", DBI = " + std::to_string(gt_values.second));
	}

	cl_int res;

	int iterations = 1;
	int neuron_count = 3;
	int noise_kern_size = 3;
	int epochs = 200; // number of iterations

	uint32_t total_sz = img.width() * img.height();
	const double ct_learning_rate = 0.1;
	const double time_constant = epochs / log(neuron_count);

	// Node initialization: 1D topology
	std::vector<Neuron> neurons;
	std::vector<float> distances(neuron_count);

	GenerateNeurons(neuron_count, neurons);

	try
	{
		cl::Kernel & som_find_bmu = *CLManager::GetInstance()->GetKernel(Constants::KERNEL_SOM_FIND_BMU);
		cl::Kernel & som_update_wieghts = *CLManager::GetInstance()->GetKernel(Constants::KERNEL_SOM_UPDATE_WEIGHTS);
		cl::Kernel & som_draw = *CLManager::GetInstance()->GetKernel(Constants::KERNEL_SOM_DRAW);
		cl::Kernel & noise_reduction = *CLManager::GetInstance()->GetKernel(Constants::KERNEL_POST_NOISE_REDUCTION);

		cl::Buffer neuronsCL = cl::Buffer(m_contextCL, CL_MEM_READ_WRITE, neurons.size() * sizeof(Neuron), 0, 0);
		m_queue.enqueueWriteBuffer(neuronsCL, CL_TRUE, 0, neurons.size() * sizeof(Neuron), &neurons[0], 0, NULL);

		cl::Buffer distancesCL = cl::Buffer(m_contextCL, CL_MEM_READ_WRITE, distances.size() * sizeof(float), 0, 0);
		m_queue.enqueueWriteBuffer(distancesCL, CL_TRUE, 0, distances.size() * sizeof(float), &distances[0], 0, NULL);

		int bmu_idx = 0;
		cl::Buffer bmu_idxCL = cl::Buffer(m_contextCL, CL_MEM_READ_WRITE, sizeof(int), 0, 0);
		m_queue.enqueueWriteBuffer(bmu_idxCL, CL_TRUE, 0, sizeof(int), &bmu_idx, 0, NULL);

		// Set arguments that dont change on host
		res = som_find_bmu.setArg(1, neuronsCL);
		res = som_find_bmu.setArg(2, distancesCL);
		res = som_find_bmu.setArg(3, bmu_idxCL);
		res = som_find_bmu.setArg(4, neuron_count);

		// Set update weights arguments
		res = som_update_wieghts.setArg(1, bmu_idxCL);
		res = som_update_wieghts.setArg(2, neuronsCL);
		res = som_update_wieghts.setArg(3, neuron_count);

		cl::Event event;

		for (int iter = 0; iter < iterations; ++iter)
		{
			for (int epoch = 0; epoch < epochs; ++epoch)
			{
				// Chose a random input value
				size_t index = (std::rand() % total_sz) * 4;

				cl_float3 value =
				{
					static_cast<cl_float>(m_values_orig[index] / 255.f),
					static_cast<cl_float>(m_values_orig[index + 1] / 255.f),
					static_cast<cl_float>(m_values_orig[index + 2] / 255.f)
				};

				// Set value argument
				res = som_find_bmu.setArg(0, value);

				res = m_queue.enqueueNDRangeKernel(som_find_bmu, cl::NullRange, cl::NDRange(neuron_count, 1), cl::NullRange, nullptr, &event);
				m_queue.finish();

				ret += event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>();

				// update weights
				float neigh_dist = (neurons.size() - 1) * exp(-static_cast<double>(epoch) / time_constant);
				float learning_rate = ct_learning_rate * exp(-static_cast<double>(epoch) / epochs);

				res = som_update_wieghts.setArg(0, value);
				res = som_update_wieghts.setArg(4, neigh_dist);
				res = som_update_wieghts.setArg(5, learning_rate);

				res = m_queue.enqueueNDRangeKernel(som_update_wieghts, cl::NullRange, cl::NDRange(neuron_count, 1), cl::NullRange, nullptr, &event);
				m_queue.finish();

				ret += event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
			}

			std::pair<float, float> qRes = CheckSegmentationNeurons(neuronsCL, neurons);

			m_queue.finish();

			if ((true == got_gt) &&
				(std::abs(gt_values.first - qRes.first) < 0.1f || std::abs(gt_values.second - qRes.second) < 0.1f))
			{
				break;
			}
		}

		res = som_draw.setArg(0, *m_data_original);
		res = som_draw.setArg(1, *m_data_front);
		res = som_draw.setArg(2, neuronsCL);
		res = som_draw.setArg(3, neuron_count);
		res = m_queue.enqueueNDRangeKernel(som_draw, cl::NullRange, m_globalRange, cl::NullRange, nullptr, &event);
		m_queue.finish();

		ret += event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>();

		//res = noise_reduction.setArg(0, *m_data_front);
		//res = noise_reduction.setArg(1, *m_data_back);
		//res = noise_reduction.setArg(2, neuronsCL);
		//res = noise_reduction.setArg(3, neuron_count);
		//res = noise_reduction.setArg(4, noise_kern_size);
		//res = m_queue.enqueueNDRangeKernel(noise_reduction, cl::NullRange, m_globalRange, cl::NullRange);

		/// todo: change to m_data_back if nosie reduction is used
		res = m_queue.enqueueReadImage(*m_data_front, CL_TRUE, m_origin, m_region, 0, 0, &m_values.front());

		m_queue.finish();
		CopyBufferToImage(m_values, img);

	}
	catch (const cl::Error & err)
	{
		std::cerr << "SOMSegmentation: " << err.what() << " with error: " << err.err() << std::endl;
	}

	return ret;
}

void GPGPUImplementation::Threshold(QImage & img, const float value)
{
	if (false == m_CLready)
		return;
	
	cl_int res;

	try
	{
		cl::Kernel & kernel = *CLManager::GetInstance()->GetKernel(Constants::KERNEL_THRESHOLD);
		// Set arguments to kernel
		res = kernel.setArg(0, *m_data_original);
		res = kernel.setArg(1, *m_data_front);
		res = kernel.setArg(2, value);

		res = m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, m_globalRange, cl::NullRange);

		res = m_queue.enqueueReadImage(*m_data_front, CL_TRUE, m_origin, m_region, 0, 0, &m_values.front());

		m_queue.finish();

		CopyBufferToImage(m_values, img);
	}
	catch (const cl::Error & err)
	{
		std::cerr << "Threshold: " << err.what() << " with error: " << err.err() << std::endl;
	}
}

void GPGPUImplementation::RunSIFT(QImage & img)
{
	try
	{
		cl_int res;
		cl::ImageFormat imf = cl::ImageFormat(CL_RGBA, CL_FLOAT);

		cl::Kernel kernel = *CLManager::GetInstance()->GetKernel(Constants::KERNEL_GRAYSCALE);
		// Set arguments to kernel
		cl::Image2D grayscaled = cl::Image2D(m_contextCL, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, imf, img.width(), img.height());

		res = kernel.setArg(0, *m_data_original);
		res = kernel.setArg(1, grayscaled);

		std::vector<float> valuesf(m_values.size());

		res = m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, m_globalRange, cl::NullRange);

		m_queue.finish();
		//res = m_queue.enqueueReadImage(grayscaled, CL_TRUE, m_origin, m_region, 0, 0, &valuesf.front());
		//
		//m_queue.finish();
		//for (size_t i = 0; i < valuesf.size(); ++i)
		//{
		//	m_values[i] = valuesf[i] * 255;
		//}
		//
		//CopyBufferToImage(m_values, img);

		cl::Image2D * sift_output = m_sift.Run(&grayscaled, m_width, m_height);
		
		res = m_queue.enqueueReadImage(*sift_output, CL_TRUE, m_origin, m_region, 0, 0, &valuesf.front());
		
		m_queue.finish();
		for (size_t i = 0; i < valuesf.size(); ++i)
		{
			m_values[i] = valuesf[i] * 255;
		}
		
		CopyBufferToImage(m_values, img);
	}
	catch (const cl::Error & err)
	{
		std::cerr << "RunSIFT: " << err.what() << " with error: " << err.err() << std::endl;
	}
}

std::vector<float> GPGPUImplementation::FindImageSIFT(QImage & img, QImage & img_to_find)
{
	std::vector<float> rect;

	try
	{
		cl_int res;
		cl::ImageFormat imf = cl::ImageFormat(CL_RGBA, CL_FLOAT);

		cl::Kernel kernel = *CLManager::GetInstance()->GetKernel(Constants::KERNEL_GRAYSCALE);
		
		// Default image
		cl::Image2D grayscaled = cl::Image2D(m_contextCL, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, imf, img.width(), img.height());

		res = kernel.setArg(0, *m_data_original);
		res = kernel.setArg(1, grayscaled);
		
		res = m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, m_globalRange, cl::NullRange);

		m_queue.finish();

		// Image to find
		//cl::Image2D img_tf = cl::Image2D(m_contextCL, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, imf, img_to_find.width(), img_to_find.height(), 0, 0, &res);
		cl::Image2D grayscaled_tf= cl::Image2D(m_contextCL, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, imf, img_to_find.width(), img_to_find.height(), 0, 0, &res);

		cl::size_t<3> origin, region;
		origin[0] = 0; origin[1] = 0, origin[2] = 0;
		region[0] = img_to_find.width(); region[1] = img_to_find.height(); region[2] = 1;

		cl::NDRange range = cl::NDRange(img_to_find.width(), img_to_find.height());

		res = m_queue.enqueueWriteImage(*m_data_back, CL_TRUE, origin, region, 0, 0, img_to_find.bits());
		//res = m_queue.enqueueWriteImage(*m_data_original, CL_TRUE, m_origin, m_region, 0, 0, img_to_find.bits());

		res = kernel.setArg(0, *m_data_back);
		res = kernel.setArg(1, grayscaled_tf);

		res = m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, range, cl::NullRange);

		m_queue.finish();

		rect = m_sift.FindImage(&grayscaled, m_width, m_height, &grayscaled_tf, img_to_find.width(), img_to_find.height());
	}
	catch (const cl::Error & err)
	{
		std::cerr << "FindImageSIFT: " << err.what() << " with error: " << err.err() << std::endl;
	}

	return rect;
}

std::pair<float, float> GPGPUImplementation::CheckSegmentationNeurons(cl::Buffer & neuronsCL, std::vector<Neuron> & neurons)
{
	float VM = 0.f, DBI = 0.f;

	try {
		cl_int res;
		res = m_queue.enqueueReadBuffer(neuronsCL, CL_TRUE, 0, neurons.size() * sizeof(Neuron), &neurons[0], 0, NULL);
		m_queue.finish();

		std::string to_log = "Final neurons: ";
		for (Neuron & neuron : neurons)
		{
			to_log += " (" + std::to_string(neuron.value_x) + ", " + std::to_string(neuron.value_y) + ", " + std::to_string(neuron.value_z) + ")";
		}
		to_log += "\n";
		m_log(to_log);

		VM = ValidityMeasure(m_values_orig, neurons);
		DBI = DaviesBouldinIndex(m_values_orig, neurons);

		m_log("[SOM segmentation]: VM = " + std::to_string(VM) + ", DBI = " + std::to_string(DBI));
	}
	catch (const cl::Error & err)
	{
		std::cerr << "CheckSegmentationNeurons: " << err.what() << " with error: " << err.err() << std::endl;
	}

	return{ VM, DBI };
}
