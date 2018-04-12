
#include "GPGPUImplementation.h"
#include "Constants.h"

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
	
	InitializeCL();
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


void GPGPUImplementation::PrintDevices(const std::vector<cl::Device>& devices) const
{
	size_t i = 0;
	for (cl::Device d : devices) {
		std::cout << "Device ID: " << i++ << std::endl;
		std::cout << "\tDevice Name: " << d.getInfo<CL_DEVICE_NAME>() << std::endl;
		std::cout << "\tDevice Type: " << d.getInfo<CL_DEVICE_TYPE>();
		std::cout << " (GPU: " << CL_DEVICE_TYPE_GPU << ", CPU: " << CL_DEVICE_TYPE_CPU << ")" << std::endl;
		std::cout << "\tDevice Vendor: " << d.getInfo<CL_DEVICE_VENDOR>() << std::endl;
		std::cout << "\tDevice Max Compute Units: " << d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
		std::cout << "\tDevice Global Memory: " << d.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() << std::endl;
		std::cout << "\tDevice Max Clock Frequency: " << d.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << std::endl;
		std::cout << "\tDevice Max Allocateable Memory: " << d.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() << std::endl;
		std::cout << "\tDevice Local Memory: " << d.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << std::endl;
		std::cout << "\tDevice Available: " << d.getInfo< CL_DEVICE_AVAILABLE>() << std::endl;
	}
	std::cout << std::endl;
}

void GPGPUImplementation::LoadProgram(cl::Program & program, std::vector<cl::Device> & devices, std::string file)
{
	std::ifstream sourceFile(file);
	std::string sourceCode(
		std::istreambuf_iterator<char>(sourceFile),
		(std::istreambuf_iterator<char>()));
	sourceFile.close();
	
	try
	{
		cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));
		// Make program of the source code in the context
		program = cl::Program(m_contextCL, source);
		// Build program for these specific devices

		program.build(devices);
	}
	catch (const cl::Error & err)
	{
		std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
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

	try
	{
		cl::ImageFormat imf = cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8);

		m_data_original = new cl::Image2D(m_contextCL, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, imf, m_width, m_height);
		m_data_front = new cl::Image2D(m_contextCL, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, imf, m_width, m_height);
		m_data_back = new cl::Image2D(m_contextCL, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, imf, m_width, m_height);

		res = m_queue.enqueueWriteImage(*m_data_original, CL_TRUE, m_origin, m_region, 0, 0, img.bits());

		m_globalRange = cl::NDRange(m_width, m_height);
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
		cl::Kernel & kernel = m_kernels[Constants::KERNEL_GRAYSCALE];
		// Set arguments to kernel
		res = kernel.setArg(0, *m_data_original);
		res = kernel.setArg(1, *m_data_front);

		auto start = std::chrono::system_clock::now();
		res = m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, m_globalRange, cl::NullRange);

		res = m_queue.enqueueReadImage(*m_data_front, CL_TRUE, m_origin, m_region, 0, 0, &m_values.front());

		m_queue.finish();
		auto end = std::chrono::system_clock::now();

		ret = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

		CopyBufferToImage(m_values, img);
	}
	catch (const cl::Error & err)
	{
		std::cerr << "Grayscale: " << err.what() << " with error: " << err.err() << std::endl;
	}

	return ret;
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
		cl::Kernel & kernel = m_kernels[Constants::KERNEL_CONVOLUTE];
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
		cl::Kernel & kernel = m_kernels[Constants::KERNEL_CONVOLUTE];
		cl::Buffer convCL = cl::Buffer(m_contextCL, CL_MEM_READ_ONLY, gaussian_kernel.size() * sizeof(float), 0, 0);
		m_queue.enqueueWriteBuffer(convCL, CL_TRUE, 0, gaussian_kernel.size() * sizeof(float), &gaussian_kernel[0], 0, NULL);

		// Set arguments to kernel
		res = kernel.setArg(0, *m_data_original);
		res = kernel.setArg(1, *m_data_front);
		res = kernel.setArg(2, convCL);

		auto start = std::chrono::system_clock::now();

		res = m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, m_globalRange, cl::NullRange);

		res = m_queue.enqueueReadImage(*m_data_front, CL_TRUE, m_origin, m_region, 0, 0, &m_values.front());

		m_queue.finish();

		auto end = std::chrono::system_clock::now();

		ret = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

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
		cl::Kernel & kernel = m_kernels[Constants::KERNEL_SHARPNESS];
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
		cl::Kernel & kernel = m_kernels[Constants::KERNEL_COLOR_SMOOTHING];
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
		cl::Kernel & kmeans = m_kernels[Constants::KERNEL_KMEANS];
		cl::Kernel & kmeans_draw = m_kernels[Constants::KERNEL_KMEANS_DRAW];
		cl::Kernel & kmeans_update_centroids = m_kernels[Constants::KERNEL_KMEANS_UPDATE_CENTROIDS];

		cl::Buffer centroidsCL = cl::Buffer(m_contextCL, CL_MEM_READ_WRITE, centroids.size() * sizeof(Centroid), 0, 0);
		res = m_queue.enqueueWriteBuffer(centroidsCL, CL_TRUE, 0, centroids.size() * sizeof(Centroid), &centroids[0], 0, NULL);

		// Set arguments to kmeans kernel
		res = kmeans.setArg(0, *m_data_original);
		res = kmeans.setArg(1, centroidsCL);
		res = kmeans.setArg(2, centroid_count);

		// Set arguments to update centroids kernel
		res = kmeans_update_centroids.setArg(0, centroidsCL);
		res = kmeans_update_centroids.setArg(1, centroid_count);

		auto start = std::chrono::system_clock::now();

		for (int i = 0; i < 5; ++i)
		{
			// Run the kmeans kernel
			res = m_queue.enqueueNDRangeKernel(kmeans, cl::NullRange, m_globalRange, cl::NullRange);

			// Run the update centroid kernel
			res = m_queue.enqueueNDRangeKernel(kmeans_update_centroids, cl::NullRange, cl::NDRange(centroid_count, 1), cl::NullRange);
		}

		// Run a last iteration of K-Means that will also change de colors
		res = kmeans_draw.setArg(0, *m_data_original);
		res = kmeans_draw.setArg(1, *m_data_front);
		res = kmeans_draw.setArg(2, centroidsCL);
		res = kmeans_draw.setArg(3, centroid_count);

		res = m_queue.enqueueNDRangeKernel(kmeans_draw, cl::NullRange, m_globalRange, cl::NullRange);
		
		res = m_queue.enqueueReadImage(*m_data_front, CL_TRUE, m_origin, m_region, 0, 0, &m_values.front());

		m_queue.finish();

		auto end = std::chrono::system_clock::now();

		ret = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

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
		cl::Kernel & som_find_bmu = m_kernels[Constants::KERNEL_SOM_FIND_BMU];
		cl::Kernel & som_update_wieghts = m_kernels[Constants::KERNEL_SOM_UPDATE_WEIGHTS];
		cl::Kernel & som_draw = m_kernels[Constants::KERNEL_SOM_DRAW];
		cl::Kernel & noise_reduction = m_kernels[Constants::KERNEL_POST_NOISE_REDUCTION];

		cl::Buffer neuronsCL = cl::Buffer(m_contextCL, CL_MEM_READ_ONLY, neurons.size() * sizeof(Neuron), 0, 0);
		m_queue.enqueueWriteBuffer(neuronsCL, CL_TRUE, 0, neurons.size() * sizeof(Neuron), &neurons[0], 0, NULL);

		cl::Buffer distancesCL = cl::Buffer(m_contextCL, CL_MEM_READ_ONLY, distances.size() * sizeof(float), 0, 0);
		m_queue.enqueueWriteBuffer(distancesCL, CL_TRUE, 0, distances.size() * sizeof(float), &distances[0], 0, NULL);

		int bmu_idx = 0;
		cl::Buffer bmu_idxCL = cl::Buffer(m_contextCL, CL_MEM_READ_ONLY, sizeof(int), 0, 0);
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

		auto start = std::chrono::system_clock::now();

		for (int iter = 0; iter < iterations; ++iter)
		{
			for (int epoch = 0; epoch < epochs; ++epoch)
			{
				// Chose a random input value
				size_t index = (std::rand() % total_sz) * 4;

				cl_uint3 value =
				{
					static_cast<cl_uint>(m_values_orig[index]),
					static_cast<cl_uint>(m_values_orig[index + 1]),
					static_cast<cl_uint>(m_values_orig[index + 2])
				};

				// Set value argument
				res = som_find_bmu.setArg(0, value);

				res = m_queue.enqueueNDRangeKernel(som_find_bmu, cl::NullRange, cl::NDRange(neuron_count, 1), cl::NullRange);
				//m_queue.finish();
				//m_queue.enqueueReadBuffer(bmu_idxCL, CL_TRUE, 0, sizeof(int), &bmu_idx, 0, NULL);
				//
				//std::cout << bmu_idx << std::endl;

				// update weights
				float neigh_dist = (neurons.size() - 1) * exp(-static_cast<double>(epoch) / time_constant);
				float learning_rate = ct_learning_rate * exp(-static_cast<double>(epoch) / epochs);

				res = som_update_wieghts.setArg(0, value);
				res = som_update_wieghts.setArg(4, neigh_dist);
				res = som_update_wieghts.setArg(5, learning_rate);

				res = m_queue.enqueueNDRangeKernel(som_update_wieghts, cl::NullRange, cl::NDRange(neuron_count, 1), cl::NullRange);
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
		res = m_queue.enqueueNDRangeKernel(som_draw, cl::NullRange, m_globalRange, cl::NullRange);

		res = noise_reduction.setArg(0, *m_data_front);
		res = noise_reduction.setArg(1, *m_data_back);
		res = noise_reduction.setArg(2, neuronsCL);
		res = noise_reduction.setArg(3, neuron_count);
		res = noise_reduction.setArg(4, noise_kern_size);
		res = m_queue.enqueueNDRangeKernel(noise_reduction, cl::NullRange, m_globalRange, cl::NullRange);

		
		res = m_queue.enqueueReadImage(*m_data_back, CL_TRUE, m_origin, m_region, 0, 0, &m_values.front());
		m_queue.finish();

		auto end = std::chrono::system_clock::now();

		ret = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();


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
		cl::Kernel & kernel = m_kernels[Constants::KERNEL_THRESHOLD];
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

void GPGPUImplementation::InitializeCL()
{
	// Used for exit codes
	cl_int res;

	// Get available platforms
	std::vector<cl::Platform> platforms;
	std::vector<cl::Device> devices;
	cl::Platform::get(&platforms);
	// Select the default platform and create a context using this platform and the GPU
	cl_context_properties cps[] = 
	{
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)(platforms[0])(),
		0
	};

	try
	{
		m_contextCL = cl::Context(CL_DEVICE_TYPE_GPU, cps);
		// Get a list of devices on this platform
		devices = m_contextCL.getInfo<CL_CONTEXT_DEVICES>();
		PrintDevices(devices);
		// Create a command queue and use the first device
		m_queue = cl::CommandQueue(m_contextCL, devices[0]);

		cl::Program program;
		LoadProgram(program, devices, "Kernels/utils.cl");
		m_kernels[Constants::KERNEL_GRAYSCALE] = cl::Kernel(program, Constants::KERNEL_GRAYSCALE);

		LoadProgram(program, devices, "Kernels/filters.cl");
		m_kernels[Constants::KERNEL_CONVOLUTE] = cl::Kernel(program, Constants::KERNEL_CONVOLUTE);
		m_kernels[Constants::KERNEL_COLOR_SMOOTHING] = cl::Kernel(program, Constants::KERNEL_COLOR_SMOOTHING);
		m_kernels[Constants::KERNEL_SHARPNESS] = cl::Kernel(program, Constants::KERNEL_SHARPNESS);

		LoadProgram(program, devices, "Kernels/segmentation.cl");
		m_kernels[Constants::KERNEL_KMEANS] = cl::Kernel(program, Constants::KERNEL_KMEANS);
		m_kernels[Constants::KERNEL_KMEANS_DRAW] = cl::Kernel(program, Constants::KERNEL_KMEANS_DRAW);
		m_kernels[Constants::KERNEL_KMEANS_UPDATE_CENTROIDS] = cl::Kernel(program, Constants::KERNEL_KMEANS_UPDATE_CENTROIDS);
		m_kernels[Constants::KERNEL_THRESHOLD] = cl::Kernel(program, Constants::KERNEL_THRESHOLD);

		LoadProgram(program, devices, "Kernels/som.cl");
		m_kernels[Constants::KERNEL_SOM_FIND_BMU] = cl::Kernel(program, Constants::KERNEL_SOM_FIND_BMU);
		m_kernels[Constants::KERNEL_SOM_UPDATE_WEIGHTS] = cl::Kernel(program, Constants::KERNEL_SOM_UPDATE_WEIGHTS);
		m_kernels[Constants::KERNEL_SOM_DRAW] = cl::Kernel(program, Constants::KERNEL_SOM_DRAW);

		LoadProgram(program, devices, "Kernels/post_processing.cl");
		m_kernels[Constants::KERNEL_POST_NOISE_REDUCTION] = cl::Kernel(program, Constants::KERNEL_POST_NOISE_REDUCTION);


		m_CLready = true;
	}
	catch (const cl::Error & err)
	{
		std::cerr << "InitializeCL: " << err.what() << " with error: " << err.err() << std::endl;
	}
}

std::pair<float, float> GPGPUImplementation::CheckSegmentationNeurons(cl::Buffer & neuronsCL, std::vector<Neuron> & neurons)
{
	float VM = 0.f, DBI = 0.f;

	try {
		cl_int res;
		res = m_queue.enqueueReadBuffer(neuronsCL, CL_TRUE, 0, neurons.size() * sizeof(Neuron), &neurons[0], 0, NULL);
		m_queue.finish();

		std::cout << "Final neurons: ";
		for (Neuron & neuron : neurons)
		{
			std::cout << " (" << neuron.x << ", " << neuron.y << ", " << neuron.z << ")";
		}
		std::cout << std::endl;

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

float GPGPUImplementation::GaussianFunction(int niu, int thetha, int cluster_count) const
{
	static const float double_pi = 2 * 3.14159f;
	float pow_thetha = thetha * thetha;
	float pow_k_niu = (cluster_count - niu) * (cluster_count - niu);

	return expf(-(pow_k_niu / (2*pow_thetha))) / sqrtf(double_pi * pow_thetha);
}

float GPGPUImplementation::NormalizedEuclideanDistance(const Neuron & n1, const Neuron & n2) const
{
	unsigned x = n1.x - n2.x;
	unsigned y = n1.y - n2.y;
	unsigned z = n1.z - n2.z;

	return sqrt(x*x + y*y + z*z);
}

float GPGPUImplementation::ValidityMeasure(const std::vector<uchar>& data, const std::vector<Neuron>& neurons) const
{
	static const int c = 15; // can be between 15 and 25
	float intra_distance = 0.f;

	for (int i = 0; i < data.size(); i += 4)
	{
		float min_dist = FLT_MAX;
		size_t c_idx = -1;

		Neuron crt_pixel =
		{
			static_cast<unsigned>(data[i]),
			static_cast<unsigned>(data[i + 1]),
			static_cast<unsigned>(data[i + 2]) 
		};

		for (int n_idx = 0; n_idx < neurons.size(); ++n_idx) {
			
			float dist = NormalizedEuclideanDistance(crt_pixel, neurons[n_idx]);
			
			dist *= dist;

			if (dist < min_dist)
			{
				min_dist = dist;
				c_idx = n_idx;
			}
		}

		intra_distance += min_dist;
	}

	intra_distance /= (data.size() / 4.f);

	float inter_distance = FLT_MAX;
	for (int i = 0; i < neurons.size(); ++i)
	{
		for (int j = 0; j < neurons.size(); ++j)
		{
			if (i == j) {
				continue;
			}

			float dist = NormalizedEuclideanDistance(neurons[i], neurons[j]);

			dist *= dist;
			
			if (dist < inter_distance)
			{
				inter_distance = dist;
			}
		}
	}

	float y = c * GaussianFunction(2, 1, neurons.size()) + 1;

	return y * (intra_distance / inter_distance);
}

float GPGPUImplementation::DaviesBouldinIndex(const std::vector<uchar>& data, const std::vector<Neuron>& neurons) const
{
	std::vector<int> cluster_size(neurons.size(), 0);
	std::vector<float> cluster_distances(neurons.size(), 0.f);

	for (int i = 0; i < data.size(); i += 4)
	{
		float min_dist = FLT_MAX;
		size_t c_idx = -1;

		Neuron crt_pixel =
		{
			static_cast<unsigned>(data[i]),
			static_cast<unsigned>(data[i + 1]),
			static_cast<unsigned>(data[i + 2])
		};

		for (int n_idx = 0; n_idx < neurons.size(); ++n_idx)
		{

			float dist = NormalizedEuclideanDistance(crt_pixel, neurons[n_idx]);

			if (dist < min_dist)
			{
				min_dist = dist;
				c_idx = n_idx;
			}
		}

		++cluster_size[c_idx];

		cluster_distances[c_idx] += min_dist;
	}

	// returns a safety error, doing it by hand
	//std::transform(cluster_distances.begin(), cluster_distances.end(), cluster_size.begin(), cluster_distances, std::divides<float>());
	for (int i = 0; i < cluster_distances.size(); ++i)
	{
		cluster_distances[i] /= static_cast<float>(cluster_size[i]);
	}

	std::vector<float> D(neurons.size(), 0.f);
	for (int i = 0; i < neurons.size(); ++i)
	{
		for (int j = 0; j < neurons.size(); ++j)
		{
			if (i == j)
			{
				continue;
			}

			float dist = NormalizedEuclideanDistance(neurons[i], neurons[j]);
			float Rij = (cluster_distances[i] + cluster_distances[j]) / dist;


			if (Rij > D[i])
			{
				D[i] = Rij;
			}
		}
	}

	return std::accumulate(D.begin(), D.end(), 0.f) / D.size();
}

std::pair<float, float> GPGPUImplementation::ComputeVMAndDBIndices(QImage * img)
{
	std::vector<uchar> values(img->byteCount());
	
	CopyImageToBuffer(*img, values);

	std::set<QRgb> uq_neuron;
	for (int i = 0; i < img->height(); ++i)
	{
		for (int j = 0; j < img->width(); ++j)
		{
			QRgb px = img->pixel(QPoint(i, j));

			//std::cout << "[Ground Truth]: (" << i << "," << j << ") = (" + std::to_string(qRed(px)) + ", " + std::to_string(qGreen(px)) + ", " + std::to_string(qBlue(px)) + ")" << std::endl;
			uq_neuron.insert(px);
		}
	}

	std::vector<Neuron> neurons;
	for (auto n : uq_neuron)
	{
		Neuron nv;
		nv.x = qRed(n) / 256.f;
		nv.y = qGreen(n) / 256.f;
		nv.z = qBlue(n) / 256.f;

		neurons.push_back(nv);
	}

	float VM = ValidityMeasure(values, neurons);
	float DBI = DaviesBouldinIndex(values, neurons);

	return{ VM, DBI };
}
