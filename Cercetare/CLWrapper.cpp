
#include "CLWrapper.hpp"

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <set>

CLWrapper::CLWrapper()
	: m_initialized(false), m_data(nullptr), m_data_aux(nullptr)
{
	std::srand(static_cast <unsigned> (time(0)));
	
	InitializeCL();
}

CLWrapper::~CLWrapper()
{
	if (nullptr == m_data)
		delete m_data;

	if (nullptr == m_data_aux)
		delete m_data_aux;
}

void CLWrapper::SetLogFunction(std::function<void(std::string)> log_func)
{
	m_log = log_func;
}

void CLWrapper::LoadProgram(cl::Program & program, std::vector<cl::Device> & devices, std::string file)
{
	std::ifstream sourceFile(file);
	std::string sourceCode(
		std::istreambuf_iterator<char>(sourceFile),
		(std::istreambuf_iterator<char>()));
	sourceFile.close();
	
	try {
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

void CLWrapper::LoadData(QImage & img)
{
	cl_int res;

	m_width = img.width();
	m_height = img.height();

	m_origin[0] = 0; m_origin[1] = 0, m_origin[2] = 0;
	m_region[0] = m_width; m_region[1] = m_height; m_region[2] = 1;

	m_values.resize(m_width * m_height * 4);

	try {
		cl::ImageFormat imf = cl::ImageFormat(CL_RGBA, CL_UNORM_INT8);

		m_data = new cl::Image2D(m_contextCL, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, imf, m_width, m_height);
		m_data_aux = new cl::Image2D(m_contextCL, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, imf, m_width, m_height);

		res = m_queue.enqueueWriteImage(*m_data, CL_TRUE, m_origin, m_region, 0, 0, img.bits());

		m_globalRange = cl::NDRange(m_width, m_height);
	}
	catch (const cl::Error & err)
	{
		std::cerr << "LoadData: " << err.what() << std::endl;
	}
}

void CLWrapper::SetData(QImage & img)
{
	LoadData(img);

	m_initialized = true;
}

void CLWrapper::Grayscale(QImage & img)
{
	if (false == m_CLready)
		return;

	cl_int res;

	try {
		// Set arguments to kernel
		res = m_kernel_grayscale.setArg(0, *m_data);
		res = m_kernel_grayscale.setArg(1, *m_data_aux);

		res = m_queue.enqueueNDRangeKernel(m_kernel_grayscale, cl::NullRange, m_globalRange, cl::NullRange);

		res = m_queue.enqueueReadImage(*m_data_aux, CL_TRUE, m_origin, m_region, 0, 0, &m_values.front());

		m_queue.finish();

		for (int i = 0, row = 0; row < img.height(); ++row, i += img.bytesPerLine()) {
			memcpy(img.scanLine(row), &m_values[i], img.bytesPerLine());
		}

		img.convertToFormat(QImage::Format_Grayscale8);
	}
	catch (const cl::Error & err)
	{
		std::cerr << "Grayscale: " << err.what() << std::endl;
	}
}

void CLWrapper::Sobel(QImage & img)
{
	if (false == m_CLready)
		return;

	cl_int res;

	try {
		// first is filter size
		std::vector<float> conv = { 3, -1.f, 0.f, 1.f, -2.f, 0, 2.f, -1.f, 0, 1.f };

		cl::Buffer convCL = cl::Buffer(m_contextCL, CL_MEM_READ_ONLY, conv.size() * sizeof(float), 0, 0);
		m_queue.enqueueWriteBuffer(convCL, CL_TRUE, 0, conv.size() * sizeof(float), &conv[0], 0, NULL);

		// Set arguments to kernel
		res = m_kernel_convolute.setArg(0, *m_data);
		res = m_kernel_convolute.setArg(1, *m_data_aux);
		res = m_kernel_convolute.setArg(2, convCL);

		res = m_queue.enqueueNDRangeKernel(m_kernel_convolute, cl::NullRange, m_globalRange, cl::NullRange);

		res = m_queue.enqueueReadImage(*m_data_aux, CL_TRUE, m_origin, m_region, 0, 0, &m_values.front());

		m_queue.finish();

		for (int i = 0, row = 0; row < img.height(); ++row, i += img.bytesPerLine()) {
			memcpy(img.scanLine(row), &m_values[i], img.bytesPerLine());
		}

	}
	catch (const cl::Error & err)
	{
		std::cerr << "Sobel: " << err.what() << std::endl;
	}
}

void CLWrapper::GaussianBlur(QImage & img)
{
	if (false == m_CLready)
		return;

	cl_int res;

	// first is filter size
	std::vector<float> gaussian_kernel = { 3.f,
		0.102059f, 0.115349f, 0.102059f,
		0.115349f, 0.130371f, 0.115349f,
		0.102059f, 0.115349f, 0.102059f };

	try {
		cl::Buffer convCL = cl::Buffer(m_contextCL, CL_MEM_READ_ONLY, gaussian_kernel.size() * sizeof(float), 0, 0);
		m_queue.enqueueWriteBuffer(convCL, CL_TRUE, 0, gaussian_kernel.size() * sizeof(float), &gaussian_kernel[0], 0, NULL);

		// Set arguments to kernel
		res = m_kernel_convolute.setArg(0, *m_data);
		res = m_kernel_convolute.setArg(1, *m_data_aux);
		res = m_kernel_convolute.setArg(2, convCL);

		res = m_queue.enqueueNDRangeKernel(m_kernel_convolute, cl::NullRange, m_globalRange, cl::NullRange);

		res = m_queue.enqueueReadImage(*m_data_aux, CL_TRUE, m_origin, m_region, 0, 0, &m_values.front());

		m_queue.finish();

		for (int i = 0, row = 0; row < img.height(); ++row, i += img.bytesPerLine()) {
			memcpy(img.scanLine(row), &m_values[i], img.bytesPerLine());
		}
	}
	catch (const cl::Error & err)
	{
		std::cerr << "GaussianBlur: " << err.what() << std::endl;
	}
}

void CLWrapper::Sharpening(QImage & img)
{
	if (false == m_CLready)
		return;

	GaussianBlur(img);

	cl_int res;

	// first is filter size
	std::vector<float> laplacian_kernel(10, -1.0f);
	laplacian_kernel[0] = 3.f;
	laplacian_kernel[5] = 8.f;

	try {
		cl::Buffer convCL = cl::Buffer(m_contextCL, CL_MEM_READ_ONLY, laplacian_kernel.size() * sizeof(float), 0, 0);
		m_queue.enqueueWriteBuffer(convCL, CL_TRUE, 0, laplacian_kernel.size() * sizeof(float), &laplacian_kernel[0], 0, NULL);

		// Set arguments to kernel
		res = m_kernel_sharpening.setArg(0, *m_data);
		res = m_kernel_sharpening.setArg(1, *m_data_aux);
		res = m_kernel_sharpening.setArg(2, convCL);
	
		res = m_queue.enqueueNDRangeKernel(m_kernel_sharpening, cl::NullRange,m_globalRange, cl::NullRange);

		res = m_queue.enqueueReadImage(*m_data_aux, CL_TRUE, m_origin, m_region, 0, 0, &m_values.front());

		m_queue.finish();

		for (int i = 0, row = 0; row < img.height(); ++row, i += img.bytesPerLine()) {
			memcpy(img.scanLine(row), &m_values[i], img.bytesPerLine());
		}
	}
	catch (const cl::Error & err)
	{
		std::cerr << "Sharpening: " << err.what() << std::endl;
	}
}

void CLWrapper::ColorSmoothing(QImage & img)
{
	if (false == m_CLready)
		return;

	cl_int res;

	// first is filter size
	std::vector<float> conv(26, 1.f);
	conv[0] = 5.f;

	try {
		cl::Buffer convCL = cl::Buffer(m_contextCL, CL_MEM_READ_ONLY, conv.size() * sizeof(float), 0, 0);
		m_queue.enqueueWriteBuffer(convCL, CL_TRUE, 0, conv.size() * sizeof(float), &conv[0], 0, NULL);

		// Set arguments to kernel
		res = m_kernel_color_smoothing.setArg(0, *m_data);
		res = m_kernel_color_smoothing.setArg(1, *m_data_aux);
		res = m_kernel_color_smoothing.setArg(2, convCL);

		res = m_queue.enqueueNDRangeKernel(m_kernel_color_smoothing, cl::NullRange, m_globalRange, cl::NullRange);

		res = m_queue.enqueueReadImage(*m_data_aux, CL_TRUE, m_origin, m_region, 0, 0, &m_values.front());

		m_queue.finish();

		for (int i = 0, row = 0; row < img.height(); ++row, i += img.bytesPerLine()) {
			memcpy(img.scanLine(row), &m_values[i], img.bytesPerLine());
		}
	}
	catch (const cl::Error & err)
	{
		std::cerr << "ColorSmoothing: " << err.what() << std::endl;
	}
}

void CLWrapper::KMeans(QImage & img, const int centroid_count)
{
	if (false == m_CLready)
		return;

	cl_int res;

	std::vector<Centroid> centroids;

	GenerateCentroids(centroid_count, centroids);

	try {
		cl::Buffer centroidsCL = cl::Buffer(m_contextCL, CL_MEM_READ_ONLY, centroids.size() * sizeof(Centroid), 0, 0);
		m_queue.enqueueWriteBuffer(centroidsCL, CL_TRUE, 0, centroids.size() * sizeof(Centroid), &centroids[0], 0, NULL);

		// Set arguments to kmeans kernel
		res = m_kernel_kmeans.setArg(0, *m_data);
		res = m_kernel_kmeans.setArg(1, centroidsCL);
		res = m_kernel_kmeans.setArg(2, centroid_count);

		// Set arguments to update centroids kernel
		res = m_kernel_update_centroids.setArg(0, centroidsCL);
		res = m_kernel_update_centroids.setArg(1, centroid_count);

		for (int i = 0; i < 5; ++i) {
			// Run the kmeans kernel
			res = m_queue.enqueueNDRangeKernel(m_kernel_kmeans, cl::NullRange, m_globalRange, cl::NullRange);
			//m_queue.finish();

			// Run the update centroid kernel
			res = m_queue.enqueueNDRangeKernel(m_kernel_update_centroids, cl::NullRange, cl::NDRange(centroid_count, 1), cl::NullRange);

			/* Prints centroid values
			m_queue.enqueueReadBuffer(centroidsCL, CL_TRUE, 0, centroids.size() * sizeof(Centroid), &centroids[0], 0, NULL);
			for (auto c : centroids) {
				std::cout << c.x << " " << c.y << " " << c.z << "\t\t";
			}
			std::cout << std::endl;
			*/
		}

		// Run a last iteration of K-Means that will also change de colors
		res = m_kernel_kmeans_draw.setArg(0, *m_data);
		res = m_kernel_kmeans_draw.setArg(1, *m_data_aux);
		res = m_kernel_kmeans_draw.setArg(2, centroidsCL);
		res = m_kernel_kmeans_draw.setArg(3, centroid_count);

		res = m_queue.enqueueNDRangeKernel(m_kernel_kmeans_draw, cl::NullRange, m_globalRange, cl::NullRange);

		res = m_queue.enqueueReadImage(*m_data_aux, CL_TRUE, m_origin, m_region, 0, 0, &m_values.front());

		m_queue.finish();

		for (int i = 0, row = 0; row < img.height(); ++row, i += img.bytesPerLine()) {
			memcpy(img.scanLine(row), &m_values[i], img.bytesPerLine());
		}
	}
	catch (const cl::Error & err)
	{
		std::cerr << "KMeans: " << err.what() << std::endl;
	}
}

void CLWrapper::CMeans(QImage & img, const int centroid_count)
{
	if (false == m_CLready)
		return;

	cl_int res;

	std::vector<Centroid> centroids;

	GenerateCentroids(centroid_count, centroids);

	try {
		cl::Buffer centroidsCL = cl::Buffer(m_contextCL, CL_MEM_READ_ONLY, centroids.size() * sizeof(Centroid), 0, 0);
		m_queue.enqueueWriteBuffer(centroidsCL, CL_TRUE, 0, centroids.size() * sizeof(Centroid), &centroids[0], 0, NULL);

		// Set arguments to kmeans kernel
		res = m_kernel_kmeans.setArg(0, *m_data);
		res = m_kernel_kmeans.setArg(1, centroidsCL);
		res = m_kernel_kmeans.setArg(2, centroid_count);

		// Set arguments to update centroids kernel
		res = m_kernel_update_centroids.setArg(0, centroidsCL);
		res = m_kernel_update_centroids.setArg(1, centroid_count);

		for (int i = 0; i < 5; ++i) {
			// Run the kmeans kernel
			res = m_queue.enqueueNDRangeKernel(m_kernel_kmeans, cl::NullRange, m_globalRange, cl::NullRange);
			m_queue.finish();

			// Run the update centroid kernel
			res = m_queue.enqueueNDRangeKernel(m_kernel_update_centroids, cl::NullRange, cl::NDRange(centroid_count, 1), cl::NullRange);

			/* Prints centroid values
			m_queue.enqueueReadBuffer(centroidsCL, CL_TRUE, 0, centroids.size() * sizeof(Centroid), &centroids[0], 0, NULL);
			for (auto c : centroids) {
			std::cout << c.x << " " << c.y << " " << c.z << "\t\t";
			}
			std::cout << std::endl;
			*/
		}

		// Run a last iteration of K-Means that will also change de colors
		res = m_kernel_kmeans_draw.setArg(0, *m_data);
		res = m_kernel_kmeans_draw.setArg(1, *m_data_aux);
		res = m_kernel_kmeans_draw.setArg(2, centroidsCL);
		res = m_kernel_kmeans_draw.setArg(3, centroid_count);

		res = m_queue.enqueueNDRangeKernel(m_kernel_kmeans_draw, cl::NullRange, m_globalRange, cl::NullRange);

		res = m_queue.enqueueReadImage(*m_data_aux, CL_TRUE, m_origin, m_region, 0, 0, &m_values.front());

		m_queue.finish();

		for (int i = 0, row = 0; row < img.height(); ++row, i += img.bytesPerLine()) {
			memcpy(img.scanLine(row), &m_values[i], img.bytesPerLine());
		}
	}
	catch (const cl::Error & err)
	{
		std::cerr << "CMeans: " << err.what() << std::endl;
	}
}

void CLWrapper::SOMSegmentation(QImage & img, QImage * ground_truth)
{ // TODO: implemente Validity Measure & Davies-Bouldin Index
	if (false == m_CLready)
		return;

	bool got_gt = false;
	std::pair<float, float> gt_values;
	if (nullptr != ground_truth) {
		got_gt = true;
		gt_values = ComputeVMAndDBIndices(ground_truth);
	}

	cl_int res;

	int neuron_count = 2;
	int epochs = (img.width() * img.height()) / 2; // number of iterations
	int x = 0, y = 0; // index of the input value -- will be chosen randomly
	const double ct_learning_rate = 0.1;
	const double time_constant = epochs / log(neuron_count);

	// Node initialization: 1D topology
	std::vector<Neuron> neurons;
	std::vector<float> distances(neuron_count);

	GenerateNeurons(neuron_count, neurons);

	try {
		cl::Buffer neuronsCL = cl::Buffer(m_contextCL, CL_MEM_READ_ONLY, neurons.size() * sizeof(Neuron), 0, 0);
		m_queue.enqueueWriteBuffer(neuronsCL, CL_TRUE, 0, neurons.size() * sizeof(Neuron), &neurons[0], 0, NULL);

		cl::Buffer distancesCL = cl::Buffer(m_contextCL, CL_MEM_READ_ONLY, distances.size() * sizeof(float), 0, 0);
		m_queue.enqueueWriteBuffer(distancesCL, CL_TRUE, 0, distances.size() * sizeof(float), &distances[0], 0, NULL);

		int bmu_idx = 0;
		cl::Buffer bmu_idxCL = cl::Buffer(m_contextCL, CL_MEM_READ_ONLY, sizeof(int), 0, 0);
		m_queue.enqueueWriteBuffer(bmu_idxCL, CL_TRUE, 0, sizeof(int), &bmu_idx, 0, NULL);

		// Set arguments that dont change on host
		res = m_kernel_find_bmu.setArg(1, neuronsCL);
		res = m_kernel_find_bmu.setArg(2, distancesCL);
		res = m_kernel_find_bmu.setArg(3, bmu_idxCL);
		res = m_kernel_find_bmu.setArg(4, neuron_count);

		// Set update weights arguments
		res = m_kernel_update_weights.setArg(1, bmu_idxCL);
		res = m_kernel_update_weights.setArg(2, neuronsCL);
		res = m_kernel_update_weights.setArg(3, neuron_count);

		for (int epoch = 0; epoch < epochs; ++epoch) {
			// Chose a random input value
			x = std::rand() % img.width();
			y = std::rand() % img.height();
			
			QRgb color = img.pixel(x, y);
			cl_float3 value = { qRed(color) / 256.f, qBlue(color) / 256.f, qGreen(color) / 256.f };

			// Set value argument
			res = m_kernel_find_bmu.setArg(0, value);
			
			res = m_queue.enqueueNDRangeKernel(m_kernel_find_bmu, cl::NullRange, cl::NDRange(neuron_count, 1), cl::NullRange);
			//m_queue.finish();
			//m_queue.enqueueReadBuffer(bmu_idxCL, CL_TRUE, 0, sizeof(int), &bmu_idx, 0, NULL);

			//std::cout << bmu_idx << std::endl;

			// update weights
			float neigh_dist = (neurons.size() - 1) * exp(-static_cast<double>(epoch) / time_constant);
			float learning_rate = ct_learning_rate * exp(-static_cast<double>(epoch) / epochs);

			res = m_kernel_update_weights.setArg(0, value);
			res = m_kernel_update_weights.setArg(4, neigh_dist);
			res = m_kernel_update_weights.setArg(5, learning_rate);

			res = m_queue.enqueueNDRangeKernel(m_kernel_update_weights, cl::NullRange, cl::NDRange(neuron_count, 1), cl::NullRange);
		}

		CheckSegmentationNeurons(m_data, neuronsCL, neurons);

		// Do segmentation
		res = m_kernel_som_draw.setArg(0, *m_data);
		res = m_kernel_som_draw.setArg(1, *m_data_aux);
		res = m_kernel_som_draw.setArg(2, neuronsCL);
		res = m_kernel_som_draw.setArg(3, neuron_count);

		res = m_queue.enqueueNDRangeKernel(m_kernel_som_draw, cl::NullRange, m_globalRange, cl::NullRange);

		res = m_queue.enqueueReadImage(*m_data_aux, CL_TRUE, m_origin, m_region, 0, 0, &m_values.front());

		m_queue.finish();

		for (int i = 0, row = 0; row < img.height(); ++row, i += img.bytesPerLine()) {
			memcpy(img.scanLine(row), &m_values[i], img.bytesPerLine());
		}
	}
	catch (const cl::Error & err)
	{
		std::cerr << "SOMSegmentation: " << err.what() << std::endl;
	}
}

void CLWrapper::Threshold(QImage & img, const float value)
{
	if (false == m_CLready)
		return;
	
	cl_int res;

	try {
		// Transform to grayscale
		res = m_kernel_grayscale.setArg(0, *m_data);
		res = m_kernel_grayscale.setArg(1, *m_data_aux);

		res = m_queue.enqueueNDRangeKernel(m_kernel_grayscale, cl::NullRange, m_globalRange, cl::NullRange);

		cl::ImageFormat imf = cl::ImageFormat(CL_RGBA, CL_UNORM_INT8);
		cl::Image2D aux = cl::Image2D(m_contextCL, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, imf, m_width, m_height);

		// Set arguments to kernel
		res = m_kernel_threshold.setArg(0, *m_data_aux);
		res = m_kernel_threshold.setArg(1, aux);
		res = m_kernel_threshold.setArg(2, value);

		res = m_queue.enqueueNDRangeKernel(m_kernel_threshold, cl::NullRange, m_globalRange, cl::NullRange);

		res = m_queue.enqueueReadImage(aux, CL_TRUE, m_origin, m_region, 0, 0, &m_values.front());

		m_queue.finish();

		for (int i = 0, row = 0; row < img.height(); ++row, i += img.bytesPerLine()) {
			memcpy(img.scanLine(row), &m_values[i], img.bytesPerLine());
		}

		//img.convertToFormat(QImage::Format_Grayscale8);
	}
	catch (const cl::Error & err)
	{
		std::cerr << "Threshold: " << err.what() << std::endl;
	}
}

void CLWrapper::InitializeCL()
{
	// Used for exit codes
	cl_int res;

	// Get available platforms
	std::vector<cl::Platform> platforms;
	std::vector<cl::Device> devices;
	cl::Platform::get(&platforms);
	// Select the default platform and create a context using this platform and the GPU
	cl_context_properties cps[] = {
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)(platforms[0])(),
		0
	};

	try {
		m_contextCL = cl::Context(CL_DEVICE_TYPE_GPU, cps);
		// Get a list of devices on this platform
		devices = m_contextCL.getInfo<CL_CONTEXT_DEVICES>();
		// Create a command queue and use the first device
		m_queue = cl::CommandQueue(m_contextCL, devices[0]);

		LoadProgram(m_program_utils, devices, "utils.cl");
		LoadProgram(m_program_filters, devices, "filters.cl");
		LoadProgram(m_program_segmentation, devices, "segmentation.cl");
		LoadProgram(m_program_som, devices, "som.cl");

		// Make kernels
		m_kernel_grayscale = cl::Kernel(m_program_utils, "grayscale");
		m_kernel_convolute = cl::Kernel(m_program_filters, "convolute");
		m_kernel_color_smoothing = cl::Kernel(m_program_filters, "color_smooth_filter");
		m_kernel_sharpening = cl::Kernel(m_program_filters, "sharpness_filter");
		m_kernel_kmeans = cl::Kernel(m_program_segmentation, "kmeans");
		m_kernel_kmeans_draw = cl::Kernel(m_program_segmentation, "kmeans_draw");
		m_kernel_update_centroids = cl::Kernel(m_program_segmentation, "update_centroids");
		m_kernel_threshold = cl::Kernel(m_program_segmentation, "threshold");
		m_kernel_find_bmu = cl::Kernel(m_program_som, "find_bmu");
		m_kernel_update_weights = cl::Kernel(m_program_som, "update_weights");
		m_kernel_som_draw = cl::Kernel(m_program_som, "som_draw");

		m_CLready = true;
	}
	catch (const cl::Error & err)
	{
		std::cerr << "InitializeCL: " << err.what() << std::endl;
	}
}

void CLWrapper::GenerateCentroids(const uint32_t count, std::vector<Centroid> & centroids)
{
	centroids.resize(count);
	for (Centroid & centroid : centroids) {
		centroid = {};

		centroid.x = static_cast <float> (std::rand()) / static_cast <float> (RAND_MAX);
		centroid.y = static_cast <float> (std::rand()) / static_cast <float> (RAND_MAX);
		centroid.z = static_cast <float> (std::rand()) / static_cast <float> (RAND_MAX);
	}
}

void CLWrapper::GenerateNeurons(const uint32_t count, std::vector<Neuron>& neurons)
{
	neurons.resize(count);
	for (Neuron & neuron : neurons) {
		neuron = {};

		neuron.x = static_cast <float> (std::rand()) / static_cast <float> (RAND_MAX);
		neuron.y = static_cast <float> (std::rand()) / static_cast <float> (RAND_MAX);
		neuron.z = static_cast <float> (std::rand()) / static_cast <float> (RAND_MAX);
	}
}

void CLWrapper::CheckSegmentationNeurons(cl::Image2D * gpu_data, cl::Buffer & neuronsCL, std::vector<Neuron> & neurons)
{
	cl_int res;
	res = m_queue.enqueueReadImage(*m_data, CL_TRUE, m_origin, m_region, 0, 0, &m_values.front());
	res = m_queue.enqueueReadBuffer(neuronsCL, CL_TRUE, 0, neurons.size() * sizeof(Neuron), &neurons[0], 0, NULL);
	m_queue.finish();

	float VM = ValidityMeasure(m_values, neurons);
	float DBI = DaviesBouldinIndex(m_values, neurons);

	m_log("[SOM segmentation]: VM = " + std::to_string(VM) + ", DBI = " + std::to_string(DBI));
}

float CLWrapper::GaussianFunction(int niu, int thetha, int cluster_count) const
{
	static const float double_pi = 2 * 3.14159f;
	float pow_thetha = thetha * thetha;
	float pow_k_niu = (cluster_count - niu) * (cluster_count - niu);

	return expf(-(pow_k_niu / (2*pow_thetha))) / sqrtf(double_pi * pow_thetha);
}

float CLWrapper::NormalizedEuclideanDistance(const Neuron & n1, const Neuron & n2) const
{
	float x = n1.x - n2.x;
	float y = n1.y - n2.y;
	float z = n1.z - n2.z;

	return sqrtf(x*x + y*y + z*z);
}

float CLWrapper::ValidityMeasure(const std::vector<uchar>& data, const std::vector<Neuron>& neurons) const
{
	static const int c = 15; // can be between 15 and 25
	float intra_distance = 0.f;

	for (int i = 0; i < data.size(); i += 4) {
		float min_dist = FLT_MAX;
		size_t c_idx = -1;

		Neuron crt_pixel = {
			static_cast<int>(data[i]) / 256.f,
			static_cast<int>(data[i + 1]) / 256.f,
			static_cast<int>(data[i + 2]) / 256.f 
		};

		for (int n_idx = 0; n_idx < neurons.size(); ++n_idx) {
			
			float dist = NormalizedEuclideanDistance(crt_pixel, neurons[n_idx]);
			
			dist *= dist;

			if (dist < min_dist) {
				min_dist = dist;
				c_idx = n_idx;
			}
		}

		intra_distance += min_dist;
	}

	intra_distance /= (data.size() / 4.f);

	float inter_distance = FLT_MAX;
	for (int i = 0; i < neurons.size(); ++i) {
		for (int j = 0; j < neurons.size(); ++j) {
			if (i == j) {
				continue;
			}

			float dist = NormalizedEuclideanDistance(neurons[i], neurons[j]);

			dist *= dist;
			
			if (dist < inter_distance) {
				inter_distance = dist;
			}
		}
	}

	float y = c * GaussianFunction(2, 1, neurons.size()) + 1;

	return y * (intra_distance / inter_distance);
}

float CLWrapper::DaviesBouldinIndex(const std::vector<uchar>& data, const std::vector<Neuron>& neurons) const
{
	std::vector<int> cluster_size(neurons.size(), 0);
	std::vector<float> cluster_distances(neurons.size(), 0.f);

	for (int i = 0; i < data.size(); i += 4) {
		float min_dist = FLT_MAX;
		size_t c_idx = -1;

		Neuron crt_pixel = {
			static_cast<int>(data[i]) / 256.f,
			static_cast<int>(data[i + 1]) / 256.f,
			static_cast<int>(data[i + 2]) / 256.f
		};

		for (int n_idx = 0; n_idx < neurons.size(); ++n_idx) {

			float dist = NormalizedEuclideanDistance(crt_pixel, neurons[n_idx]);

			if (dist < min_dist) {
				min_dist = dist;
				c_idx = n_idx;
			}
		}

		++cluster_size[c_idx];

		cluster_distances[c_idx] += min_dist;
	}

	// returns a safety error, doing it by hand
	//std::transform(cluster_distances.begin(), cluster_distances.end(), cluster_size.begin(), cluster_distances, std::divides<float>());
	for (int i = 0; i < cluster_distances.size(); ++i) {
		cluster_distances[i] /= static_cast<float>(cluster_size[i]);
	}

	std::vector<float> D(neurons.size(), 0.f);
	for (int i = 0; i < neurons.size(); ++i) {
		for (int j = 0; j < neurons.size(); ++j) {
			if (i == j) {
				continue;
			}

			float dist = NormalizedEuclideanDistance(neurons[i], neurons[j]);
			float Rij = (cluster_distances[i] + cluster_distances[j]) / dist;


			if (Rij > D[i]) {
				D[i] = Rij;
			}
		}
	}

	return std::accumulate(D.begin(), D.end(), 0.f) / D.size();
}

std::pair<float, float> CLWrapper::ComputeVMAndDBIndices(QImage * img) const
{
	std::vector<uchar> values(img->byteCount());

	for (int i = 0, row = 0; row < img->height(); ++row, i += img->bytesPerLine()) {
		memcpy(img->scanLine(row), &values[i], img->bytesPerLine());
	}

	std::set<QRgb> uq_neuron;
	for (int i = 0; i < img->height(); ++i) {
		for (int j = 0; j < img->width(); ++j) {
			QRgb px = img->pixel(QPoint(i, j));
			uq_neuron.insert(px);
		}
	}

	std::vector<Neuron> neurons;
	for (auto n : uq_neuron) {
		Neuron nv;
		nv.x = qRed(n);
		nv.y = qGreen(n);
		nv.z = qBlue(n);

		neurons.push_back(nv);
	}

	float VM = ValidityMeasure(values, neurons);
	float DBI = DaviesBouldinIndex(values, neurons);

	return std::pair<float, float>();
}
