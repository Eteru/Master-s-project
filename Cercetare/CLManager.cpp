
#include "CLManager.h"

#include <fstream>
#include <iostream>

CLManager *CLManager::m_instance = nullptr;

CLManager::~CLManager()
{
	if (nullptr != m_instance) {
		delete m_instance;
	}
}

CLManager * CLManager::GetInstance()
{
	if (nullptr == m_instance) {
		m_instance = new CLManager;
	}

	return m_instance;
}

bool CLManager::Init()
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
		m_queue = cl::CommandQueue(m_contextCL, devices[0], CL_QUEUE_PROFILING_ENABLE);

		cl::Program program;
		LoadProgram(program, devices, "Kernels/utils.cl");
		m_kernels[Constants::KERNEL_GRAYSCALE] = cl::Kernel(program, Constants::KERNEL_GRAYSCALE.c_str());
		m_kernels[Constants::KERNEL_RESIZE] = cl::Kernel(program, Constants::KERNEL_RESIZE.c_str());
		m_kernels[Constants::KERNEL_IMAGE_DIFFERENCE] = cl::Kernel(program, Constants::KERNEL_IMAGE_DIFFERENCE.c_str());
		m_kernels[Constants::KERNEL_FIND_EXTREME_POINTS] = cl::Kernel(program, Constants::KERNEL_FIND_EXTREME_POINTS.c_str());
		m_kernels[Constants::KERNEL_INT_TO_FLOAT] = cl::Kernel(program, Constants::KERNEL_INT_TO_FLOAT.c_str());
		m_kernels[Constants::KERNEL_MAGN_AND_ORIEN] = cl::Kernel(program, Constants::KERNEL_MAGN_AND_ORIEN.c_str());
		m_kernels[Constants::KERNEL_MAGN_AND_ORIEN_INTERP] = cl::Kernel(program, Constants::KERNEL_MAGN_AND_ORIEN_INTERP.c_str());
		m_kernels[Constants::KERNEL_GENERATE_FEATURE_POINTS] = cl::Kernel(program, Constants::KERNEL_GENERATE_FEATURE_POINTS.c_str());
		m_kernels[Constants::KERNEL_EXTRACT_FEATURE_POINTS] = cl::Kernel(program, Constants::KERNEL_EXTRACT_FEATURE_POINTS.c_str());

		LoadProgram(program, devices, "Kernels/filters.cl");
		m_kernels[Constants::KERNEL_CONVOLUTE] = cl::Kernel(program, Constants::KERNEL_CONVOLUTE.c_str());
		m_kernels[Constants::KERNEL_COLOR_SMOOTHING] = cl::Kernel(program, Constants::KERNEL_COLOR_SMOOTHING.c_str());
		m_kernels[Constants::KERNEL_SHARPNESS] = cl::Kernel(program, Constants::KERNEL_SHARPNESS.c_str());

		LoadProgram(program, devices, "Kernels/segmentation.cl");
		m_kernels[Constants::KERNEL_KMEANS] = cl::Kernel(program, Constants::KERNEL_KMEANS.c_str());
		m_kernels[Constants::KERNEL_KMEANS_DRAW] = cl::Kernel(program, Constants::KERNEL_KMEANS_DRAW.c_str());
		m_kernels[Constants::KERNEL_KMEANS_UPDATE_CENTROIDS] = cl::Kernel(program, Constants::KERNEL_KMEANS_UPDATE_CENTROIDS.c_str());
		m_kernels[Constants::KERNEL_THRESHOLD] = cl::Kernel(program, Constants::KERNEL_THRESHOLD.c_str());

		LoadProgram(program, devices, "Kernels/som.cl");
		m_kernels[Constants::KERNEL_SOM_FIND_BMU] = cl::Kernel(program, Constants::KERNEL_SOM_FIND_BMU.c_str());
		m_kernels[Constants::KERNEL_SOM_UPDATE_WEIGHTS] = cl::Kernel(program, Constants::KERNEL_SOM_UPDATE_WEIGHTS.c_str());
		m_kernels[Constants::KERNEL_SOM_DRAW] = cl::Kernel(program, Constants::KERNEL_SOM_DRAW.c_str());

		LoadProgram(program, devices, "Kernels/post_processing.cl");
		m_kernels[Constants::KERNEL_POST_NOISE_REDUCTION] = cl::Kernel(program, Constants::KERNEL_POST_NOISE_REDUCTION.c_str());

		return true;
	}
	catch (const cl::Error & err)
	{
		std::cerr << "InitializeCL: " << err.what() << " with error: " << err.err() << std::endl;
		return false;
	}
}

cl::Context * CLManager::GetContext()
{
	return &m_contextCL;
}

cl::CommandQueue * CLManager::GetQueue()
{
	return &m_queue;
}

cl::Kernel * CLManager::GetKernel(const std::string kern)
{
	if (m_kernels.find(kern) != m_kernels.end())
	{
		return &m_kernels[kern];
	}

	// {} cannot be expanded to cl::Kernel, need to find a better fallback
	return &m_kernels[kern];
}

void CLManager::PrintDevices(const std::vector<cl::Device>& devices)
{
	size_t i = 0;
	for (cl::Device d : devices) {
		std::cout << "Device ID: " << i++ << std::endl;
		std::cout << "\tDevice Name: " << d.getInfo<CL_DEVICE_NAME>() << std::endl;
		std::cout << "\tDevice Type: " << d.getInfo<CL_DEVICE_TYPE>();
		std::cout << " (GPU: " << CL_DEVICE_TYPE_GPU << ", CPU: " << CL_DEVICE_TYPE_CPU << ")" << std::endl;
		std::cout << "\tDevice Vendor: " << d.getInfo<CL_DEVICE_VENDOR>() << std::endl;
		std::cout << "\tDevice Max Compute Units: " << d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
		std::cout << "\tDevice Max Clock Frequency: " << d.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << std::endl;
		std::cout << "\tDevice Max Allocateable Memory: " << d.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() << std::endl;
		std::cout << "\tDevice Global Memory: " << d.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() << std::endl;
		std::cout << "\tDevice Local Memory: " << d.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << std::endl;
		std::cout << "\tDevice Constant Memory: " << d.getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>() << std::endl;

		std::cout << "\tDevice Available: " << d.getInfo< CL_DEVICE_AVAILABLE>() << std::endl;
	}
	std::cout << std::endl;
}

void CLManager::LoadProgram(cl::Program & program, std::vector<cl::Device> & devices, std::string file)
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

		program.build(devices, "-cl-std=CL2.0");
	}
	catch (const cl::Error & err)
	{
		std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
	}
}