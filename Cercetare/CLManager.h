#pragma once

#define __CL_ENABLE_EXCEPTIONS

#include <map>
#include <CL/cl.hpp>

#include "Constants.h"

class CLManager
{
public:
	~CLManager();

	static CLManager * GetInstance();

	bool Init();

	cl::Context * GetContext();
	cl::CommandQueue * GetQueue();
	cl::Kernel * GetKernel(const std::string kern);

private:
	CLManager() {}

	static CLManager *m_instance;
	cl::Context m_contextCL;
	cl::CommandQueue m_queue;
	std::map<std::string, cl::Kernel> m_kernels;

	void PrintDevices(const std::vector<cl::Device> & devices);
	void LoadProgram(cl::Program & program, std::vector<cl::Device> & devices, std::string file);
};

