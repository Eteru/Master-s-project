#pragma once
#define __CL_ENABLE_EXCEPTIONS

#include <vector>
#include <CL/cl.hpp>

class Octave
{
public:
	Octave(cl::Image2D * image, uint32_t w, uint32_t h, uint32_t size);
	~Octave();

	cl::Image2D *GetImage(uint32_t idx);

	void Blur();
	void DoG();

private:
	uint32_t m_width;
	uint32_t m_height;
	cl::Context m_context;
	cl::CommandQueue m_queue;
	cl::NDRange m_range;

	std::vector<cl::Image2D *> m_images;
};

