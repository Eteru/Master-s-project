#pragma once
#define __CL_ENABLE_EXCEPTIONS

#include <vector>
#include <CL/cl.hpp>

class Octave
{
public:
	Octave();
	Octave(cl::Image2D * image, uint32_t w, uint32_t h, uint32_t size);
	Octave(Octave & octave, uint32_t w, uint32_t h);
	~Octave();


	uint32_t GetWidth() const;
	uint32_t GetHeight() const;
	size_t GetScaleSpaceSize();

	cl::Image2D *GetImage(uint32_t idx);
	cl::Image2D *GetLastImage();

	std::vector<cl::Image2D *> & GetImages();
	std::vector<cl::Image2D *> & GetDoGs();
	std::vector<cl::Image2D *> & GetFeatures();

	void DoG();
	void ComputeLocalMaxima();

private:
	uint32_t m_width;
	uint32_t m_height;
	cl::Context m_context;
	cl::CommandQueue m_queue;
	cl::NDRange m_range;

	std::vector<cl::Image2D *> m_images;
	std::vector<cl::Image2D *> m_DoGs;
	std::vector<cl::Image2D *> m_points;

	void Blur();
};

