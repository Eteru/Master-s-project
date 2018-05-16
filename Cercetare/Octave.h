#pragma once
#define __CL_ENABLE_EXCEPTIONS

#include <vector>
#include <CL/cl.hpp>

class Octave
{
public:
	Octave();
	Octave(cl::Image2D * image, float sigma, uint32_t w, uint32_t h, uint32_t size);
	Octave(Octave & octave, uint32_t w, uint32_t h);
	~Octave();


	uint32_t GetWidth() const;
	uint32_t GetHeight() const;
	size_t GetScaleSpaceSize();
	float GetMiddleSigma() const;
	
	cl::Image2D *GetDefaultImage();
	cl::Image2D *GetImage(uint32_t idx);
	cl::Image2D *GetLastImage();

	std::vector<cl::Image2D *> & GetImages();
	std::vector<cl::Image2D *> & GetDoGs();
	std::vector<cl::Image2D *> & GetFeatures();

	void DoG();
	void ComputeLocalMaxima();

private:
	static const uint32_t BLUR_KERNEL_SIZE = 7;
	static float SIGMA_INCREMENT;
	uint32_t m_width;
	uint32_t m_height;
	float m_starting_sigma;
	cl::Context m_context;
	cl::CommandQueue m_queue;
	cl::NDRange m_range;

	cl::Image2D *m_default_image;
	std::vector<cl::Image2D *> m_images;
	std::vector<cl::Image2D *> m_DoGs;
	std::vector<cl::Image2D *> m_points;

	void Blur();
	float Gaussian(const int x, const int y, const float sigma);
	std::vector<float> Octave::GaussianKernel(const uint32_t kernel_size, const float sigma);
};

