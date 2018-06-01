#pragma once
#define __CL_ENABLE_EXCEPTIONS

#include <vector>
#include <CL/cl.hpp>

#include "Structs.h"

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
	std::vector<cl::Image2D *> & GetMagnitudes();
	std::vector<cl::Image2D *> & GetOrientations();

	void DoG();
	void ComputeLocalMaxima();
	std::vector<FeaturePoint> ComputeOrientation();

private:
	static const uint32_t BLUR_KERNEL_SIZE = 7;
	static const uint32_t WEIGHT_KERNEL_SIZE = 16;
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
	std::vector<cl::Image2D *> m_magnitudes;
	std::vector<cl::Image2D *> m_orientations;

	void Blur();
	float Gaussian(const int x, const int y, const float sigma);
	std::vector<float> Octave::GaussianKernel(const uint32_t kernel_size, const float sigma);

	unsigned int GetKernelSize(float sigma, float cut_off = 0.001f);
};

