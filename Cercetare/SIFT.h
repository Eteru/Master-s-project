#pragma once
#define __CL_ENABLE_EXCEPTIONS

#include <map>
#include <vector>

#include "Octave.h"

class SIFT
{
public:
	SIFT();
	virtual ~SIFT();

	cl::Image2D * Run(cl::Image2D * image, uint32_t w, uint32_t h);

private:
	static const uint32_t NUMBER_OF_BLURS = 5;
	static const uint32_t NUMBER_OF_OCTAVES = 4;
	std::map<cl::Image2D *, std::vector<double>> m_fvps;

	cl::Image2D * SetupReferenceImage(cl::Image2D * image, uint32_t w, uint32_t h);
	cl::Image2D * SetupReferenceImageOld(cl::Image2D * image, uint32_t w, uint32_t h);

	void WriteOctaveImagesOnDisk(Octave & o, uint32_t o_idx) const;
	void WriteImageOnDisk(cl::Image2D *img, uint32_t w, uint32_t h, std::string name) const;
};

