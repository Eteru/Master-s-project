#pragma once
#define __CL_ENABLE_EXCEPTIONS

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
	static const uint32_t NUMBER_OF_OCTAVES = 5;
};

