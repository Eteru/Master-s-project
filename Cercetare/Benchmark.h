#pragma once

#include <string>
#include <QImage>

#include "GPGPUImplementation.h"

class Benchmark
{
public:
	Benchmark();
	~Benchmark();

	static std::string RunTests(GPGPUImplementation & gpgpu, QImage & img);
};

