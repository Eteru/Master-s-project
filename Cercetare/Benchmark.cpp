#include "Benchmark.h"

#include "SequentialImplementation.h"
#include "ParallelImplementation.h"

Benchmark::Benchmark()
{
}

Benchmark::~Benchmark()
{
}

std::string Benchmark::RunTests(GPGPUImplementation & gpgpu, QImage & img)
{
	size_t iterations_no = 10;

	ParallelImplementation pi;
	SequentialImplementation si;

	std::vector<Implementation *> impls = { &si, &pi, &gpgpu };

	size_t targets = impls.size();
	std::vector<float> grayscale(targets, 0.f);

	for (size_t target = 0; target < targets; ++target)
	{
		for (size_t i = 0; i < iterations_no; ++i)
		{
			grayscale[target] += impls[target]->Grayscale(img.copy());
		}

		grayscale[target] /= static_cast<float>(iterations_no);
	}

	std::string output = ",grayscale,gaussian blur,k-means,som\n";

	for (size_t target = 0; target < targets; ++target)
	{
		output += std::to_string(target) + "," + std::to_string(grayscale[target]) + "\n";
	}

	return output;
}
