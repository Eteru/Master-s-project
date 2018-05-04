#include "Benchmark.h"

#include "SequentialImplementation.h"
#include "ParallelImplementation.h"

#include <iostream>

Benchmark::Benchmark()
{
}

Benchmark::~Benchmark()
{
}

std::string Benchmark::RunTests(GPGPUImplementation & gpgpu, QImage & img)
{
	using Matrix = std::vector<std::vector<float>>;

	size_t iterations_no = 10;

	ParallelImplementation pi;
	SequentialImplementation si;

	std::vector<std::string> names = { "Sequential", "Naive Parallelism", "GPGPU" };
	std::vector<Implementation *> impls = { &si, &pi, &gpgpu };
	std::vector<std::string> images = { "knife_128.jpg", "knife_256.jpg" , "knife_512.jpg" , "knife_1024.jpg" , "knife_2048.jpg" };

	size_t targets = impls.size();
	Matrix grayscale(targets);
	Matrix gaussian(targets);
	Matrix kmeans(targets);
	Matrix som(targets);

	for (int i = 0; i < targets; ++i)
	{
		grayscale[i].resize(images.size(), 0.f);
		gaussian[i].resize(images.size(), 0.f);
		kmeans[i].resize(images.size(), 0.f);
		som[i].resize(images.size(), 0.f);
	}

	std::string grayscale_output = "Grayscale, Sequential, Naive Parallelism, GPGPU\n";
	std::string gaussian_output = "Gaussian, Sequential, Naive Parallelism, GPGPU\n";
	std::string kmeans_output = "KMeans, Sequential, Naive Parallelism, GPGPU\n";
	std::string som_output = "SOM, Sequential, Naive Parallelism, GPGPU\n";

	for (int i = 0; i < images.size(); ++i)
	{
		QImage crt_img;
		bool b = crt_img.load(QString::fromStdString("D:\\workspace\\git\\Master\-s\-project\\dataset\\" + images[i]));

		gpgpu.SetData(crt_img);

		for (size_t target = 0; target < targets; ++target)
		{
			for (size_t iter = 0; iter < iterations_no; ++iter)
			{
				grayscale[target][i] += impls[target]->Grayscale(crt_img.copy());
				gaussian[target][i] += impls[target]->GaussianBlur(crt_img.copy());
				kmeans[target][i] += impls[target]->KMeans(img.copy(), 3);
				som[target][i] += impls[target]->SOMSegmentation(img.copy());
			}

			grayscale[target][i] /= (iterations_no * 1000000.f);
			gaussian[target][i] /= (iterations_no * 1000000.f);
			kmeans[target][i] /= (iterations_no * 1000000.f);
			som[target][i] /= (iterations_no * 1000000.f);
		}

		//output += names[target] + "," + std::to_string(grayscale[target]) + ", " + std::to_string(gaussian[target]) + ", " + std::to_string(kmeans[target]) + ", " + std::to_string(som[target]) + "\n";
	}

	std::vector<int> sizes = { 128, 256, 512, 1024, 2048 };
	for (int i = 0; i < sizes.size(); ++i)
	{
		grayscale_output += std::to_string(sizes[i]);
		gaussian_output += std::to_string(sizes[i]);
		kmeans_output += std::to_string(sizes[i]);
		som_output += std::to_string(sizes[i]);

		for (size_t target = 0; target < targets; ++target)
		{
			grayscale_output += ", " + std::to_string(grayscale[target][i]);
			gaussian_output += ", " + std::to_string(gaussian[target][i]);
			kmeans_output += ", " + std::to_string(kmeans[target][i]);
			som_output += ", " + std::to_string(som[target][i]);
		}

		grayscale_output += "\n";
		gaussian_output += "\n";
		kmeans_output += "\n";
		som_output += "\n";
	}

	//for (size_t target = 0; target < targets; ++target)
	//{
	//	grayscale_output += names[target];
	//	gaussian_output += names[target];
	//	kmeans_output += names[target];
	//	som_output += names[target];
	//	for (int i = 0; i < images.size(); ++i)
	//	{
	//		grayscale_output += ", " + std::to_string(grayscale[target][i]);
	//		gaussian_output += ", " + std::to_string(gaussian[target][i]);
	//		kmeans_output += ", " + std::to_string(gaussian[target][i]);
	//		som_output += ", " + std::to_string(gaussian[target][i]);
	//	}
	//
	//	grayscale_output += "\n";
	//	gaussian_output += "\n";
	//	kmeans_output += "\n";
	//	som_output += "\n";
	//
	//}

	return grayscale_output + gaussian_output + kmeans_output + som_output;
}
