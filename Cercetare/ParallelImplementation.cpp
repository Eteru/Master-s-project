#include "ParallelImplementation.h"

#include <omp.h> // version 2.0
#include <chrono>
#include <iostream>


ParallelImplementation::ParallelImplementation()
{
}

ParallelImplementation::~ParallelImplementation()
{
}

void ParallelImplementation::CustomFilter(QImage & img, const std::vector<float>& kernel_values)
{
}

float ParallelImplementation::Grayscale(QImage & img)
{
	std::vector<uchar> values;
	CopyImageToBuffer(img, values);
	size_t sz = img.width() * img.height();

	std::vector<uchar> values_out(sz * 4);
	auto start = std::chrono::system_clock::now();
	
	#pragma omp parallel for shared(img) schedule(static)
	for (int i = 0; i < sz; i+=4)
	{
		values_out[i] = static_cast<uchar>(values[i] * 0.21);
		values_out[i + 1] = static_cast<uchar>(values[i + 1] * 0.72);
		values_out[i + 2] = static_cast<uchar>(values[i + 2] * 0.07);
	}

	auto end = std::chrono::system_clock::now();

	return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

void ParallelImplementation::Sobel(QImage & img)
{
}

float ParallelImplementation::GaussianBlur(QImage & img)
{
	std::vector<float> gaussian_kernel =
	{
		0.102059f, 0.115349f, 0.102059f,
		0.115349f, 0.130371f, 0.115349f,
		0.102059f, 0.115349f, 0.102059f
	};

	int HALF_FILTER_SIZE = 1;

	std::vector<uchar> values;
	CopyImageToBuffer(img, values);
	size_t sz = img.width() * img.height();

	std::vector<uchar> values_out(sz * 4);

	auto start = std::chrono::system_clock::now();

	#pragma omp parallel for shared(img) schedule(static)
	for (int i = 0; i < img.height(); ++i)
	{
		for (int j = 0; j < img.width(); ++j)
		{
			size_t filter_index = 0;
			uchar R = 0, G = 0, B = 0;
			for (int r = -HALF_FILTER_SIZE; r <= HALF_FILTER_SIZE; r++)
			{
				for (int c = -HALF_FILTER_SIZE; c <= HALF_FILTER_SIZE; c++)
				{
					if (i + r < 0 || i + r >= img.height() || j + c < 0 || j + c >= img.width())
					{
						continue;
					}

					size_t index = ((i + r) * img.height() + j + c) * 4;

					R += gaussian_kernel[filter_index] * values[index];
					G += gaussian_kernel[filter_index] * values[index + 1];
					B += gaussian_kernel[filter_index] * values[index + 2];

					++filter_index;
				}
			}

			size_t index = (i * img.height() + j) * 4;

			values_out[index] = R;
			values_out[index + 1] = G;
			values_out[index + 2] = B;
		}
	}

	auto end = std::chrono::system_clock::now();

	return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

void ParallelImplementation::Sharpening(QImage & img)
{
}

void ParallelImplementation::ColorSmoothing(QImage & img)
{
}

float ParallelImplementation::KMeans(QImage & img, const int centroid_count)
{
	std::vector<uchar> values_ui;

	CopyImageToBuffer(img, values_ui);
	std::vector<float> values;

	for (size_t i = 0; i < values_ui.size(); ++i)
	{
		values.push_back(values_ui[i] / 255.f);
	}

	// generate centroids
	std::vector<Centroid> centroids;

	GenerateCentroids(centroid_count, centroids);

	int max_iterations = 5;
	
	auto start = std::chrono::system_clock::now();

	for (int iter = 0; iter < max_iterations; ++iter)
	{
		#pragma omp parallel for shared(centroids) schedule(static)
		for (int i = 0; i < values.size(); i += 4)
		{
			float dist = FLT_MAX;
			size_t centroid_idx = -1;

			for (size_t c = 0; c < centroids.size(); ++c)
			{
				float d = Distance(centroids[c], values[i], values[i + 1], values[i + 2]);

				if (d < dist)
				{
					dist = d;
					centroid_idx = c;
				}
			}

			#pragma omp critical
			{
				centroids[centroid_idx].sum_x += values[i];
				centroids[centroid_idx].sum_y += values[i + 1];
				centroids[centroid_idx].sum_z += values[i + 2];
				++centroids[centroid_idx].count;
			}
		}

		// Update centroids
		#pragma omp barrier
		#pragma omp parallel for shared(centroids) schedule(static)
		for (int c = 0; c < centroids.size(); ++c)
		{
			if (0 != centroids[c].count)
			{
				centroids[c].value_x = centroids[c].sum_x / centroids[c].count;
				centroids[c].value_y = centroids[c].sum_y / centroids[c].count;
				centroids[c].value_z = centroids[c].sum_z / centroids[c].count;
			}

			centroids[c].sum_x = 0;
			centroids[c].sum_y = 0;
			centroids[c].sum_z = 0;
			centroids[c].count = 0;
		}

		#pragma omp barrier
	}

	// "Draw"
	#pragma omp parallel for shared(centroids) schedule(static)
	for (int i = 0; i < values.size(); i += 4)
	{
		float dist = FLT_MAX;
		size_t centroid_idx = -1;

		for (size_t c = 0; c < centroids.size(); ++c)
		{
			float d = Distance(centroids[c], values[i], values[i + 1], values[i + 2]);

			if (d < dist)
			{
				dist = d;
				centroid_idx = c;
			}
		}

		values[i] = centroids[centroid_idx].value_x;
		values[i + 1] = centroids[centroid_idx].value_y;
		values[i + 2] = centroids[centroid_idx].value_z;
	}

	auto end = std::chrono::system_clock::now();

	for (size_t i = 0; i < values_ui.size(); ++i)
	{
		values_ui[i] = static_cast<unsigned>(values[i] * 255.f);
	}

	CopyBufferToImage(values_ui, img);

	return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

float ParallelImplementation::SOMSegmentation(QImage & img, QImage * ground_truth)
{
	// copy data to local vector
	std::vector<uchar> values;

	CopyImageToBuffer(img, values);

	int max_iterations = 1;
	int epochs = 200; // number of iterations
	int neuron_count = 3;
	uint32_t total_sz = img.width() * img.height();
	const double ct_learning_rate = 0.1;
	const double time_constant = epochs / log(neuron_count);

	// generate neurons
	std::vector<Neuron> neurons;
	GenerateNeurons(3, neurons);

	auto start = std::chrono::system_clock::now();

	for (int iter = 0; iter < max_iterations; ++iter)
	{
		for (int epoch = 0; epoch < epochs; ++epoch)
		{
			size_t index = (std::rand() % total_sz) * 4;

			// find BMU
			float dist = FLT_MAX;
			size_t bmu = -1;

			for (int c = 0; c < neurons.size(); ++c)
			{
				float dist_X = values[index] - neurons[c].value_x;
				float dist_Y = values[index + 1] - neurons[c].value_y;
				float dist_Z = values[index + 2] - neurons[c].value_z;

				float d = sqrt((float)(dist_X * dist_X + dist_Y * dist_Y + dist_Z * dist_Z));


				if (d < dist)
				{
					dist = d;
					bmu = c;
				}
			}

			// Update neurons
			float neigh_dist = (neurons.size() - 1) * exp(-static_cast<double>(epoch) / time_constant);
			float learning_rate = ct_learning_rate * exp(-static_cast<double>(epoch) / epochs);

			#pragma omp parallel for shared(neurons) schedule(static)
			for (int c = 0; c < neurons.size(); ++c)
			{
				int n_dist = abs((int)(bmu - c));

				if (n_dist > neigh_dist)
				{
					continue;
				}

				float influence = exp(-(n_dist * n_dist) / (2.f * (neigh_dist * neigh_dist)));

				neurons[c].value_x += learning_rate * influence * (int)(values[index] - neurons[c].value_x);
				neurons[c].value_y += learning_rate * influence * (int)(values[index + 1] - neurons[c].value_y);
				neurons[c].value_z += learning_rate * influence * (int)(values[index + 2] - neurons[c].value_z);
			}
		}
	}

	#pragma omp parallel for shared(neurons) schedule(static)
	for (int index = 0; index < values.size(); index += 4)
	{
		float dist = FLT_MAX;
		size_t bmu = -1;

		for (size_t c = 0; c < neurons.size(); ++c)
		{
			float dist_X = values[index] - neurons[c].value_x;
			float dist_Y = values[index + 1] - neurons[c].value_y;
			float dist_Z = values[index + 2] - neurons[c].value_z;

			float d = sqrt((float)(dist_X * dist_X + dist_Y * dist_Y + dist_Z * dist_Z));


			if (d < dist)
			{
				dist = d;
				bmu = c;
			}
		}

		values[index] = neurons[bmu].value_x;
		values[index + 1] = neurons[bmu].value_y;
		values[index + 2] = neurons[bmu].value_z;
	}

	auto end = std::chrono::system_clock::now();

	CopyBufferToImage(values, img);

	return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

void ParallelImplementation::Threshold(QImage & img, const float value)
{
}

void ParallelImplementation::RunSIFT(QImage & img)
{
}
