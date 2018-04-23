#include "ParallelImplementation.h"

#include <omp.h> // version 2.0
#include <chrono>


ParallelImplementation::ParallelImplementation()
{
}

ParallelImplementation::~ParallelImplementation()
{
}

float ParallelImplementation::Grayscale(QImage & img)
{
	auto start = std::chrono::system_clock::now();

	int j;
	#pragma omp parallel for private(j)
	for (int i = 0; i < img.width(); ++i)
	{
		for (j = 0; j < img.height(); ++j)
		{
			QRgb px = img.pixel(i, j);
			px = 0.21 * qRed(px) + 0.72 * qGreen(px) + 0.07 * qBlue(px);
			px = qRgb(px, px, px);
			img.setPixel(i, j, px);
		}
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

	auto start = std::chrono::system_clock::now();

	#pragma omp parallel for
	for (int i = 0; i < img.width(); ++i)
	{
		for (int j = 0; j < img.height(); ++j)
		{
			size_t filter_index = 0;
			QRgb value = qRgb(0, 0, 0);
			for (int r = -HALF_FILTER_SIZE; r <= HALF_FILTER_SIZE; r++)
			{
				for (int c = -HALF_FILTER_SIZE; c <= HALF_FILTER_SIZE; c++)
				{
					if (i + r < 0 || i + r >= img.width() || j + c < 0 || j + c >= img.height())
					{
						continue;
					}

					QRgb px;
					#pragma omp critical
					{
						 px = img.pixel(i + r, j + c);
					}
					px = qRgb(qRed(px) *gaussian_kernel[filter_index], qGreen(px) *gaussian_kernel[filter_index], qBlue(px) *gaussian_kernel[filter_index]);
					value += px;

					++filter_index;
				}
			}

			#pragma omp critical
			{
				img.setPixel(i, j, value);
			}
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
	// copy data to local vector
	std::vector<uchar> values;

	CopyImageToBuffer(img, values);

	// generate centroids
	std::vector<Centroid> centroids;

	GenerateCentroids(centroid_count, centroids);

	int max_iterations = 1;
	
	auto start = std::chrono::system_clock::now();

	for (int iter = 0; iter < max_iterations; ++iter)
	{
		#pragma omp parallel for
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
				centroids[centroid_idx].sum_x += static_cast<unsigned>(values[i]);
				centroids[centroid_idx].sum_y += static_cast<unsigned>(values[i + 1]);
				centroids[centroid_idx].sum_z += static_cast<unsigned>(values[i + 2]);
				++centroids[centroid_idx].count;
			}
		}

		// Update centroids
		#pragma omp barrier
		#pragma omp parallel for
		for (int c = 0; c < centroids.size(); ++c)
		{
			if (0 != centroids[c].count)
			{
				centroids[c].x = centroids[c].sum_x / centroids[c].count;
				centroids[c].y = centroids[c].sum_y / centroids[c].count;
				centroids[c].z = centroids[c].sum_z / centroids[c].count;
			}

			centroids[c].sum_x = 0;
			centroids[c].sum_y = 0;
			centroids[c].sum_z = 0;
			centroids[c].count = 0;
		}

		#pragma omp barrier
	}

	// "Draw"
	#pragma omp parallel for
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

		values[i] = centroids[centroid_idx].x;
		values[i + 1] = centroids[centroid_idx].y;
		values[i + 2] = centroids[centroid_idx].z;
	}

	auto end = std::chrono::system_clock::now();

	CopyBufferToImage(values, img);

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
				uint dist_X = values[index] - neurons[c].x;
				uint dist_Y = values[index + 1] - neurons[c].y;
				uint dist_Z = values[index + 2] - neurons[c].z;

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

			#pragma omp parallel
			for (int c = 0; c < neurons.size(); ++c)
			{
				int n_dist = abs((int)(bmu - c));

				if (n_dist > neigh_dist)
				{
					continue;
				}

				float influence = exp(-(n_dist * n_dist) / (2.f * (neigh_dist * neigh_dist)));

				neurons[c].x += learning_rate * influence * (int)(values[index] - neurons[c].x);
				neurons[c].y += learning_rate * influence * (int)(values[index + 1] - neurons[c].y);
				neurons[c].z += learning_rate * influence * (int)(values[index + 2] - neurons[c].z);
			}
		}
	}

	#pragma omp parallel
	for (int index = 0; index < values.size(); index += 4)
	{
		float dist = FLT_MAX;
		size_t bmu = -1;

		for (size_t c = 0; c < neurons.size(); ++c)
		{
			uint dist_X = values[index] - neurons[c].x;
			uint dist_Y = values[index + 1] - neurons[c].y;
			uint dist_Z = values[index + 2] - neurons[c].z;

			float d = sqrt((float)(dist_X * dist_X + dist_Y * dist_Y + dist_Z * dist_Z));


			if (d < dist)
			{
				dist = d;
				bmu = c;
			}
		}

		values[index] = neurons[bmu].x;
		values[index + 1] = neurons[bmu].y;
		values[index + 2] = neurons[bmu].z;
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
