#include "SequentialImplementation.h"

#include <QColor>
#include <chrono>


SequentialImplementation::SequentialImplementation()
{
}

SequentialImplementation::~SequentialImplementation()
{
}

void SequentialImplementation::CustomFilter(QImage & img, const std::vector<float>& kernel_values)
{
}

float SequentialImplementation::Grayscale(QImage & img)
{
	std::vector<uchar> values;
	CopyImageToBuffer(img, values);
	size_t sz = img.width() * img.height();

	std::vector<uchar> values_out(sz * 4);

	auto start = std::chrono::system_clock::now();

	for (int i = 0; i < sz * 4; i += 4)
	{
		float value = values[i] * 0.21f + values[i + 1] * 0.72f + values[i + 2] * 0.07f;
		values_out[i] = static_cast<uchar>(value);
		values_out[i + 1] = static_cast<uchar>(value);
		values_out[i + 2] = static_cast<uchar>(value);
	}

	auto end = std::chrono::system_clock::now();

	CopyBufferToImage(values_out, img);
	return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

void SequentialImplementation::Sobel(QImage & img)
{
}

float SequentialImplementation::GaussianBlur(QImage & img)
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

	for (int i = 0; i < img.height(); ++i)
	{
		for (int j = 0; j < img.width(); ++j)
		{
			size_t filter_index = 0;
			float R = 0.f, G = 0.f, B = 0.f;
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

			values_out[index] = static_cast<uchar>(R);
			values_out[index + 1] = static_cast<uchar>(G);
			values_out[index + 2] = static_cast<uchar>(B);
		}
	}

	auto end = std::chrono::system_clock::now();

	CopyBufferToImage(values_out, img);
	return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

void SequentialImplementation::Sharpening(QImage & img)
{
}

void SequentialImplementation::ColorSmoothing(QImage & img)
{
}

float SequentialImplementation::KMeans(QImage & img, const int centroid_count)
{
	// copy data to local vector
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
		for (size_t i = 0; i < values.size(); i += 4)
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

			centroids[centroid_idx].sum_x += values[i];
			centroids[centroid_idx].sum_y += values[i + 1];
			centroids[centroid_idx].sum_z += values[i + 2];
			++centroids[centroid_idx].count;
		}

		// Update centroids
		for (size_t c = 0; c < centroids.size(); ++c)
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
	}

	// "Draw"
	for (size_t i = 0; i < values.size(); i += 4)
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

float SequentialImplementation::SOMSegmentation(QImage & img, QImage * ground_truth)
{
	int max_iterations = 1;
	int epochs = 200; // number of iterations
	int neuron_count = 3;
	uint32_t total_sz = img.width() * img.height();
	const double ct_learning_rate = 0.1;
	const double time_constant = epochs / log(neuron_count);

	// generate neurons
	std::vector<Neuron> neurons;

	// copy data to local vector
	std::vector<uchar> values_ui;

	CopyImageToBuffer(img, values_ui);
	std::vector<float> values;

	for (size_t i = 0; i < values_ui.size(); ++i)
	{
		values.push_back(values_ui[i] / 255.f);
	}

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

			// Update neurons
			float neigh_dist = (neurons.size() - 1) * exp(-static_cast<double>(epoch) / time_constant);
			float learning_rate = ct_learning_rate * exp(-static_cast<double>(epoch) / epochs);

			for (size_t c = 0; c < neurons.size(); ++c)
			{
				int n_dist = abs((int)(bmu - c));

				if (n_dist > neigh_dist)
				{
					continue;
				}

				float influence = exp(-(n_dist * n_dist) / (2.f * (neigh_dist * neigh_dist)));

				neurons[c].value_x += learning_rate * influence * (values[index] - neurons[c].value_x);
				neurons[c].value_y += learning_rate * influence * (values[index + 1] - neurons[c].value_y);
				neurons[c].value_z += learning_rate * influence * (values[index + 2] - neurons[c].value_z);
			}
		}
	}

	for (size_t index = 0; index < values.size(); index += 4)
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


	for (size_t i = 0; i < values_ui.size(); ++i)
	{
		values_ui[i] = static_cast<unsigned>(values[i] * 255.f);
	}

	CopyBufferToImage(values_ui, img);

	return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

void SequentialImplementation::Threshold(QImage & img, const float value)
{
}

void SequentialImplementation::RunSIFT(QImage & img)
{
}

std::vector<float> SequentialImplementation::FindImageSIFT(QImage & img, QImage & img_to_find)
{
	return{};
}
