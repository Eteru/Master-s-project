
#include "Implementation.h"
#include <iostream>
#include <set>

Implementation::Implementation()
{
}

Implementation::~Implementation()
{
}


void Implementation::SetLogFunction(std::function<void(std::string)> log_func)
{
	m_log = log_func;
}


void Implementation::CopyImageToBuffer(QImage & img, std::vector<uchar>& values)
{
	values.resize(img.width() * img.height() * 4);

	for (int i = 0, row = 0; row < img.height(); ++row, i += img.bytesPerLine())
	{
		memcpy(&values[i], img.scanLine(row), img.bytesPerLine());
	}
}

void Implementation::CopyBufferToImage(std::vector<uchar>& values, QImage & img, uint32_t row_count, uint32_t col_count)
{
	int bpl = img.bytesPerLine();
	if (row_count > 0 || col_count > 0)
	{
		bpl = bpl / img.width() * col_count;
	}
	else
	{
		row_count = img.height();
	}

	for (int i = 0, row = 0; row < row_count; ++row, i += bpl)
	{
		memcpy(img.scanLine(row), &values[i], bpl);
	}
	
}

void Implementation::GenerateCentroids(const uint32_t count, std::vector<Centroid> & centroids)
{
	centroids.resize(count);

	std::cout << "Generated centroids: ";
	for (Centroid & centroid : centroids)
	{
		centroid = {};

		centroid.value_x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		centroid.value_y = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		centroid.value_z = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		
		centroid.count = 0;
	}

	for (Centroid & centroid : centroids)
	{
		std::cout << " (" << centroid.value_x << ", " << centroid.value_y << ", " << centroid.value_z << "), count = " << centroid.count;
	}

	std::cout << std::endl;
}

void Implementation::GenerateNeurons(const uint32_t count, std::vector<Neuron>& neurons)
{
	neurons.resize(count);

	std::cout << "Generated neurons: ";
	for (Neuron & neuron : neurons)
	{
		neuron = {};

		neuron.value_x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		neuron.value_y = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		neuron.value_z = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

		std::cout << " (" << neuron.value_x << ", " << neuron.value_y << ", " << neuron.value_z << ")";
	}

	std::cout << std::endl;
}

float Implementation::Distance(const Centroid & c, float x, float y, float z) const
{
	return c.value_x * c.value_x + x * x - 2 * c.value_x * x +
		c.value_y * c.value_y + y * y - 2 * c.value_y * y +
		c.value_z * c.value_z + z * z - 2 * c.value_z * z;
}

float Implementation::GaussianFunction(int niu, int thetha, int cluster_count) const
{
	static const float double_pi = 2 * 3.14159f;
	float pow_thetha = thetha * thetha;
	float pow_k_niu = (cluster_count - niu) * (cluster_count - niu);

	return expf(-(pow_k_niu / (2 * pow_thetha))) / sqrtf(double_pi * pow_thetha);
}

float Implementation::NormalizedEuclideanDistance(const Neuron & n1, const Neuron & n2) const
{
	float diffx = n1.value_x - n2.value_x;
	float diffy = n1.value_y - n2.value_y;
	float diffz = n1.value_z - n2.value_z;

	return sqrt(diffx*diffx + diffy*diffy + diffz*diffz);
}


std::vector<float> Implementation::MSE(QImage & imgGPU, QImage & imgCPU) const
{
	float sumR = 0.f;
	float sumG = 0.f;
	float sumB = 0.f;
	int MN = imgGPU.height() * imgGPU.width();

	for (int i = 0; i < imgGPU.height(); ++i)
	{
		for (int j = 0; j < imgGPU.width(); ++j)
		{
			QRgb pxGPU = imgGPU.pixel(j, i);
			QRgb pxCPU = imgCPU.pixel(j, i);
			int r = qRed(pxGPU) - qRed(pxCPU);
			int g = qGreen(pxGPU) - qGreen(pxCPU);
			int b = qBlue(pxGPU) - qBlue(pxCPU);

			sumR += (r*r);
			sumG += (g*g);
			sumB += (b*b);
		}
	}

	sumR /= MN;
	sumG /= MN;
	sumB /= MN;

	return{ sumR, sumG, sumB };
}

std::vector<float> Implementation::PSNR(QImage & imgGPU, QImage & imgCPU) const
{
	std::vector<float> mse = MSE(imgGPU, imgCPU);
	std::vector<float> results(4); // mse, mse, mse, psnr

	float mse_1 = (mse[0] + mse[1] + mse[2]) * 0.33f;

	results[0] = mse[0];
	results[1] = mse[1];
	results[2] = mse[2];
	results[3] = -10.f * log(mse_1 / (255 * 255));

	return results;
}


float Implementation::ValidityMeasure(const std::vector<uchar>& data, const std::vector<Neuron>& neurons) const
{
	static const int c = 15; // can be between 15 and 25
	float intra_distance = 0.f;

	for (int i = 0; i < data.size(); i += 4)
	{
		float min_dist = FLT_MAX;
		size_t c_idx = -1;

		Neuron crt_pixel = {
			static_cast<unsigned>(data[i]) / 255.f,
			static_cast<unsigned>(data[i + 1]) / 255.f,
			static_cast<unsigned>(data[i + 2]) / 255.f
		};


		for (int n_idx = 0; n_idx < neurons.size(); ++n_idx) {

			float dist = NormalizedEuclideanDistance(crt_pixel, neurons[n_idx]);

			dist *= dist;

			if (dist < min_dist)
			{
				min_dist = dist;
				c_idx = n_idx;
			}
		}

		intra_distance += min_dist;
	}

	intra_distance /= (data.size() / 4.f);

	float inter_distance = FLT_MAX;
	for (int i = 0; i < neurons.size(); ++i)
	{
		for (int j = 0; j < neurons.size(); ++j)
		{
			if (i == j) {
				continue;
			}

			float dist = NormalizedEuclideanDistance(neurons[i], neurons[j]);

			dist *= dist;

			if (dist < inter_distance)
			{
				inter_distance = dist;
			}
		}
	}

	float y = c * GaussianFunction(2, 1, neurons.size()) + 1;

	return y * (intra_distance / inter_distance);
}

float Implementation::DaviesBouldinIndex(const std::vector<uchar>& data, const std::vector<Neuron>& neurons) const
{
	std::vector<int> cluster_size(neurons.size(), 0);
	std::vector<float> cluster_distances(neurons.size(), 0.f);

	for (int i = 0; i < data.size(); i += 4)
	{
		float min_dist = FLT_MAX;
		size_t c_idx = -1;

		Neuron crt_pixel = {
			static_cast<float>(static_cast<unsigned>(data[i]) / 255.f),
			static_cast<float>(static_cast<unsigned>(data[i + 1]) / 255.f),
			static_cast<float>(static_cast<unsigned>(data[i + 2] / 255.f))
		};


		for (int n_idx = 0; n_idx < neurons.size(); ++n_idx)
		{

			float dist = NormalizedEuclideanDistance(crt_pixel, neurons[n_idx]);

			if (dist < min_dist)
			{
				min_dist = dist;
				c_idx = n_idx;
			}
		}

		++cluster_size[c_idx];

		cluster_distances[c_idx] += min_dist;
	}

	for (int i = 0; i < cluster_distances.size(); ++i)
	{
		if (0 != cluster_size[i])
			cluster_distances[i] /= static_cast<float>(cluster_size[i]);
	}

	std::vector<float> D(neurons.size(), 0.f);
	for (int i = 0; i < neurons.size(); ++i)
	{
		for (int j = 0; j < neurons.size(); ++j)
		{
			if (i == j)
			{
				continue;
			}

			float dist = NormalizedEuclideanDistance(neurons[i], neurons[j]);
			float Rij = (cluster_distances[i] + cluster_distances[j]) / dist;
			
			if (Rij > D[i])
			{
				D[i] = Rij;
			}
		}
	}

	float dbi = std::accumulate(D.begin(), D.end(), 0.f) / D.size();

	dbi = (dbi < 0.0001f) ? 100.f : dbi;

	return dbi;
}

std::pair<float, float> Implementation::ComputeVMAndDBIndices(QImage * img)
{
	std::vector<uchar> values(img->byteCount());

	CopyImageToBuffer(*img, values);

	std::set<QRgb> uq_neuron;
	for (int i = 0; i < img->height(); ++i)
	{
		for (int j = 0; j < img->width(); ++j)
		{
			QRgb px = img->pixel(QPoint(j, i));

			//std::cout << "[Ground Truth]: (" << i << "," << j << ") = (" + std::to_string(qRed(px)) + ", " + std::to_string(qGreen(px)) + ", " + std::to_string(qBlue(px)) + ")" << std::endl;
			uq_neuron.insert(px);
		}
	}

	std::vector<Neuron> neurons;
	for (auto n : uq_neuron)
	{
		Neuron nv = { qRed(n) / 256.f, qGreen(n) / 256.f, qBlue(n) / 256.f };

		neurons.push_back(nv);
	}

	float VM = ValidityMeasure(values, neurons);
	float DBI = DaviesBouldinIndex(values, neurons);

	return{ VM, DBI };
}

std::pair<float, float> Implementation::ComputeVMAndDBIndices(std::vector<uchar> & values, std::vector<Neuron>& neurons)
{
	float VM = ValidityMeasure(values, neurons);
	float DBI = DaviesBouldinIndex(values, neurons);

	return{ VM, DBI };
}
