
#include "Implementation.h"
#include <iostream>

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

void Implementation::CopyBufferToImage(std::vector<uchar>& values, QImage & img)
{
	for (int i = 0, row = 0; row < img.height(); ++row, i += img.bytesPerLine())
	{
		memcpy(img.scanLine(row), &values[i], img.bytesPerLine());
	}
}

void Implementation::GenerateCentroids(const uint32_t count, std::vector<Centroid> & centroids)
{
	centroids.resize(count);
	for (Centroid & centroid : centroids)
	{
		centroid = {};

		centroid.x = std::rand() % 256;
		centroid.y = std::rand() % 256;
		centroid.z = std::rand() % 256;

		centroid.sum_x = 0;
		centroid.sum_y = 0;
		centroid.sum_z = 0;

		centroid.count = 0;
	}
}

void Implementation::GenerateNeurons(const uint32_t count, std::vector<Neuron>& neurons)
{
	neurons.resize(count);

	std::cout << "Generated neurons: ";
	for (Neuron & neuron : neurons)
	{
		neuron = {};

		neuron.x = std::rand() % 256;
		neuron.y = std::rand() % 256;
		neuron.z = std::rand() % 256;

		std::cout << " (" << neuron.x << ", " << neuron.y << ", " << neuron.z << ")";
	}

	std::cout << std::endl;
}

float Implementation::Distance(const Centroid & c, uint x, uint y, uint z) const
{
	return c.x * c.x + x * x - 2 * c.x * x +
		c.y * c.y + y * y - 2 * c.y * y +
		c.z * c.z + z * z - 2 * c.z * z;
}
