
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
	for (Centroid & centroid : centroids)
	{
		centroid = {};

		centroid.value.x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		centroid.value.y = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		centroid.value.z = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

		centroid.sum.x = 0;
		centroid.sum.y = 0;
		centroid.sum.z = 0;

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

		neuron.value.x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		neuron.value.y = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		neuron.value.z = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

		std::cout << " (" << neuron.value.x << ", " << neuron.value.y << ", " << neuron.value.z << ")";
	}

	std::cout << std::endl;
}

float Implementation::Distance(const Centroid & c, uint x, uint y, uint z) const
{
	return c.value.x * c.value.x + x * x - 2 * c.value.x * x +
		c.value.y * c.value.y + y * y - 2 * c.value.y * y +
		c.value.z * c.value.z + z * z - 2 * c.value.z * z;
}
