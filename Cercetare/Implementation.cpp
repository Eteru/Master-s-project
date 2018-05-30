
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
