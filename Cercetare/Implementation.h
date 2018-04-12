#pragma once

#include <QImage>
#include <functional>

#include "Structs.h"

class Implementation
{
public:
	Implementation();
	virtual ~Implementation();

	void SetLogFunction(std::function<void(std::string)> log_func);

	// filters
	virtual float Grayscale(QImage & img) = 0;
	virtual void Sobel(QImage & img) = 0;
	virtual float GaussianBlur(QImage & img) = 0;
	virtual void Sharpening(QImage & img) = 0;
	virtual void ColorSmoothing(QImage & img) = 0;

	// segmentation
	virtual float KMeans(QImage & img, const int centroid_count) = 0;
	virtual float SOMSegmentation(QImage & img, QImage * ground_truth = nullptr) = 0;
	virtual void Threshold(QImage & img, const float value) = 0;

protected:
	std::function<void(std::string)> m_log;

	void CopyImageToBuffer(QImage & img, std::vector<uchar> & values);
	void CopyBufferToImage(std::vector<uchar> & values, QImage & img);

	void GenerateCentroids(const uint32_t count, std::vector<Centroid> & centroids);
	void GenerateNeurons(const uint32_t count, std::vector<Neuron> & neurons);

	float Distance(const Centroid & c, uint x, uint y, uint z) const;
};

