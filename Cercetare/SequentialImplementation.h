#pragma once
#include "Implementation.h"
class SequentialImplementation :
	public Implementation
{
public:
	SequentialImplementation();
	virtual ~SequentialImplementation();

	// filters
	virtual void CustomFilter(QImage & img, const std::vector<float> & kernel_values);
	virtual float Grayscale(QImage & img);
	virtual void Sobel(QImage & img);
	virtual float GaussianBlur(QImage & img);
	virtual void Sharpening(QImage & img);
	virtual void ColorSmoothing(QImage & img);

	// segmentation
	virtual float KMeans(QImage & img, const int centroid_count);
	virtual float SOMSegmentation(QImage & img, QImage * ground_truth = nullptr);
	virtual void Threshold(QImage & img, const float value);

	virtual void RunSIFT(QImage & img);
	virtual std::vector<float> FindImageSIFT(QImage & img, QImage & img_to_find);
};

