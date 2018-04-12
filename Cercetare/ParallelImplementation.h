#pragma once
#include "Implementation.h"
class ParallelImplementation :
	public Implementation
{
public:
	ParallelImplementation();
	virtual ~ParallelImplementation();
	
	// filters
	virtual float Grayscale(QImage & img);
	virtual void Sobel(QImage & img);
	virtual float GaussianBlur(QImage & img);
	virtual void Sharpening(QImage & img);
	virtual void ColorSmoothing(QImage & img);

	// segmentation
	virtual float KMeans(QImage & img, const int centroid_count);
	virtual float SOMSegmentation(QImage & img, QImage * ground_truth = nullptr);
	virtual void Threshold(QImage & img, const float value);
};

