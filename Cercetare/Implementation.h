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
	virtual void CustomFilter(QImage & img, const std::vector<float> & kernel_values) = 0;
	virtual float Grayscale(QImage & img) = 0;
	virtual void Sobel(QImage & img) = 0;
	virtual float GaussianBlur(QImage & img) = 0;
	virtual void Sharpening(QImage & img) = 0;
	virtual void ColorSmoothing(QImage & img) = 0;

	// segmentation
	virtual float KMeans(QImage & img, const int centroid_count) = 0;
	virtual float SOMSegmentation(QImage & img, QImage * ground_truth = nullptr) = 0;
	virtual void Threshold(QImage & img, const float value) = 0;

	virtual void RunSIFT(QImage & img) = 0;
	virtual std::vector<float> FindImageSIFT(QImage & img, QImage & img_to_find) = 0;

	std::vector<float> MSE(QImage & imgGPU, QImage & imgCPU) const;
	std::vector<float> PSNR(QImage & imgGPU, QImage & imgCPU) const;

protected:
	std::function<void(std::string)> m_log;

	void CopyImageToBuffer(QImage & img, std::vector<uchar> & values);
	void CopyBufferToImage(std::vector<uchar> & values, QImage & img, uint32_t row_count = 0, uint32_t col_count = 0);

	void GenerateCentroids(const uint32_t count, std::vector<Centroid> & centroids);
	void GenerateNeurons(const uint32_t count, std::vector<Neuron> & neurons);

	float Distance(const Centroid & c, float x, float y, float z) const;

	float GaussianFunction(int niu, int thetha, int cluster_count) const;
	float NormalizedEuclideanDistance(const Neuron & n1, const Neuron & n2) const;

	float ValidityMeasure(const std::vector<uchar> & data, const std::vector<Neuron> & neurons) const;
	float DaviesBouldinIndex(const std::vector<uchar> & data, const std::vector<Neuron> & neurons) const;

	std::pair<float, float> ComputeVMAndDBIndices(QImage * img);
	std::pair<float, float> ComputeVMAndDBIndices(std::vector<uchar> & values, std::vector<Neuron> & neurons);
};

