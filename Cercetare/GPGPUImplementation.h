#pragma once
#ifndef GPGPUImplementation_H
#define GPGPUImplementation_H

#define __CL_ENABLE_EXCEPTIONS

#include <QColor>

#include <vector>
#include <fstream>
#include <functional>

#include <CL/cl.hpp>

#include "Implementation.h"
#include "Structs.h"

class GPGPUImplementation : public Implementation
{
public:
	explicit GPGPUImplementation();
	~GPGPUImplementation(void);
	
	void SetData(QImage & img);

	virtual float Grayscale(QImage & img);
	virtual void Sobel(QImage & img);
	virtual float GaussianBlur(QImage & img);
	virtual void Sharpening(QImage & img);
	virtual void ColorSmoothing(QImage & img);

	virtual float KMeans(QImage & img, const int centroid_count);
	virtual float SOMSegmentation(QImage & img, QImage * ground_truth = nullptr);
	virtual void Threshold(QImage & img, const float value);

private:
	bool m_initialized;
	bool m_CLready;
	uint32_t m_width, m_height;

	std::vector<uchar> m_values_orig;
	std::vector<uchar> m_values;

	// OpenCL
	cl::Context m_contextCL;
	cl::CommandQueue m_queue;
	cl::size_t<3> m_origin, m_region;
	cl::Image2D *m_data_original, *m_data_front, *m_data_back;

	std::map<const char *, cl::Kernel> m_kernels;
	cl::NDRange m_globalRange;
	
	void PrintDevices(const std::vector<cl::Device> & devices) const;
	void LoadProgram(cl::Program & program, std::vector<cl::Device> & devices, std::string file);
	
	void LoadData(QImage & img);
	void InitializeCL(void);

	std::pair<float, float> CheckSegmentationNeurons(cl::Buffer & neuronsCL, std::vector<Neuron> & neurons);

	float GaussianFunction(int niu, int thetha, int cluster_count) const;
	float NormalizedEuclideanDistance(const Neuron & n1, const Neuron & n2) const;
	float ValidityMeasure(const std::vector<uchar> & data, const std::vector<Neuron> & neurons) const;
	float DaviesBouldinIndex(const std::vector<uchar> & data, const std::vector<Neuron> & neurons) const;

	std::pair<float, float> ComputeVMAndDBIndices(QImage * img);
};

#endif // GPGPUImplementation_H