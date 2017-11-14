#pragma once
#ifndef GLWidget_HPP
#define GLWidget_HPP

#define __CL_ENABLE_EXCEPTIONS

#include <QColor>
#include <QImage>

#include <vector>
#include <fstream>
#include <functional>

#include <CL/cl.hpp>

#include "structs.hpp"

class CLWrapper
{
public:
	explicit CLWrapper();
	~CLWrapper(void);

	void SetLogFunction(std::function<void(std::string)> log_func);

	void SetData(QImage & img);

	void Grayscale(QImage & img);
	void Sobel(QImage & img);
	void GaussianBlur(QImage & img);
	void Sharpening(QImage & img);
	void ColorSmoothing(QImage & img);

	void KMeans(QImage & img, const int centroid_count);
	void CMeans(QImage & img, const int centroid_count);
	void SOMSegmentation(QImage & img);
	void Threshold(QImage & img, const float value);

private:
	bool m_initialized;
	bool m_CLready;
	uint32_t m_width, m_height;

	cl::size_t<3> m_origin, m_region;

	std::vector<uchar> m_values;

	// OpenCL
	cl::Context m_contextCL;
	cl::CommandQueue m_queue;
	cl::Image2D *m_data, *m_data_aux;
	cl::Program m_program_utils;
	cl::Program m_program_filters;
	cl::Program m_program_segmentation;
	cl::Program m_program_som;
	cl::Kernel m_kernel_grayscale;
	cl::Kernel m_kernel_convolute, m_kernel_sharpening, m_kernel_color_smoothing;
	cl::Kernel m_kernel_kmeans, m_kernel_update_centroids, m_kernel_kmeans_draw;
	cl::Kernel m_kernel_threshold;
	cl::Kernel m_kernel_find_bmu;
	cl::Kernel m_kernel_update_weights;
	cl::Kernel m_kernel_som_draw;
	cl::NDRange m_globalRange;
	
	QImage m_crtImage;

	std::function<void(std::string)> m_log;
	

	void LoadProgram(cl::Program & program, std::vector<cl::Device> & devices, std::string file);
	
	void LoadData(QImage & img);
	void InitializeCL(void);

	void GenerateCentroids(const uint32_t count, std::vector<Centroid> & centroids);
	void GenerateNeurons(const uint32_t count, std::vector<Neuron> & neurons);

	void CheckSegmentationNeurons(cl::Image2D * gpu_data, cl::Buffer & neuronsCL, std::vector<Neuron> & neurons);

	float GaussianFunction(int niu, int thetha, int cluster_count) const;
	float NormalizedEuclideanDistance(const Neuron & n1, const Neuron & n2) const;
	float ValidityMeasure(const std::vector<uchar> & data, const std::vector<Neuron> & neurons) const;
	float DaviesBouldinIndex(const std::vector<uchar> & data, const std::vector<Neuron> & neurons) const;

};

#endif // GLWidget_HPP