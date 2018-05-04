#pragma once

#include <string>

namespace Constants
{
	// General
	const std::string APP_NAME = "Placeholder";
	const std::string ORG_NAME = "Ciprian Aprodu";

	// Kernels
	const std::string KERNEL_GRAYSCALE = "grayscale";
	const std::string KERNEL_CONVOLUTE = "convolute";
	const std::string KERNEL_COLOR_SMOOTHING = "color_smooth_filter";
	const std::string KERNEL_SHARPNESS = "sharpness_filter";
	const std::string KERNEL_KMEANS = "kmeans";
	const std::string KERNEL_KMEANS_DRAW = "kmeans_draw";
	const std::string KERNEL_KMEANS_UPDATE_CENTROIDS = "update_centroids";
	const std::string KERNEL_THRESHOLD = "threshold";
	const std::string KERNEL_SOM_FIND_BMU = "find_bmu";
	const std::string KERNEL_SOM_UPDATE_WEIGHTS = "update_weights";
	const std::string KERNEL_SOM_DRAW = "som_draw";
	const std::string KERNEL_POST_NOISE_REDUCTION = "noise_reduction";
	const std::string KERNEL_POST_REGION_MERGING = "region_merging";
	const std::string KERNEL_RESIZE = "resize_image";
	const std::string KERNEL_IMAGE_DIFFERENCE = "difference";
	const std::string KERNEL_FIND_EXTREME_POINTS = "find_extreme_points";
	const std::string KERNEL_INT_TO_FLOAT = "int_to_float";
}