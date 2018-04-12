#pragma once

namespace Constants
{
	// General
	const char APP_NAME[] = "Placeholder";
	const char ORG_NAME[] = "Ciprian Aprodu";

	// Kernels
	const char KERNEL_GRAYSCALE[] = "grayscale";
	const char KERNEL_CONVOLUTE[] = "convolute";
	const char KERNEL_COLOR_SMOOTHING[] = "color_smooth_filter";
	const char KERNEL_SHARPNESS[] = "sharpness_filter";
	const char KERNEL_KMEANS[] = "kmeans";
	const char KERNEL_KMEANS_DRAW[] = "kmeans_draw";
	const char KERNEL_KMEANS_UPDATE_CENTROIDS[] = "update_centroids";
	const char KERNEL_THRESHOLD[] = "threshold";
	const char KERNEL_SOM_FIND_BMU[] = "find_bmu";
	const char KERNEL_SOM_UPDATE_WEIGHTS[] = "update_weights";
	const char KERNEL_SOM_DRAW[] = "som_draw";
	const char KERNEL_POST_NOISE_REDUCTION[] = "noise_reduction";
	const char KERNEL_POST_REGION_MERGING[] = "region_merging";
}