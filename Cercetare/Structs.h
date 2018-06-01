#pragma once

#define MAX_ORIS 128

struct Centroid
{
	float value_x;
	float value_y;
	float value_z;
	float sum_x;
	float sum_y;
	float sum_z;
	unsigned count;
};

struct Neuron
{
	float value_x;
	float value_y;
	float value_z;
};

struct KeyPoint
{
	float x_interp;
	float y_interp;
	float magnitude;
	float orientation;
	unsigned x;
	unsigned y;
	unsigned scale;
};

struct FeaturePoint
{
	float x;
	float y;
	double orientations[MAX_ORIS];
};