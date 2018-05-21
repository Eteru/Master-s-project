#pragma once

struct Centroid
{
	float x;
	float y;
	float z;
	float sum_x;
	float sum_y;
	float sum_z;
	int count;
};

struct Neuron
{
	float x;
	float y;
	float z;
};

struct KeyPoint
{
	float x;
	float y;
	float magnitude;
	float orientation;
	unsigned scale;
};