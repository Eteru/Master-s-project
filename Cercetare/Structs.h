#pragma once

#include <CL\cl.h>

struct Centroid
{
	cl_float3 value;
	cl_float3 sum;
	int count;
};

struct Neuron
{
	cl_float3 value;
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