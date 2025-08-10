/**************************************************************
 *                                                            *
 *  Project:   CudaRayTracer                                  *
 *  Authors:   Muppetsg2 & MAIPA01                            *
 *  License:   MIT License                                    *
 *  Last Update: 10.08.2025                                   *
 *                                                            *
 **************************************************************/

#pragma once
#include "vec.hpp"

namespace craytracer {
	struct Vertex {
		vec3 pos;
		vec2 tex;
		vec3 normal;
	};
}