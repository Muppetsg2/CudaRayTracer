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

#define AIR_INDEX 1.0f

using namespace MSTD_NAMESPACE;

namespace craytracer {
	enum class MaterialType : uint8_t { Diffuse = 0, Reflect = 1, Refractive = 2 };

	struct Material {
		MaterialType type;
		vec4 ambient;
		vec4 diffuse;
		vec4 specular;
		float shininess;
		float refractIndex;
	};
}