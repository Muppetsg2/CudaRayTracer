/**************************************************************
 *                                                            *
 *  Project:   CudaRayTracer                                  *
 *  Authors:   Muppetsg2 & MAIPA01                            *
 *  License:   MIT License                                    *
 *  Last Update: 10.08.2025                                   *
 *                                                            *
 **************************************************************/

#pragma once
#include "Ray.hpp"
#include "Material.hpp"
#include "Color.hpp"

namespace craytracer {
	class Geometry {
	protected:
		Material _mat = { 
#if _HAS_CXX20
			.ambient = Color::white() * 0.1f,
			.diffuse = Color::white(),
			.specular = Color::white()
#else
			MaterialType::Diffuse,
			Color::white() * 0.1f,
			Color::white(),
			Color::white()
#endif
		};
	public:
		Geometry() = default;

		__device__ Material getMaterial() const { return _mat; }

		__device__ void setMaterial(Material value) { _mat = value; }

		__device__ virtual bool hit(const Ray& ray, RayHit& hit) const = 0;
	};
}