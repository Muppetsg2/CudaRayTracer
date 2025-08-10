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
#include "Ray.hpp"
#include "Geometry.hpp"
#include "math_functions.hpp"
#include "ldg_helpers.hpp"

namespace craytracer {
	class Sphere : public Geometry {
	private:
		vec3 _center;
		float _radius;

	public:
		__device__ Sphere() : _center(vec3::zero()), _radius(0.0f) {}

		__device__ Sphere(float radius) : _center(vec3::zero()), _radius(radius) {}

		__device__ Sphere(vec3 center) : _center(center), _radius(1.0f) {}

		__device__ Sphere(vec3 center, float radius) : _center(center), _radius(radius) {}

		__device__ vec3 getCenter() const { return _center; }

		__device__ float getRadius() const { return _radius; }

		__device__ void setCenter(vec3 value) { _center = value; }

		__device__ void setRadius(float value) { _radius = value; }

		__device__ bool hit(const Ray& ray, RayHit& hit) const {
			float t0, t1; // Solutions for t if the ray intersects the sphere
#ifdef __CUDACC__
			const vec3 center = ldg_vec3(&_center);
			const float radius = ldg_float(&_radius);
#endif

#ifdef __CUDACC__
			vec3 L = ray.getOrigin() - center;
#else
			vec3 L = ray.getOrigin() - _center;
#endif

			float a = ray.getDirection().dot(ray.getDirection());
			float b = 2.f * ray.getDirection().dot(L);
#ifdef __CUDACC__
			float c = L.dot(L) - radius * radius;
#else
			float c = L.dot(L) - _radius * _radius;
#endif

			float discr = b * b - 4.f * a * c;
			if (discr < 0.f) return false;
#ifdef __CUDACC__
			else if (epsilon_equal(discr, 0.f, MSTD_CUDA_EPSILON)) t0 = t1 = -0.5f * __fdividef(b, a);
#else
			else if (epsilon_equal(discr, 0.f, MSTD_EPSILON)) t0 = t1 = -0.5f * b / a;
#endif
			else {
#ifdef __CUDACC__
				float discr_sq = __fdividef(1.0f, rsqrtf(discr));
#else
				float discr_sq = ::std::sqrtf(discr);
#endif
				float q = (b > 0) ?
					-0.5f * (b + discr_sq) :
					-0.5f * (b - discr_sq);

#ifdef __CUDACC__
				t0 = __fdividef(q, a);
				t1 = __fdividef(c, q);
#else
				t0 = q / a;
				t1 = c / q;
#endif
			}
			if (t0 > t1) 
#ifdef __CUDACC__
				::thrust::swap(t0, t1);
#else
				::std::swap(t0, t1);
#endif

			if (t0 < 0.f) {
				t0 = t1; // If t0 is negative, let's use t1 instead.
				if (t0 < 0.f) return false; // Both t0 and t1 are negative.
			}

			if (ray.getDistance() > 0.0f && t0 > ray.getDistance()) return false;

			// RayHit
			hit.isHit = true;
			hit.hitPoint = ray.getOrigin() + ray.getDirection() * t0;
			hit.hitMat = &_mat;
			hit.hitDist = t0;

#ifdef __CUDACC__
			hit.hitNormal = (hit.hitPoint - center).normalize();
#else
			hit.hitNormal = (hit.hitPoint - _center).normalize();
#endif
			hit.hitUV = vec2(
#ifdef __CUDACC__
				0.5f + atan2f(hit.hitNormal.z(), hit.hitNormal.x()) * MSTD_CUDA_1_PI_2,
				0.5f - asinf(hit.hitNormal.y()) * MSTD_CUDA_1_PI
#else
				0.5f + ::std::atan2(hit.hitNormal.z(), hit.hitNormal.x()) / (2.f * M_PI),
				0.5f - ::std::asin(hit.hitNormal.y()) / M_PI
#endif
			);

			return true;
		}
	};
}