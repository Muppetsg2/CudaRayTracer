#pragma once
#include "vec.hpp"
#include "Material.hpp"
#include "math_functions.hpp"
#include "ldg_helpers.hpp"

using namespace MSTD_NAMESPACE;

namespace craytracer {
	struct RayHit {
		bool isHit = false;
		vec3 hitPoint = vec3::zero();
		vec3 hitNormal = vec3(0.0f, 1.0f, 0.0f);
		vec2 hitUV = vec2::zero();
		float hitDist = FLT_MAX;
		const Material* hitMat = nullptr;
	};

	class Ray {
	private:
		vec3 _origin = vec3::zero();
		vec3 _direction = vec3(0.0f, 0.0f, -1.0f);
		float _distance = 0.0f;

	public:
		__device__ Ray() {}

		__device__ Ray(vec3 origin, vec3 direction) : _distance(0.0f), _origin(origin) {
			_direction = epsilon_equal(direction.length(), 0.f, MSTD_EPSILON) ? direction : direction.normalized();
		}

		__device__ Ray(vec3 origin, vec3 direction, float distance) : _origin(origin) {
			_direction = epsilon_equal(direction.length(), 0.f, MSTD_EPSILON) ? direction : direction.normalized();
			setDistance(distance);
		}

		__device__ vec3 getOrigin() const { return _origin; }

		__device__ vec3 getDirection() const { return _direction; }

		__device__ float getDistance() const { return _distance; }

		__device__ void setOrigin(vec3 value) { _origin = value; }

		__device__ void setDirection(vec3 value) {
			if (epsilon_equal(value.length(), 0.f, MSTD_EPSILON)) return;
			_direction = value.normalized();
		}

		__device__ void setDistance(float value) 
		{
#ifdef __CUDACC__
			_distance = ::cuda::std::max(0.f, value);
#else
			_distance = ::std::max(0.f, value);
#endif
		}

		__device__ bool isPointOnRay(vec3 point) const {
#ifdef __CUDACC__
			const vec3 origin = ldg_vec3(&_origin);
			const vec3 direction = ldg_vec3(&_direction);
			const float distance = ldg_float(&_distance);
			vec3 v = point - origin;
			vec3 cross = direction.cross(v);
#else
			vec3 v = point - _origin;
			vec3 cross = _direction.cross(v);
#endif

			if (cross.length() >= MSTD_EPSILON) return false;

#ifdef __CUDACC__
			float dot = v.dot(direction);
			if (epsilon_equal(distance, 0.f, MSTD_EPSILON)) return dot >= 0.f;
			return dot >= 0.f && dot <= distance;
#else
			float dot = v.dot(_direction);
			if (epsilon_equal(_distance, 0.f, MSTD_EPSILON)) return dot >= 0.f;
			return dot >= 0.f && dot <= _distance;
#endif
		}
	};
}