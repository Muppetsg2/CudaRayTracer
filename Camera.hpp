#pragma once
#include "vec.hpp"
#include "Ray.hpp"
#include "math_functions.hpp"
#include "ldg_helpers.hpp"

using namespace MSTD_NAMESPACE;

namespace craytracer {
	enum class CameraType : uint8_t {
		ORTHOGRAPHIC = 0,
		PERSPECTIVE = 1
	};

	class Camera {
	private:
		float _fov = 0.785f;
		float _orthoScale = 2.0f;
		CameraType _type = CameraType::PERSPECTIVE;
		vec3 _pos = vec3::zero();
		vec3 _front = vec3(0.0f, 0.0f, -1.0f);

	public:
		Camera() = default;

		__device__ Camera(vec3 position, vec3 front, CameraType type, float fovRad, float orthoScale) {
			_pos = position;
			_type = type;

			_front = epsilon_equal(front.length(), 0.f, MSTD_EPSILON) ? vec3(0.f, 0.f, -1.f) : front.normalized();

			setFov(fovRad);
			setOrthographicScale(orthoScale);
		}

		__device__ float getFov() const { return _fov; }

		__device__ float getOrthographicScale() const { return _orthoScale; }

		__device__ vec3 getPosition() const { return _pos; }

		__device__ vec3 getFront() const { return _front; }

		__device__ void setFov(float radians)
		{
#ifdef __CUDACC__
			_fov = ::cuda::std::max(radians, MSTD_EPSILON);
#else
	        _fov = ::std::max(radians, MSTD_EPSILON);
#endif
		}

		__device__ void setOrthographicScale(float value)
		{
#ifdef __CUDACC__
			_orthoScale = ::cuda::std::max(value, MSTD_EPSILON);
#else
			_orthoScale = ::std::max(value, MSTD_EPSILON);
#endif
		}

		__device__ void setPosition(vec3 value) { _pos = value; }

		__device__ void setFront(vec3 value) {
			if (epsilon_equal(value.length(), 0.f, MSTD_EPSILON)) return;
			_front = value.normalized();
		}

		__device__ Ray getRay(float x, float y, float width, float height) const {
#ifdef __CUDACC__
			const vec3 front = ldg_vec3(&_front);
			const vec3 pos = ldg_vec3(&_pos);
			const float fov = ldg_float(&_fov);
			const float orthoScale = ldg_float(&_orthoScale);
			const CameraType type = ldg_enum_uint8_type(&_type);
#endif

			vec3 rayOrigin;
#ifdef __CUDACC__
			vec3 rightDir = front.cross(vec3(0.f, 1.f, 0.f)).normalized();
			vec3 upDir = rightDir.cross(front).normalize();
#else
			vec3 rightDir = _front.cross(vec3(0.f, 1.f, 0.f)).normalized();
			vec3 upDir = rightDir.cross(_front).normalize();
#endif

#ifdef __CUDACC__
			switch (type) {
#else
			switch (_type) {
#endif
				case CameraType::ORTHOGRAPHIC: {
					float aspect = height / width;
#ifdef __CUDACC__
					float orthoWidth = orthoScale * (aspect > 1.f ? aspect : 1.f);
					float orthoHeight = orthoScale * (aspect > 1.f ? aspect : 1.f);
#else
					float orthoWidth = _orthoScale * (aspect > 1.f ? aspect : 1.f);
					float orthoHeight = _orthoScale * (aspect > 1.f ? aspect : 1.f);
#endif

					rayOrigin = x * rightDir * orthoWidth + y * upDir * orthoHeight;
#ifdef __CUDACC__
					rayOrigin += pos;
					return Ray(rayOrigin, front);
#else
					rayOrigin += _pos;
					return Ray(rayOrigin, _front);
#endif
				}
				case CameraType::PERSPECTIVE: {
					rayOrigin = x * rightDir + y * upDir;
#ifdef __CUDACC__
					rayOrigin += pos;
					vec3 camPos = pos - (((width * 0.5f) / ::cuda::std::tanf(fov * 0.5f)) * front);
#else
					rayOrigin += _pos;
					vec3 camPos = _pos - (((width * 0.5f) / ::std::tanf(_fov * 0.5f)) * _front);
#endif
					return Ray(rayOrigin, rayOrigin - camPos);
				}
			}
			return Ray();
		}
	};
}