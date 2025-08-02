#pragma once
#include "vec.hpp"
#include "Material.hpp"
#include "Geometry.hpp"
#include "math_functions.hpp"
#include "Color.hpp"
#include "ldg_helpers.hpp"

using namespace MSTD_NAMESPACE;

namespace craytracer {
	struct LightInput {
		vec3 fragPos = vec3::zero();
		vec2 uv = vec2::zero();
		vec3 norm = vec3(0.0f, 1.0f, 0.0f);
		vec3 viewDir = vec3(0.0f, 0.0f, -1.0f);
		const Material* mat = nullptr;
	};

	class Light {
	protected:
		vec3 _pos = vec3::zero();
		vec4 _color = Color::white();
		float _intensity = 1.f;

		__device__ bool _isCovered(const Ray& ray, Geometry* objects, unsigned int iterations) const {
			RayHit hit;
			Ray r = ray;

			int i = 0;
			do {
				if (!objects->hit(r, hit)) return false;

				if (hit.isHit && hit.hitMat->type != MaterialType::Refractive) {
					i = iterations;
					continue;
				}
				else
				{
					if (iterations == 0) return false;
					bool front_face = r.getDirection().dot(hit.hitNormal) < 0.f;

					vec3 norm = front_face ? hit.hitNormal : -hit.hitNormal;

					float ratio = hit.hitMat->refractIndex / AIR_INDEX;
					if (front_face) ratio = 1.0f / ratio;
					vec3 rayDirection = r.getDirection().normalized();

#ifdef __CUDACC__
					float cos_theta = ::cuda::std::fminf(dot(-rayDirection, norm), 1.0f);
					float sin_theta = 1.0f / rsqrtf(1.0f - cos_theta * cos_theta);
#else
					float cos_theta = ::std::fminf(dot(-rayDirection, norm), 1.0f);
					float sin_theta = ::std::sqrtf(1.0f - cos_theta * cos_theta);
#endif

					bool cannot_refract = ratio * sin_theta > 1.0f || reflectance(cos_theta, ratio) > 1.0f;

					if (cannot_refract) {
						return true;
					}
					else {
						vec3 dir = refract(rayDirection, norm, ratio);
						Ray refracted = Ray(hit.hitPoint + 0.01f * dir, dir);
						if (!front_face) return !_hittedLight(refracted);
						r = refracted;
					}
				}

				++i;
			} while (i < iterations);

			return hit.isHit;
		}

		__device__ virtual bool _hittedLight(const Ray& ray) const {
			return ray.isPointOnRay(_pos);
		}

		__device__ virtual bool _calculateVisibility(const vec3& position, Geometry* objects, float& visibility, curandState* rand_state) const {
#ifdef __CUDACC__
			vec3 rayDir = ldg_vec3(&_pos) - position;
#else
			vec3 rayDir = _pos - position;
#endif
			float lightDist = rayDir.length();

			if (!epsilon_equal(lightDist, 0.f, MSTD_EPSILON)) rayDir.normalize();

			Ray ray = Ray(position + 0.01f * rayDir, rayDir, lightDist);
			bool hitObject = _isCovered(ray, objects, 10u);
			visibility = hitObject ? 0.0f : 1.0f;

			return hitObject;
		}

	public:
		Light() = default;

		__device__ Light(vec3 position, vec4 color, float intensity) { _pos = position; _color = color; _intensity = intensity; }

#ifdef __CUDACC__
		__device__ virtual vec4 calculateColor(const LightInput& input, Geometry* objects, curandState* rand_state) const {
			const vec3 lcol = vec3(ldg_vec4(&_color));
			const float intensity = ldg_float(&_intensity);
			const vec3 pos = ldg_vec3(&_pos);

			// ambient
			const vec3 ambient = ldg_vec4(&input.mat->ambient) * lcol;
#else
		virtual vec4 calculateColor(const LightInput& input, Geometry* objects) const {
			const vec3 lcol = _color;

			// ambient
			const vec3 ambient = input.mat->ambient * lcol;
#endif
			float visibility = 1.0f;
#ifdef __CUDACC__
			if (_calculateVisibility(input.fragPos, objects, visibility, rand_state)) return vec4(ambient, 1.f);

			vec3 lightDir = pos - input.fragPos;
#else
			if (_calculateVisibility(input.fragPos, objects, visibility)) return vec4(ambient, 1.f);

			vec3 lightDir = _pos - input.fragPos;
#endif

			if (!epsilon_equal(lightDir.length_sq(), 0.0f, MSTD_EPSILON_SQ)) lightDir.normalize();

#ifdef __CUDACC__
			// diffuse
			float diff = ::cuda::std::max(input.norm.dot(lightDir), 0.0f);
			vec3 diffuse = lcol * (diff * ldg_vec4(&input.mat->diffuse)) * intensity;
#else
			// diffuse
			float diff = ::std::max(input.norm.dot(input.lightDir), 0.0f);
			vec3 diffuse = lcol * (diff * input.mat->diffuse) * _intensity;
#endif

#ifdef __CUDACC__
			// specular
			// PHONG
			//vec3 reflectDir = reflect(-lightDir, input.norm);
			//float spec = __powf(::cuda::std::max(input.viewDir.dot(reflectDir), 0.0f), ldg_float(&input.mat->shininess));
#else
			// specular
			// PHONG
			//vec3 reflectDir = reflect(-lightDir, input.norm);
			//float spec = ::std::expf(::std::logf(::std::max(input.viewDir.dot(reflectDir), 0.0f)) * input.mat->shininess);
#endif

			// BLINN-PHONG
			vec3 halfwayDir = (lightDir + input.viewDir).normalized();
#ifdef __CUDACC__
			float spec = __powf(::cuda::std::max(input.norm.dot(halfwayDir), 0.0f), ldg_float(&input.mat->shininess));
			vec3 specular = lcol * (spec * ldg_vec4(&input.mat->specular)) * intensity;
#else
			float spec = ::std::expf(::std::logf(::std::max(input.norm.dot(halfwayDir), 0.0f)) * input.mat->shininess);
			vec3 specular = lcol * (spec * input.mat->specular) * _intensity;
#endif

			vec3 result = ambient + diffuse + specular;
			return vec4(result, 1.0f);
		}
	};
}