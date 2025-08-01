#pragma once
#include "vec.hpp"
#include "mat.hpp"
#include "ltc.hpp"
#include "Light.hpp"
#include "Texture.hpp"
#include "math_functions.hpp"
#include "ldg_helpers.hpp"

using namespace MSTD_NAMESPACE;

namespace craytracer {
	enum class VERTEX_ORDER : uint8_t {
		CCW = 0,
		CW = 1
	};

	class AreaLight : public Light {
	private:
		vec3 _points[4] = {
			vec3(-1.f, -1.f, 0.f),
			vec3(-1.f, 1.f, 0.f),
			vec3(1.f, 1.f, 0.f),
			vec3(1.f, -1.f, 0.f)
		};
		bool _twoSided = false;
		bool _initialized = false;

		// Predefine
		Texture _ltc_1;
		Texture _ltc_2;
		//float _roughness = 1.0f;
		bool _clipless = true;

		unsigned int _shadowSamples = 10u;

		// Const
		const float LUT_SIZE = 64.0f;
		const float LUT_SCALE = (LUT_SIZE - 1.0f) / LUT_SIZE;
		const float LUT_BIAS = 0.5f / LUT_SIZE;

		__device__ void _initLTCTextures() {
			_ltc_1 = Texture(ltc_1, LTC_WIDTH, LTC_HEIGHT, LTC_CHANNELS, WRAP_MODE::CLAMP, WRAP_MODE::CLAMP);
			_ltc_2 = Texture(ltc_2, LTC_WIDTH, LTC_HEIGHT, LTC_CHANNELS, WRAP_MODE::CLAMP, WRAP_MODE::CLAMP);
		}

		__device__ void _defineCenterPoint() {
			for (const vec3& pt : _points) {
				_pos.x() += pt.x();
				_pos.y() += pt.y();
				_pos.z() += pt.z();
			}

			_pos.x() *= 0.25f;
			_pos.y() *= 0.25f;
			_pos.z() *= 0.25f;

			for (vec3& pt : _points) {
				pt = pt - _pos;
			}
		}

		__device__ vec3 _integrateEdgeVec(vec3 v1, vec3 v2) const {
			float x = v1.dot(v2);
#ifdef __CUDACC__
			float y = ::cuda::std::fabsf(x);
#else
			float y = ::std::fabsf(x);
#endif

			float a = 0.8543985f + (0.4965155f + 0.0145206f * y) * y;
			float b = 3.4175940f + (4.1616724f + y) * y;
			float v = a / b;

			float theta_sintheta = (x > 0.0f) ? 
				v : 
#ifdef __CUDACC__
				0.5f * rsqrtf(::cuda::std::max(1.0f - x * x, MSTD_EPSILON)) - v;
#else
				0.5f * Q_rsqrtf(::std::max(1.0f - x * x, MSTD_EPSILON)) - v;
#endif

			return v1.cross(v2) * theta_sintheta;
		}

		__device__ float _integrateEdge(vec3 v1, vec3 v2) const {
			return _integrateEdgeVec(v1, v2).z();
		}

		__device__ int _clipQuadToHorizon(vec3 L[5]) const {
			// detect clipping config
			int config = 0;
			if (L[0].z() > 0.0f) config += 1;
			if (L[1].z() > 0.0f) config += 2;
			if (L[2].z() > 0.0f) config += 4;
			if (L[3].z() > 0.0f) config += 8;

			// clip
			int n = 0;

			if (config == 0)
			{
				// clip all
			}
			else if (config == 1) // V1 clip V2 V3 V4
			{
				n = 3;
				L[1] = -L[1].z() * L[0] + L[0].z() * L[1];
				L[2] = -L[3].z() * L[0] + L[0].z() * L[3];
			}
			else if (config == 2) // V2 clip V1 V3 V4
			{
				n = 3;
				L[0] = -L[0].z() * L[1] + L[1].z() * L[0];
				L[2] = -L[2].z() * L[1] + L[1].z() * L[2];
			}
			else if (config == 3) // V1 V2 clip V3 V4
			{
				n = 4;
				L[2] = -L[2].z() * L[1] + L[1].z() * L[2];
				L[3] = -L[3].z() * L[0] + L[0].z() * L[3];
			}
			else if (config == 4) // V3 clip V1 V2 V4
			{
				n = 3;
				L[0] = -L[3].z() * L[2] + L[2].z() * L[3];
				L[1] = -L[1].z() * L[2] + L[2].z() * L[1];
			}
			else if (config == 5) // V1 V3 clip V2 V4) impossible
			{
				n = 0;
			}
			else if (config == 6) // V2 V3 clip V1 V4
			{
				n = 4;
				L[0] = -L[0].z() * L[1] + L[1].z() * L[0];
				L[3] = -L[3].z() * L[2] + L[2].z() * L[3];
			}
			else if (config == 7) // V1 V2 V3 clip V4
			{
				n = 5;
				L[4] = -L[3].z() * L[0] + L[0].z() * L[3];
				L[3] = -L[3].z() * L[2] + L[2].z() * L[3];
			}
			else if (config == 8) // V4 clip V1 V2 V3
			{
				n = 3;
				L[0] = -L[0].z() * L[3] + L[3].z() * L[0];
				L[1] = -L[2].z() * L[3] + L[3].z() * L[2];
				L[2] = L[3];
			}
			else if (config == 9) // V1 V4 clip V2 V3
			{
				n = 4;
				L[1] = -L[1].z() * L[0] + L[0].z() * L[1];
				L[2] = -L[2].z() * L[3] + L[3].z() * L[2];
			}
			else if (config == 10) // V2 V4 clip V1 V3) impossible
			{
				n = 0;
			}
			else if (config == 11) // V1 V2 V4 clip V3
			{
				n = 5;
				L[4] = L[3];
				L[3] = -L[2].z() * L[3] + L[3].z() * L[2];
				L[2] = -L[2].z() * L[1] + L[1].z() * L[2];
			}
			else if (config == 12) // V3 V4 clip V1 V2
			{
				n = 4;
				L[1] = -L[1].z() * L[2] + L[2].z() * L[1];
				L[0] = -L[0].z() * L[3] + L[3].z() * L[0];
			}
			else if (config == 13) // V1 V3 V4 clip V2
			{
				n = 5;
				L[4] = L[3];
				L[3] = L[2];
				L[2] = -L[1].z() * L[2] + L[2].z() * L[1];
				L[1] = -L[1].z() * L[0] + L[0].z() * L[1];
			}
			else if (config == 14) // V2 V3 V4 clip V1
			{
				n = 5;
				L[4] = -L[0].z() * L[3] + L[3].z() * L[0];
				L[0] = -L[0].z() * L[1] + L[1].z() * L[0];
			}
			else if (config == 15) // V1 V2 V3 V4
			{
				n = 4;
			}

			if (n == 3)
				L[3] = L[0];
			if (n == 4)
				L[4] = L[0];

			return n;
		}

		__device__ vec4 _evaluateLTC(vec3 norm, vec3 viewDir, vec3 fragPos, mat3 minV, vec3 points[4]) const {
#ifdef __CUDACC__
			const bool clipless = ldg_bool(&_clipless);
			const bool twoSided = ldg_bool(&_twoSided);
#endif

			// construct orthonormal basis around N
			vec3 T1, T2;
			T1 = (viewDir - norm * viewDir.dot(norm));
			if (!epsilon_equal(T1.length_sq(), 0.f, MSTD_EPSILON_SQ)) T1.normalize();
			T2 = norm.cross(T1);

			// rotate area light in (T1, T2, N) basis
			mat3 Minv = minV * mat3(T1, T2, norm).transpose();

			// polygon (allocate 5 vertices for clipping)
			vec3 L[5];
			L[0] = Minv * (points[0] - fragPos);
			L[1] = Minv * (points[1] - fragPos);
			L[2] = Minv * (points[2] - fragPos);
			L[3] = Minv * (points[3] - fragPos);

			// integrate
			float sum = 0.0f;
#ifdef __CUDACC__
			if (clipless) {
#else
			if (_clipless) {
#endif
				vec3 dir = points[0] - fragPos;
				vec3 lightNormal = (points[1] - points[0]).cross(points[3] - points[0]);
				bool behind = (dir.dot(lightNormal) < 0.0f);

				L[0] = epsilon_equal(L[0].length_sq(), 0.f, MSTD_EPSILON_SQ) ? L[0] : L[0].normalized();
				L[1] = epsilon_equal(L[1].length_sq(), 0.f, MSTD_EPSILON_SQ) ? L[1] : L[1].normalized();
				L[2] = epsilon_equal(L[2].length_sq(), 0.f, MSTD_EPSILON_SQ) ? L[2] : L[2].normalized();
				L[3] = epsilon_equal(L[3].length_sq(), 0.f, MSTD_EPSILON_SQ) ? L[3] : L[3].normalized();

				vec3 vsum = vec3::zero();

				vsum += _integrateEdgeVec(L[0], L[1]);
				vsum += _integrateEdgeVec(L[1], L[2]);
				vsum += _integrateEdgeVec(L[2], L[3]);
				vsum += _integrateEdgeVec(L[3], L[0]);

				float len = vsum.length();
				float z = !epsilon_equal(len, 0.f, MSTD_EPSILON) ? vsum.z() / len : 0.f;

				if (behind)
					z = -z;

				vec2 uv = vec2(z * 0.5f + 0.5f, len);
				uv = uv * LUT_SCALE + LUT_BIAS;

				float scale = _ltc_2.sample(uv).w();

				sum = len * scale;

#ifdef __CUDACC__
				if (!behind && !twoSided)
#else
				if (!behind && !_twoSided)
#endif
					sum = 0.0f;
			}
			else
			{
				int n = _clipQuadToHorizon(L);

				if (n == 0)
					return vec3::zero();

				// project onto sphere
				L[0] = epsilon_equal(L[0].length_sq(), 0.f, MSTD_EPSILON_SQ) ? L[0] : L[0].normalized();
				L[1] = epsilon_equal(L[1].length_sq(), 0.f, MSTD_EPSILON_SQ) ? L[1] : L[1].normalized();
				L[2] = epsilon_equal(L[2].length_sq(), 0.f, MSTD_EPSILON_SQ) ? L[2] : L[2].normalized();
				L[3] = epsilon_equal(L[3].length_sq(), 0.f, MSTD_EPSILON_SQ) ? L[3] : L[3].normalized();
				L[4] = epsilon_equal(L[4].length_sq(), 0.f, MSTD_EPSILON_SQ) ? L[4] : L[4].normalized();

				// integrate
				sum += _integrateEdge(L[0], L[1]);
				sum += _integrateEdge(L[1], L[2]);
				sum += _integrateEdge(L[2], L[3]);
				if (n >= 4)
					sum += _integrateEdge(L[3], L[4]);
				if (n == 5)
					sum += _integrateEdge(L[4], L[0]);

#ifdef __CUDACC__
				sum = twoSided ? ::cuda::std::fabsf(sum) : ::cuda::std::max(0.0f, sum);
#else
				sum = _twoSided ? ::std::fabsf(sum) : ::std::max(0.0f, sum);
#endif
			}

			return vec4(sum, sum, sum, 1.0f);
		}

#ifdef __CUDACC__
		__device__ vec3 _randomPoint(curandState* rand_state) const {
			const vec3 pos = ldg_vec3(&_pos);
			float u = curand_uniform(rand_state);
			float v = curand_uniform(rand_state);
#else
		vec3 _randomPoint() const {
			float u = static_cast<float>(rand()) / RAND_MAX;
			float v = static_cast<float>(rand()) / RAND_MAX;
#endif

#ifdef __CUDACC__
			const vec3 p1 = ldg_vec3(&_points[1]);
			const vec3 p3 = ldg_vec3(&_points[3]);
#endif
			// Sp³aszczamy do dwóch trójk¹tów: p0-p1-p3 i p1-p2-p3
			if (u + v < 1.0f) {
#ifdef __CUDACC__
				const vec3 p0 = ldg_vec3(&_points[0]);

				return (p0 + (p1 - p0) * u + (p3 - p0) * v) + pos;
#else
				return (_points[0] + (_points[1] - _points[0]) * u + (_points[3] - _points[0]) * v) + _pos;
#endif
			}
			else {
				u = 1.0f - u;
				v = 1.0f - v;
#ifdef __CUDACC__
				const vec3 p2 = ldg_vec3(&_points[2]);

				return (p2 + (p3 - p2) * u + (p1 - p2) * v) + pos;
#else
				return (_points[2] + (_points[3] - _points[2]) * u + (_points[1] - _points[2]) * v) + _pos;
#endif
			}
		}

	protected:
		__device__ bool _hittedLight(const Ray& ray) const {
			static const int lut[4] = { 1, 2, 0, 1 };

#ifdef __CUDACC__
			const vec3 p0 = ldg_vec3(&_points[0]);

			// lets make v0 the origin
			vec3 a = ldg_vec3(&_points[1]) - p0;
			vec3 b = ldg_vec3(&_points[3]) - p0;
			vec3 c = ldg_vec3(&_points[2]) - p0;
			vec3 p = ray.getOrigin() - p0 - ldg_vec3(&_pos);
#else
			// lets make v0 the origin
			vec3 a = _points[1] - _points[0];
			vec3 b = _points[3] - _points[0];
			vec3 c = _points[2] - _points[0];
			vec3 p = ray.getOrigin() - _points[0] - _pos;
#endif

			// intersect plane
			vec3 nor = a.cross(b);
			float t = -p.dot(nor) / ray.getDirection().dot(nor);
			if (t < 0.0f || (ray.getDistance() > 0.0f && t > ray.getDistance())) return { false };

			// intersection point
			vec3 pos = p + t * ray.getDirection();

			// select projection plane
#ifdef __CUDACC__
			vec3 mor = vec3(::cuda::std::fabsf(nor.x()), ::cuda::std::fabsf(nor.y()), ::cuda::std::fabsf(nor.z()));
#else
			vec3 mor = vec3(::std::fabsf(nor.x()), ::std::fabsf(nor.y()), ::std::fabsf(nor.z()));
#endif
			int id = (mor.x() > mor.y() && mor.x() > mor.z()) ? 0 : (mor.y() > mor.z()) ? 1 : 2;

			int idu = lut[id];
			int idv = lut[id + 1];

			// project to 2D
			vec2 kp = vec2(pos[idu], pos[idv]);
			vec2 ka = vec2(a[idu], a[idv]);
			vec2 kb = vec2(b[idu], b[idv]);
			vec2 kc = vec2(c[idu], c[idv]);

			// find barycentric coords of the quadrilateral
			vec2 kg = kc - kb - ka;

			float k0 = kp.x() * kb.y() - kp.y() * kb.x();
			vec2 kcb = kc - kb;
			float k2 = kcb.x() * ka.y() - kcb.y() * ka.x();             // float k2 = cross2d( kg, ka );
			float k1 = (kp.x() * kg.y() - kp.y() * kg.x()) - nor[id];   // float k1 = cross2d( kb, ka ) + cross2d( kp, kg );

			// if edges are parallel, this is a linear equation
			float u, v;
			if (epsilon_equal(k2, 0.f, MSTD_EPSILON))
			{
				v = -k0 / k1;
				u = (kp.x() * ka.y() - kp.y() * ka.x()) / k1;
			}
			else
			{
				// otherwise, it's a quadratic
				float w = k1 * k1 - 4.0f * k0 * k2;
				if (w < 0.0f) return { false };
#ifdef __CUDACC__
				w = 1.0f / rsqrtf(w);
#else
				w = ::std::sqrtf(w);
#endif

				float ik2 = 1.0f / (2.0f * k2);

				v = (-k1 - w) * ik2;
				if (v < 0.0f || v > 1.0f) v = (-k1 + w) * ik2;

				u = (kp.x() - ka.x() * v) / (kb.x() + kg.x() * v);
			}

#ifdef __CUDACC__
			if (::cuda::std::min(u, v) < 0.0f || ::cuda::std::max(u, v) > 1.0f)
#else
			if (::std::min(u, v) < 0.0f || ::std::max(u, v) > 1.0f)
#endif
				return false;

			return true;
		}

#ifdef __CUDACC__
		__device__ bool _calculateVisibility(const vec3& position, Geometry* objects, float& visibility, curandState* rand_state) const {
#else
		bool _calculateVisibility(const vec3& position, Geometry* objects, float& visibility) const {
#endif
			if (!_initialized) return false;

#ifdef __CUDACC__
			const unsigned int shadowSamples = ldg_uint(&_shadowSamples);
#endif

			unsigned int shadowed = 0;
			bool hit_any = false;

#ifdef __CUDACC__
			for (unsigned int i = 0u; i < shadowSamples; ++i) {
				vec3 lightSample = _randomPoint(rand_state);
#else
			for (unsigned int i = 0u; i < _shadowSamples; ++i) {
				vec3 lightSample = _randomPoint();
#endif
				vec3 rayDir = lightSample - position;
				float lightDist = rayDir.length();

				if (!epsilon_equal(lightDist, 0.f, MSTD_EPSILON)) rayDir.normalize();

				Ray shadowRay = Ray(position + 0.01f * rayDir, rayDir, lightDist);

				Ray ray = Ray(position + 0.01f * rayDir, rayDir, lightDist);
				bool hitObject = _isCovered(ray, objects, 10u);
				if (hitObject) {
					hit_any = true;
					++shadowed;
				}
			}

#ifdef __CUDACC__
			visibility = 1.0f - static_cast<float>(shadowed) / static_cast<float>(shadowSamples);
#else
			visibility = 1.0f - static_cast<float>(shadowed) / static_cast<float>(_shadowSamples);
#endif
			return hit_any;
		}

	public:
		AreaLight() = default;

		__device__ AreaLight(vec3 p0, vec3 p1, vec3 p2, vec3 p3, vec4 color, unsigned int shadowSamples = 10u, float intensity = 1.f, bool twoSided = false, VERTEX_ORDER order = VERTEX_ORDER::CCW)
			: Light(vec3::zero(), color, intensity), _twoSided(twoSided), _shadowSamples(shadowSamples) {
			switch (order) {
				case VERTEX_ORDER::CCW: {
					_points[0] = p0;
					_points[1] = p3;
					_points[2] = p2;
					_points[3] = p1;
					break;
				}
				case VERTEX_ORDER::CW: {
					_points[0] = p0;
					_points[1] = p1;
					_points[2] = p2;
					_points[3] = p3;
					break;
				}
			}

			_initLTCTextures();
			_defineCenterPoint();
			_initialized = true;
		}

		__device__ void rotate(vec3 axis, float radians) {
			if (epsilon_equal(axis.length_sq(), 0.f, MSTD_EPSILON_SQ)) return;
			vec3 rotAxis = axis.normalized();
			for (vec3& pt : _points) {
				pt.rotate(rotAxis, radians);
			}
		}

#ifdef __CUDACC__
		__device__ vec4 calculateColor(const LightInput& input, Geometry* objects, curandState* rand_state) const {
			const vec3 lcol = vec3(ldg_vec4(&_color));
			const float intensity = ldg_float(&_intensity);
			const vec3 pos = ldg_vec3(&_pos);

			// ambient
			const vec3 ambient = ldg_vec4(&input.mat->ambient) * lcol;
#else
		vec4 calculateColor(const LightInput& input, Geometry* objects) const {
			const vec3 lcol = _color;

			// ambient
			const vec3 ambient = input.mat->ambient * lcol;
#endif

			if (!_initialized) return vec4(ambient, 1.f);

			float visibility = 1.0f;
#ifdef __CUDACC__
			_calculateVisibility(input.fragPos, objects, visibility, rand_state);
#else
			_calculateVisibility(input.fragPos, objects, visibility);
#endif

			vec3 points[4];
			for (size_t i = 0; i < 4; ++i) {
#ifdef __CUDACC__
				points[i] = pos + ldg_vec3(&_points[i]);
#else
				points[i] = _pos + _points[i];
#endif
			}

#ifdef __CUDACC__
			const vec3 dcol = ldg_vec4(&input.mat->diffuse);
			const vec3 scol = ldg_vec4(&input.mat->specular);

			float roughness = 1.0f - 0.25f * ::cuda::std::powf(ldg_float(&input.mat->shininess), 0.2f);
#else
			const vec3 dcol = input.mat->diffuse;
			const vec3 scol = input.mat->specular;

			float roughness = 1.0f - 0.25f * ::std::powf(input.mat->shininess, 0.2f);
#endif

			float ndotv = saturate(input.norm.dot(input.viewDir));

#ifdef __CUDACC__
			vec2 uv = vec2(roughness, 1.0f / rsqrtf(1.0f - ndotv));
#else
			vec2 uv = vec2(roughness, ::std::sqrtf(1.0f - ndotv));
#endif
			uv = uv * LUT_SCALE + LUT_BIAS;

			vec4 t1 = _ltc_1.sample(uv);
			vec4 t2 = _ltc_2.sample(uv);

			// Column Major
			mat3 Minv = mat3(
				vec3(t1.x(), 0.f, t1.y()),
				vec3(0.f, 1.f, 0.f),
				vec3(t1.z(), 0.f, t1.w())
			);

			// diffuse
			vec3 diff = _evaluateLTC(input.norm, input.viewDir, input.fragPos, mat3::identity(), points);
			vec3 diffuse = lcol * diff * dcol;

			// specular
			vec3 spec = _evaluateLTC(input.norm, input.viewDir, input.fragPos, Minv, points);

			// BRDF shadowing and Fresnel
			spec *= scol * t2.x() + (vec3::one() - scol) * t2.y();
			vec3 specular = lcol * spec;

			vec3 result = ambient + (diffuse + specular) * intensity * visibility;
			return vec4(result, 1.0f);
		}
	};
}