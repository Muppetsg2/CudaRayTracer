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
#include "Vertex.hpp"
#include "Geometry.hpp"
#include "math_functions.hpp"
#include "ldg_helpers.hpp"

using namespace MSTD_NAMESPACE;

namespace craytracer {
	class Quad : public Geometry {
	private:
		Vertex _v0;
		Vertex _v1;
		Vertex _v2;
		Vertex _v3;

#ifdef __CUDACC__
        __device__ ::thrust::tuple<vec2, vec2, vec2, vec2> _getUVs(vec3 v0, vec3 v1, vec3 v2, vec3 v3, vec3 normal) {

            float (*cabs)(float) = ::cuda::std::fabsf;
#else
        __device__ ::std::tuple<vec2, vec2, vec2, vec2> _getUVs(vec3 v0, vec3 v1, vec3 v2, vec3 v3, vec3 normal) {

            float (*cabs)(float) = ::std::fabsf;
#endif
            // Selecting the best plane based on the normal
            vec2(*project)(const vec3&) = nullptr;
            if (cabs(normal.x()) > cabs(normal.y()) && cabs(normal.x()) > cabs(normal.z())) {
                // YZ projection
                project = [](const vec3& v) { return vec2(v.y(), v.z()); };
            }
            else if (cabs(normal.y()) > cabs(normal.z())) {
                // XZ projection
                project = [](const vec3& v) { return vec2(v.x(), v.z()); };
            }
            else {
                // XY projection
                project = [](const vec3& v) { return vec2(v.x(), v.y()); };
            }

            // Transforming Vertices to Selected UV Space
            vec2 uv0 = project(v0);
            vec2 uv1 = project(v1);
            vec2 uv2 = project(v2);
            vec2 uv3 = project(v3);

            // Calculation of minimum and maximum UV values
#ifdef __CUDACC__
            float minU = ::cuda::std::min({ uv0.x(), uv1.x(), uv2.x(), uv3.x() });
            float maxU = ::cuda::std::max({ uv0.x(), uv1.x(), uv2.x(), uv3.x() });
            float minV = ::cuda::std::min({ uv0.y(), uv1.y(), uv2.y(), uv3.y() });
            float maxV = ::cuda::std::max({ uv0.y(), uv1.y(), uv2.y(), uv3.y() });
#else
            float minU = ::std::min({ uv0.x(), uv1.x(), uv2.x(), uv3.x() });
            float maxU = ::std::max({ uv0.x(), uv1.x(), uv2.x(), uv3.x() });
            float minV = ::std::min({ uv0.y(), uv1.y(), uv2.y(), uv3.y() });
            float maxV = ::std::max({ uv0.y(), uv1.y(), uv2.y(), uv3.y() });
#endif

            float rangeU = maxU - minU;
            float rangeV = maxV - minV;

            // Preventing division by zero
#ifdef __CUDACC__
            if (epsilon_equal(rangeU, 0.f, MSTD_CUDA_EPSILON)) rangeU = 1.0f;
            if (epsilon_equal(rangeV, 0.f, MSTD_CUDA_EPSILON)) rangeV = 1.0f;
#else
            if (epsilon_equal(rangeU, 0.f, MSTD_EPSILON<float>)) rangeU = 1.0f;
            if (epsilon_equal(rangeV, 0.f, MSTD_EPSILON<float>)) rangeV = 1.0f;
#endif

            // UV normalization to range [0,1]
#ifdef __CUDACC__
            ::thrust::tuple<vec2, vec2, vec2, vec2> ret = ::thrust::make_tuple(
                vec2(__fdividef((uv0.x() - minU), rangeU), __fdividef((uv0.y() - minV), rangeV)),
                vec2(__fdividef((uv1.x() - minU), rangeU), __fdividef((uv1.y() - minV), rangeV)),
                vec2(__fdividef((uv2.x() - minU), rangeU), __fdividef((uv2.y() - minV), rangeV)),
                vec2(__fdividef((uv3.x() - minU), rangeU), __fdividef((uv3.y() - minV), rangeV))
            );
#else
            ::std::tuple<vec2, vec2, vec2, vec2> ret = ::std::make_tuple(
                vec2((uv0.x() - minU) / rangeU, (uv0.y() - minV) / rangeV),
                vec2((uv1.x() - minU) / rangeU, (uv1.y() - minV) / rangeV),
                vec2((uv2.x() - minU) / rangeU, (uv2.y() - minV) / rangeV),
                vec2((uv3.x() - minU) / rangeU, (uv3.y() - minV) / rangeV)
            );
#endif
            return ret;
		}

	public:
		Quad() = default;

        __device__ Quad(vec3 v0, vec3 v1, vec3 v2, vec3 v3) {
            _v0.pos = v0;
            _v1.pos = v1;
            _v2.pos = v2;
            _v3.pos = v3;

            vec3 normal = ((v1 - v0).cross(v2 - v0)).normalize();
            _v0.normal = normal;
            _v1.normal = normal;
            _v2.normal = normal;
            _v3.normal = normal;

#ifdef __CUDACC__
            ::thrust::tuple<vec2, vec2, vec2, vec2> uvs = _getUVs(v0, v1, v2, v3, normal);

            _v0.tex = ::thrust::get<0>(uvs);
            _v1.tex = ::thrust::get<1>(uvs);
            _v2.tex = ::thrust::get<2>(uvs);
            _v3.tex = ::thrust::get<3>(uvs);
#else
            ::std::tuple<vec2, vec2, vec2, vec2> uvs = _getUVs(v0, v1, v2, v3, normal);

            _v0.tex = ::std::get<0>(uvs);
            _v1.tex = ::std::get<1>(uvs);
            _v2.tex = ::std::get<2>(uvs);
            _v3.tex = ::std::get<3>(uvs);
#endif
        }

        __device__ Quad(vec3 v0, vec3 v1, vec3 v2, vec3 v3, vec2 vt0, vec2 vt1, vec2 vt2, vec2 vt3) {
            _v0.pos = v0;
            _v1.pos = v1;
            _v2.pos = v2;
            _v3.pos = v3;

            vec3 normal = ((v1 - v0).cross(v2 - v0)).normalize();
            _v0.normal = normal;
            _v1.normal = normal;
            _v2.normal = normal;
            _v3.normal = normal;

            _v0.tex = vt0;
            _v1.tex = vt1;
            _v2.tex = vt2;
            _v3.tex = vt3;
        }

        __device__ Quad(vec3 v0, vec3 v1, vec3 v2, vec3 v3, vec3 vn0, vec3 vn1, vec3 vn2, vec3 vn3) {
            _v0.pos = v0;
            _v1.pos = v1;
            _v2.pos = v2;
            _v3.pos = v3;

            _v0.normal = vn0;
            _v1.normal = vn1;
            _v2.normal = vn2;
            _v3.normal = vn3;

            vec3 normal = ((v1 - v0).cross(v2 - v0)).normalize();

#ifdef __CUDACC__
            ::thrust::tuple<vec2, vec2, vec2, vec2> uvs = _getUVs(v0, v1, v2, v3, normal);

            _v0.tex = ::thrust::get<0>(uvs);
            _v1.tex = ::thrust::get<1>(uvs);
            _v2.tex = ::thrust::get<2>(uvs);
            _v3.tex = ::thrust::get<3>(uvs);
#else
            ::std::tuple<vec2, vec2, vec2, vec2> uvs = _getUVs(v0, v1, v2, v3, normal);

            _v0.tex = ::std::get<0>(uvs);
            _v1.tex = ::std::get<1>(uvs);
            _v2.tex = ::std::get<2>(uvs);
            _v3.tex = ::std::get<3>(uvs);
#endif
        }

		__device__ Quad(Vertex v0, Vertex v1, Vertex v2, Vertex v3) : _v0(v0), _v1(v1), _v2(v2), _v3(v3) {}

#ifdef __CUDACC__
        __device__ ::thrust::tuple<Vertex, Vertex, Vertex, Vertex> getVertices() const
        {
            return ::thrust::make_tuple(_v0, _v1, _v2, _v3);
        }
#else
        __device__ ::std::tuple<Vertex, Vertex, Vertex, Vertex> getVertices() const
        {
            return ::std::make_tuple(_v0, _v1, _v2, _v3);
        }
#endif

		__device__ bool hit(const Ray& ray, RayHit& hit) const {
            static const int lut[4] = { 1, 2, 0, 1 };

#ifdef __CUDACC__
            float (*cabs)(float) = ::cuda::std::fabsf;

            const vec3 v0p = ldg_vec3(&_v0.pos);

            // lets make v0 the origin
            vec3 a = ldg_vec3(&_v1.pos) - v0p;
            vec3 b = ldg_vec3(&_v3.pos) - v0p;
            vec3 c = ldg_vec3(&_v2.pos) - v0p;
            vec3 p = ray.getOrigin() - v0p;
#else
            float (*cabs)(float) = ::std::fabsf;

            // lets make v0 the origin
            vec3 a = _v1.pos - _v0.pos;
            vec3 b = _v3.pos - _v0.pos;
            vec3 c = _v2.pos - _v0.pos;
            vec3 p = ray.getOrigin() - _v0.pos;
#endif

            // intersect plane
            vec3 nor = a.cross(b);
#ifdef __CUDACC__
            float t = -__fdividef(p.dot(nor), ray.getDirection().dot(nor));
#else
            float t = -p.dot(nor) / ray.getDirection().dot(nor);
#endif
            if (t < 0.0f || (ray.getDistance() > 0.0f && t > ray.getDistance())) return false;

            // intersection point
            vec3 pos = p + t * ray.getDirection();

            // select projection plane
            vec3 mor = vec3(cabs(nor.x()), cabs(nor.y()), cabs(nor.z()));
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
#ifdef __CUDACC__
            if (epsilon_equal(k2, 0.f, MSTD_CUDA_EPSILON))
            {
                v = -__fdividef(k0, k1);
                u = __fdividef((kp.x() * ka.y() - kp.y() * ka.x()), k1);
#else
            if (epsilon_equal(k2, 0.f, MSTD_EPSILON<float>))
            {
                v = -k0 / k1;
                u = (kp.x() * ka.y() - kp.y() * ka.x()) / k1;
#endif
            }
            else
            {
                // otherwise, it's a quadratic
                float w = k1 * k1 - 4.0f * k0 * k2;
                if (w < 0.0f) return false;
#ifdef __CUDACC__
                w = __fdividef(1.0f, rsqrtf(w));

                float ik2 = __fdividef(1.0f, (2.0f * k2));
#else
                w = ::std::sqrtf(w);

                float ik2 = 1.0f / (2.0f * k2);
#endif

                v = (-k1 - w) * ik2;
                if (v < 0.0f || v > 1.0f) v = (-k1 + w) * ik2;

#ifdef __CUDACC__
                u = __fdividef((kp.x() - ka.x() * v), (kb.x() + kg.x() * v));
#else
                u = (kp.x() - ka.x() * v) / (kb.x() + kg.x() * v);
#endif
            }

#ifdef __CUDACC__
            if (::cuda::std::min(u, v) < 0.0f || ::cuda::std::max(u, v) > 1.0f) return false;
#else
            if (::std::min(u, v) < 0.0f || ::std::max(u, v) > 1.0f) return false;
#endif

            // RayHit
            hit.isHit = true;
            hit.hitPoint = ray.getOrigin() + ray.getDirection() * t;
            hit.hitMat = &_mat;
            hit.hitDist = t;

            // Interpolate UV and Normal
            float l0 = (1 - u) * (1 - v);
            float l1 = u * (1 - v);
            float l2 = u * v;
            float l3 = (1 - u) * v;

#ifdef __CUDACC__
            vec3 n0 = ldg_vec3(&_v0.normal);
            vec3 n1 = ldg_vec3(&_v1.normal);
            vec3 n2 = ldg_vec3(&_v2.normal);
            vec3 n3 = ldg_vec3(&_v3.normal);
#else
            vec3 n0 = _v0.normal;
            vec3 n1 = _v1.normal;
            vec3 n2 = _v2.normal;
            vec3 n3 = _v3.normal;
#endif

            hit.hitNormal = vec3(
                l0 * n0.x() + l1 * n1.x() + l2 * n2.x() + l3 * n3.x(),
                l0 * n0.y() + l1 * n1.y() + l2 * n2.y() + l3 * n3.y(),
                l0 * n0.z() + l1 * n1.z() + l2 * n2.z() + l3 * n3.z()
            );

#ifdef __CUDACC__
            vec2 t0 = ldg_vec2(&_v0.tex);
            vec2 t1 = ldg_vec2(&_v1.tex);
            vec2 t2 = ldg_vec2(&_v2.tex);
            vec2 t3 = ldg_vec2(&_v3.tex);
#else
            vec2 t0 = _v0.tex;
            vec2 t1 = _v1.tex;
            vec2 t2 = _v2.tex;
            vec2 t3 = _v3.tex;
#endif

            hit.hitUV = vec2(
                l0 * t0.x() + l1 * t1.x() + l2 * t2.x() + l3 * t3.x(),
                l0 * t0.y() + l1 * t1.y() + l2 * t2.y() + l3 * t3.y()
            );

            return true;
		}
	};
}
