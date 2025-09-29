/**************************************************************
 *                                                            *
 *  Project:   CudaRayTracer                                  *
 *  Authors:   Muppetsg2 & MAIPA01                            *
 *  License:   MIT License                                    *
 *  Last Update: 28.09.2025                                   *
 *                                                            *
 **************************************************************/

#pragma once
#include "vec.hpp"
#include "Ray.hpp"

using namespace MSTD_NAMESPACE;

namespace craytracer {
    struct AABB {
        vec3 min;
        vec3 max;

        __host__ __device__ AABB() {
            min = vec3(FLT_MAX, FLT_MAX, FLT_MAX);
            max = vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        }

        __host__ __device__ void expand(const vec3& p) {
            if (p.x() < min.x()) min.x() = p.x();
            if (p.y() < min.y()) min.y() = p.y();
            if (p.z() < min.z()) min.z() = p.z();
            if (p.x() > max.x()) max.x() = p.x();
            if (p.y() > max.y()) max.y() = p.y();
            if (p.z() > max.z()) max.z() = p.z();
        }

        __host__ __device__ void expand(const AABB& b) {
            expand(b.min); expand(b.max);
        }

        __host__ __device__ float extent(int axis) const {
            if (axis == 0) return max.x() - min.x();
            if (axis == 1) return max.y() - min.y();
            return max.z() - min.z();
        }

        __host__ __device__ float surfaceArea() const {
            float dx = max.x() - min.x();
            float dy = max.y() - min.y();
            float dz = max.z() - min.z();
            if (dx < 0 || dy < 0 || dz < 0) return 0.0f;
            return 2.0f * (dx * dy + dx * dz + dy * dz);
        }

        __device__ bool hit(const Ray& ray, float& out_tmin, float& out_tmax) const {
            float tmin = 0.0f;
            float tmax = FLT_MAX;

            const vec3 origin = ray.getOrigin();
            const vec3 direction = ray.getDirection();

            for (int axis = 0; axis < 3; ++axis) {
                float origin_axis = (axis == 0 ? origin.x() : (axis == 1 ? origin.y() : origin.z()));
                float dir_axis = (axis == 0 ? direction.x() : (axis == 1 ? direction.y() : direction.z()));
                float min_axis = (axis == 0 ? min.x() : (axis == 1 ? min.y() : min.z()));
                float max_axis = (axis == 0 ? max.x() : (axis == 1 ? max.y() : max.z()));

                if (epsilon_equal(dir_axis, 0.0f, MSTD_EPSILON<float>)) {
                    if (origin_axis < min_axis || origin_axis > max_axis) return false;
                    else continue;
                }

                float invD = 1.0f / dir_axis;

                float t0 = (min_axis - origin_axis) * invD;
                float t1 = (max_axis - origin_axis) * invD;
                if (t0 > t1) { float tmp = t0; t0 = t1; t1 = tmp; }

                if (t0 > tmin) tmin = t0;
                if (t1 < tmax) tmax = t1;

                if (tmax <= tmin) return false;
            }

            out_tmin = tmin;
            out_tmax = tmax;
            return true;
        }
    };
}