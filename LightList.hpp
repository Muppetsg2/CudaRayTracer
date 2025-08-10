/**************************************************************
 *                                                            *
 *  Project:   CudaRayTracer                                  *
 *  Authors:   Muppetsg2 & MAIPA01                            *
 *  License:   MIT License                                    *
 *  Last Update: 10.08.2025                                   *
 *                                                            *
 **************************************************************/

#pragma once
#include "Light.hpp"

namespace craytracer {
    class LightList : public Light {
    private:
        Light** list;
        int list_size;
    public:
        LightList() = default;
        __device__ LightList(Light** l, int n) { list = l; list_size = n; }

#ifdef __CUDACC__
		__device__ vec4 calculateColor(const LightInput& input, Geometry* objects, curandState* rand_state) const {
#else
        vec4 calculateColor(const LightInput& input, Geometry* objects) const {
#endif
            vec4 color = vec4::zero();
            for (int i = 0; i < list_size; ++i) {
#ifdef __CUDACC__
                color += list[i]->calculateColor(input, objects, rand_state);
#else
                color += list[i]->calculateColor(input, objects);
#endif
            }
            return color;
		}
    };
}