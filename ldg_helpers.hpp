/**************************************************************
 *                                                            *
 *  Project:   CudaRayTracer                                  *
 *  Authors:   Muppetsg2 & MAIPA01                            *
 *  License:   MIT License                                    *
 *  Last Update: 10.08.2025                                   *
 *                                                            *
 **************************************************************/

#pragma once

#ifdef __CUDACC__
namespace craytracer {
	__device__ __forceinline__ bool ldg_bool(const bool* ptr) {
		return static_cast<bool>(__ldg(reinterpret_cast<const unsigned char*>(ptr)));
	}

	__device__ __forceinline__ float ldg_float(const float* ptr) {
		return __ldg(ptr);
	}

	__device__ __forceinline__ unsigned int ldg_uint(const unsigned int* ptr) {
		return __ldg(ptr);
	}

	__device__ __forceinline__ vec2 ldg_vec2(const vec2* ptr) {
		const float* fptr = reinterpret_cast<const float*>(ptr);
		return vec2(__ldg(&fptr[0]), __ldg(&fptr[1]));
	}

	__device__ __forceinline__ vec3 ldg_vec3(const vec3* ptr) {
		const float* fptr = reinterpret_cast<const float*>(ptr);
		return vec3(__ldg(&fptr[0]), __ldg(&fptr[1]), __ldg(&fptr[2]));
	}

	__device__ __forceinline__ vec4 ldg_vec4(const vec4* ptr) {
		const float* fptr = reinterpret_cast<const float*>(ptr);
		return vec4(__ldg(&fptr[0]), __ldg(&fptr[1]), __ldg(&fptr[2]), __ldg(&fptr[3]));
	}

	template<class T>
	__device__ __forceinline__ T ldg_enum_uint8_type(const T* ptr) {
		return static_cast<T>(__ldg(reinterpret_cast<const uint8_t*>(ptr)));
	}
}
#endif