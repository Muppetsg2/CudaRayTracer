/*****************************************************************
 *                                                               *
 *  Project:   MSTD                                              *
 *  Authors:   MAIPA01                                           *
 *  License:   BSD 3-Clause License with Attribution Requirement *
 *                                                               *
 *****************************************************************/

#pragma once
#include <cstdint>

#if defined(__CUDACC__) && !defined(MSTD_NO_CUDA)
	#define MSTD_USE_CUDA
#endif

#ifdef MSTD_USE_CUDA
#include <thrust/fill.h>
#include <thrust/equal.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <cuda_runtime.h>
#define MSTD_FRIEND friend
#define MSTD_NAMESPACE cmstd
#define MSTD_STD_NAMESPACE ::cuda::std
#define MSTD_CLAMP(x, mn, mx) MSTD_STD_NAMESPACE::max((mn), MSTD_STD_NAMESPACE::min((mx), (x)))
#define MSTD_CUDA_EXPR __device__
#else
#include <string>
#include <ostream>
#include <iomanip>
#include <type_traits>
#include <math.h>
#include <algorithm>
#include <utility>
#define MSTD_FRIEND friend static
#define MSTD_CUDA_EXPR
#define MSTD_NAMESPACE mstd
#define MSTD_STD_NAMESPACE ::std
#define MSTD_CLAMP(x, mn, mx) ::std::clamp(x, mn, mx)
#endif

namespace MSTD_NAMESPACE {

	template<typename T>
	static constexpr T MSTD_PI = static_cast<T>(3.14159265358979323846); // pi

	template<typename T>
	static constexpr T MSTD_1_PI = static_cast<T>(0.31830988618379067153); // 1/pi

	template<typename T>
	static constexpr T MSTD_1_PI_2 = static_cast<T>(0.15915494309189533576); // 1/(2*pi)

	template<typename T>
	static constexpr T MSTD_HALF_PI = static_cast<T>(1.57079632679489661923); // pi/2

	template<typename T>
	static constexpr T MSTD_PI_2 = static_cast<T>(6.28318530717958647692); // 2*pi

	template<typename T>
	static constexpr T MSTD_EPSILON = static_cast<T>(0.0001);

	template<typename T>
	static constexpr T MSTD_EPSILON_SQ = static_cast<T>(0.00000001);

	template<typename T>
	static constexpr T MSTD_DEG_TO_RAD = static_cast<T>(0.01745329251994329576); // 180/pi

	template<typename T>
	static constexpr T MSTD_RAD_TO_DEG = static_cast<T>(57.29577951308232087679); // pi/180

#ifdef MSTD_USE_CUDA

	__device__ __constant__ float MSTD_CUDA_PI = 3.14159265358979323846f; // pi
	__device__ __constant__ float MSTD_CUDA_1_PI = 0.31830988618379067153f; // 1/pi
	__device__ __constant__ float MSTD_CUDA_1_PI_2 = 0.15915494309189533576f; // 1/(2*pi)
	__device__ __constant__ float MSTD_CUDA_HALF_PI = 1.57079632679489661923f; // pi/2
	__device__ __constant__ float MSTD_CUDA_PI_2 = 6.28318530717958647692f; // 2*pi
	__device__ __constant__ float MSTD_CUDA_EPSILON = 1e-4f;
	__device__ __constant__ float MSTD_CUDA_EPSILON_SQ = 1e-8f;
	__device__ __constant__ float MSTD_CUDA_DEG_TO_RAD = 0.01745329251994329576f; // 180/pi
	__device__ __constant__ float MSTD_CUDA_RAD_TO_DEG = 57.29577951308232087679f; // pi/180

#endif

#if _HAS_CXX20
	template<class T>
	concept arithmetic = MSTD_STD_NAMESPACE::is_arithmetic_v<T>;

	template<size_t N, arithmetic T>
		requires (N > 0)
	class vec;

	template<arithmetic T = float>
	class quat;

	template<size_t C, size_t R, arithmetic T>
		requires (C > 0 && R > 0)
	class mat;
#else
	template<size_t N, class T, MSTD_STD_NAMESPACE::enable_if_t<(N > 0 && MSTD_STD_NAMESPACE::is_arithmetic_v<T>), bool> = true>
		class vec;

	template<class T = float, MSTD_STD_NAMESPACE::enable_if_t<MSTD_STD_NAMESPACE::is_arithmetic_v<T>, bool> = true>
	class quat;

	template<size_t C, size_t R, class T,
		MSTD_STD_NAMESPACE::enable_if_t<(C > 0 && R > 0 && MSTD_STD_NAMESPACE::is_arithmetic_v<T>), bool> = true>
		class mat;
#endif
}