#pragma once
#include <cstdint>

#define MSTD_PI      3.14159265358979323846
#define MSTD_EPSILON 1e-4f
#define MSTD_EPSILON_SQ 1e-8f

#if defined(__CUDACC__) && !defined(MSTD_NO_CUDA)
	#define MSTD_USE_CUDA
#endif

#ifdef MSTD_USE_CUDA
#include <thrust/fill.h>
#include <thrust/equal.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
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