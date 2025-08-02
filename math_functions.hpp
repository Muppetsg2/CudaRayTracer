#pragma once
#ifndef MSTD_USE_CUDA
#include <bit>
#endif
#include "math_types.hpp"

namespace MSTD_NAMESPACE {
	template<class T>
	MSTD_CUDA_EXPR static constexpr T signum(const T& x) noexcept {
		if constexpr (MSTD_STD_NAMESPACE::is_signed_v<T>) {
			return (T(0) < x) - (x < T(0));
		}
		else {
			return T(0) < x;
		}
	}

	template<class T>
	MSTD_CUDA_EXPR static constexpr T step(const T& edge, const T& x) noexcept {
		return x < edge ? T(0) : T(1);
	}

	template<class T>
	MSTD_CUDA_EXPR static constexpr T remap(const T& input, const T& currStart, const T& currEnd, const T& expectedStart, const T& expectedEnd) noexcept {
		return expectedStart + ((expectedEnd - expectedStart) / (currEnd - currStart)) * (input - currStart);
	}

	template<class T>
	MSTD_CUDA_EXPR static constexpr T deg_to_rad(const T& angle) noexcept {
		return angle * (static_cast<T>(MSTD_PI) / static_cast<T>(180));
	}

	template<class T>
	MSTD_CUDA_EXPR static constexpr T rad_to_deg(const T& rad) noexcept {
		return rad * (static_cast<T>(180) / static_cast<T>(MSTD_PI));
	}

	template<class T>
	MSTD_CUDA_EXPR static bool epsilon_equal(const T& a, const T& b, const T& epsilon) noexcept {
		return MSTD_STD_NAMESPACE::abs(a - b) < epsilon;
	}

	template<class T, bool cuda_version = false>
	MSTD_CUDA_EXPR static constexpr T saturate(const T& a) noexcept {
		if constexpr (cuda_version && MSTD_STD_NAMESPACE::is_same_v<T, float>) {
			return __saturatef(a);
		}
		else {
			return MSTD_CLAMP(a, T(0), T(1));
		}
	}

	template<class T>
	MSTD_CUDA_EXPR static T fract(const T& x) {
		return x - (T)MSTD_STD_NAMESPACE::floor(x);
	}

	MSTD_CUDA_EXPR static constexpr float Q_rsqrtf(float number) noexcept
	{
		const auto y = MSTD_STD_NAMESPACE::bit_cast<float>(
			0x5f3759df - (MSTD_STD_NAMESPACE::bit_cast<uint32_t>(number) >> 1));
		return y * (1.5f - (number * 0.5f * y * y));
	}

	MSTD_CUDA_EXPR static float reflectance(float cosine, float refraction_index) noexcept {
		// Use Schlick's approximation for reflectance.
		float r0 = (1.0f - refraction_index) / (1.0f + refraction_index);
		r0 = r0 * r0;
#ifdef MSTD_USE_CUDA
		return r0 + (1.0f - r0) * __powf(1.0f - cosine, 5.0f);
#else
		return r0 + (1.0f - r0) * MSTD_STD_NAMESPACE::expf(MSTD_STD_NAMESPACE::logf(1.0f - cosine) * 5.0f);
#endif
	}
}