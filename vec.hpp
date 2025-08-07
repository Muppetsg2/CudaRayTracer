#pragma once
#include "math_types.hpp"
#include "math_functions.hpp"

namespace MSTD_NAMESPACE {
#pragma region CONSTEXPR
	template<template<class> class Test, class... Ts>
	static constexpr const bool are_all_v = (Test<Ts>::value && ...);

	template<size_t Start, size_t... Indices>
	constexpr MSTD_STD_NAMESPACE::index_sequence<(Start + Indices)...> shift_index_sequence(MSTD_STD_NAMESPACE::index_sequence<Indices...>) {
		return {};
	}

	template<size_t Start, size_t End>
	using make_index_sequence_from_to = decltype(shift_index_sequence<Start>(MSTD_STD_NAMESPACE::make_index_sequence<End - Start>()));

	template<size_t Start, size_t Size>
	using make_index_sequence_from = decltype(shift_index_sequence<Start>(MSTD_STD_NAMESPACE::make_index_sequence<Size>()));

	template<size_t Start, class... Ts>
	using make_index_sequence_for_from = decltype(shift_index_sequence<Start>(MSTD_STD_NAMESPACE::index_sequence_for<Ts...>()));
#pragma endregion // CONSTEXPR

#if _HAS_CXX20
	template<size_t N, arithmetic T>
		requires (N > 0)
#else
	template<size_t N, class T, MSTD_STD_NAMESPACE::enable_if_t<(N > 0 && MSTD_STD_NAMESPACE::is_arithmetic_v<T>), bool>>
#endif
	class vec {
	public:
		static constexpr const size_t size = N;
		using value_type = T;

	private:
		T _values[N] = {};

#pragma region PRIVATE_METHODS
#if _HAS_CXX20
		template<arithmetic... Ts, size_t... Idxs>
#else
		template<class... Ts, size_t... Idxs>
#endif
		MSTD_CUDA_EXPR constexpr void _set_values(const MSTD_STD_NAMESPACE::index_sequence<Idxs...>&, const Ts&... values) {
			((_values[Idxs] = (T)values), ...);
		}

		MSTD_CUDA_EXPR constexpr void _fill_values(const T& value) {
#ifdef MSTD_USE_CUDA
			::thrust::fill_n(::thrust::device, &_values[0], N, value);
#else
			MSTD_STD_NAMESPACE::fill_n(&_values[0], N, value);
#endif
		}

		MSTD_CUDA_EXPR constexpr void _fill_values_from(size_t first_idx, const T& value) {
			if (first_idx >= N) return;
#ifdef MSTD_USE_CUDA
			::thrust::fill_n(::thrust::device, &_values[0] + first_idx, N - first_idx, value);
#else
			MSTD_STD_NAMESPACE::fill_n(&_values[0] + first_idx, N - first_idx, value);
#endif
		}

#if _HAS_CXX20
		template<arithmetic OT>
#else
		template<class OT>
#endif
		MSTD_CUDA_EXPR constexpr void _copy_values_from(const OT*& values, const size_t& size) {
			if constexpr (MSTD_STD_NAMESPACE::is_same_v<OT, T>) {
				MSTD_STD_NAMESPACE::memcpy(&_values[0], values, MSTD_STD_NAMESPACE::min(N, size) * sizeof(T));
			}
			else {
				for (size_t i = 0; i != MSTD_STD_NAMESPACE::min(N, size); ++i) {
					_values[i] = (T)values[i];
				}
			}
		}

#if _HAS_CXX20
		template<size_t TN, arithmetic OT>
#else
		template<size_t TN, class OT>
#endif
		MSTD_CUDA_EXPR constexpr void _copy_values_from(const OT(&values)[TN]) {
			if constexpr (MSTD_STD_NAMESPACE::is_same_v<T, OT>) {
				MSTD_STD_NAMESPACE::memcpy(&_values[0], &values[0], MSTD_STD_NAMESPACE::min(N, TN) * sizeof(T));
			}
			else {
				for (size_t i = 0; i != MSTD_STD_NAMESPACE::min(N, TN); ++i) {
					_values[i] = (T)values[i];
				}
			}
		}

#if _HAS_CXX20
		template<size_t ON, arithmetic OT>
			requires (ON > 0)
#else
		template<size_t ON, class OT>
#endif
		MSTD_CUDA_EXPR constexpr void _copy_values_from(const vec<ON, OT>& other) {
			if constexpr (MSTD_STD_NAMESPACE::is_same_v<OT, T>) {
				MSTD_STD_NAMESPACE::memcpy(&_values[0], static_cast<const T*>(other), MSTD_STD_NAMESPACE::min(N, ON) * sizeof(T));
			}
			else {
				for (size_t i = 0; i != MSTD_STD_NAMESPACE::min(N, ON); ++i) {
					_values[i] = (T)other[i];
				}
			}
		}
#pragma endregion // PRIVATE_METHODS

	public:
#pragma region CONSTRUCTORS
		// vecN()
		MSTD_CUDA_EXPR vec() {
			_fill_values(T(0));
		}

		// vecN(x, y, ...)
#if _HAS_CXX20
		template<arithmetic... Ts>
			requires (sizeof...(Ts) > 0 && sizeof...(Ts) <= N)
#else
		template<class... Ts,
			MSTD_STD_NAMESPACE::enable_if_t<(sizeof...(Ts) > 0 && sizeof...(Ts) <= N &&
				are_all_v<MSTD_STD_NAMESPACE::is_arithmetic, Ts...>), bool> = true>
#endif
		MSTD_CUDA_EXPR vec(const Ts&... values) {
			_set_values<Ts...>(MSTD_STD_NAMESPACE::index_sequence_for<Ts...>(), values...);
			_fill_values_from(sizeof...(Ts), T(0));
		}

		// vecN(vec, z, ...)
#if _HAS_CXX20
		template<size_t ON, arithmetic OT, arithmetic... Ts>
			requires (sizeof...(Ts) > 0 && sizeof...(Ts) <= N - ON && ON < N)
#else
		template<size_t ON, class OT, class... Ts,
			MSTD_STD_NAMESPACE::enable_if_t<(sizeof...(Ts) > 0 && sizeof...(Ts) <= N - ON && ON < N &&
				are_all_v<MSTD_STD_NAMESPACE::is_arithmetic, OT, Ts...>), bool> = true>

#endif
		MSTD_CUDA_EXPR vec(const vec<ON, OT>& other, const Ts&... values) {
			_copy_values_from(other);
			_set_values<Ts...>(make_index_sequence_for_from<ON, Ts...>(), values...);
			_fill_values_from(sizeof...(Ts) + ON, T(0));
		}

		// vecN({ 1, 2 })
#if _HAS_CXX20
		template<size_t TN, arithmetic OT>
#else
		template<size_t TN, class OT, MSTD_STD_NAMESPACE::enable_if_t<MSTD_STD_NAMESPACE::is_arithmetic_v<OT>, bool> = true>
#endif
		MSTD_CUDA_EXPR vec(const OT(&values)[TN]) {
			_copy_values_from(values);
			_fill_values_from(TN, T(0));
		}

		// vecN(&table)
#if _HAS_CXX20
		template<arithmetic OT>
#else
		template<class OT, MSTD_STD_NAMESPACE::enable_if_t<MSTD_STD_NAMESPACE::is_arithmetic_v<OT>, bool> = true>
#endif
		MSTD_CUDA_EXPR vec(const OT* values, const size_t& size) {
			_copy_values_from(values, size);
			_fill_values_from(size, T(0));
		}

		// vecN(vecON)
#if _HAS_CXX20
		template<size_t ON, arithmetic OT>
#else
		template<size_t ON, class OT>
#endif
		MSTD_CUDA_EXPR vec(const vec<ON, OT>& other) {
			_copy_values_from(other);
			_fill_values_from(ON, T(0));
		}

#pragma region VECTOR_3_CONSTRUCTORS
#if _HAS_CXX20
		template<arithmetic AT, arithmetic BT, size_t ON>
			requires (ON == 3)
#else
		template<class AT, class BT, size_t ON, MSTD_STD_NAMESPACE::enable_if_t<(ON == 3), bool> = true>
#endif
		MSTD_CUDA_EXPR vec(const vec<ON, AT>& other_a, const vec<ON, BT>& other_b) : vec(other_a.cross(other_b)) {}
#pragma endregion // VECTOR_3_CONSTRUCTORS
#pragma endregion // CONSTRUCTORS

#pragma region DESTRUCTOR
#ifndef MSTD_USE_CUDA
		virtual ~vec() = default;
#endif
#pragma endregion // DESTRUCTOR

#pragma region ASSIGN
#if _HAS_CXX20
		template<size_t TN, arithmetic OT>
#else
		template<size_t TN, class OT, MSTD_STD_NAMESPACE::enable_if_t<MSTD_STD_NAMESPACE::is_arithmetic_v<OT>, bool> = true>
#endif
		MSTD_CUDA_EXPR vec<N, T>& operator=(const OT(&values)[TN]) {
			_copy_values_from(values);
			_fill_values_from(TN, T(0));
			return *this;
		}
#if _HAS_CXX20
		template<size_t ON, arithmetic OT>
#else
		template<size_t ON, class OT, MSTD_STD_NAMESPACE::enable_if_t<MSTD_STD_NAMESPACE::is_arithmetic_v<OT>, bool> = true>
#endif
		MSTD_CUDA_EXPR vec<N, T>& operator=(const vec<ON, OT>& other) {
			_copy_values_from(other);
			_fill_values_from(ON, T(0));
			return *this;
		}
#pragma endregion // ASSIGN

#pragma region PREDEFINED
		MSTD_CUDA_EXPR static vec<N, T> zero() {
			return vec<N, T>();
		}
		MSTD_CUDA_EXPR static vec<N, T> one() {
			return fill(T(1));
		}
		MSTD_CUDA_EXPR static vec<N, T> fill(const T& value) {
			vec<N, T> res;
			res._fill_values(value);
			return res;
		}
#pragma endregion // PREDEFINED

#pragma region PREDEFINED_CHECKS
		MSTD_CUDA_EXPR bool is_zero() const {
			return is_filled_with(T(0));
		}

		MSTD_CUDA_EXPR bool is_one() const {
			return is_filled_with(T(1));
		}

		MSTD_CUDA_EXPR bool is_filled_with(const T& value) const {
			for (size_t i = 0; i != N; ++i) {
				if (_values[i] != value) {
					return false;
				}
			}
			return true;
		}

		MSTD_CUDA_EXPR bool is_normalized() const {
			return length_sq() == T(1);
		}
#pragma endregion // PREDEFINED_CHECKS

#pragma region VECTOR_GETTERS
		MSTD_CUDA_EXPR T& x() {
			return _values[0];
		}
		MSTD_CUDA_EXPR T x() const {
			return _values[0];
		}

		MSTD_CUDA_EXPR T& r() {
			return _values[0];
		}
		MSTD_CUDA_EXPR T r() const {
			return _values[0];
		}

#if _HAS_CXX20
		MSTD_CUDA_EXPR T& y() requires (N > 1) {
#else
		template<size_t _N = N, typename = typename MSTD_STD_NAMESPACE::enable_if<(_N > 1 && _N == N)>::type>
		MSTD_CUDA_EXPR T& y() {
#endif
			return _values[1];
		}
#if _HAS_CXX20
		MSTD_CUDA_EXPR T y() const requires (N > 1) {
#else
		template<size_t _N = N, typename = typename MSTD_STD_NAMESPACE::enable_if<(_N > 1 && _N == N)>::type>
		MSTD_CUDA_EXPR T y() const {
#endif
			return _values[1];
		}

#if _HAS_CXX20
		MSTD_CUDA_EXPR T& g() requires (N > 1) {
#else
		template<size_t _N = N, typename = typename MSTD_STD_NAMESPACE::enable_if<(_N > 1 && _N == N)>::type>
		MSTD_CUDA_EXPR T& g() {
#endif
			return _values[1];
		}
#if _HAS_CXX20
		MSTD_CUDA_EXPR T g() const requires (N > 1) {
#else
		template<size_t _N = N, typename = typename MSTD_STD_NAMESPACE::enable_if<(_N > 1 && _N == N)>::type>
		MSTD_CUDA_EXPR T g() const {
#endif
			return _values[1];
		}

#if _HAS_CXX20
		MSTD_CUDA_EXPR T& z() requires (N > 2) {
#else
		template<size_t _N = N, typename = typename MSTD_STD_NAMESPACE::enable_if<(_N > 2 && _N == N)>::type>
		MSTD_CUDA_EXPR T& z() {
#endif
			return _values[2];
		}
#if _HAS_CXX20
		MSTD_CUDA_EXPR T z() const requires (N > 2) {
#else
		template<size_t _N = N, typename = typename MSTD_STD_NAMESPACE::enable_if<(_N > 2 && _N == N)>::type>
		MSTD_CUDA_EXPR T z() const {
#endif
			return _values[2];
		}

#if _HAS_CXX20
		MSTD_CUDA_EXPR T& b() requires (N > 2) {
#else
		template<size_t _N = N, typename = typename MSTD_STD_NAMESPACE::enable_if<(_N > 2 && _N == N)>::type>
		MSTD_CUDA_EXPR T& b() {
#endif
			return _values[2];
		}
#if _HAS_CXX20
		MSTD_CUDA_EXPR T b() const requires (N > 2) {
#else
		template<size_t _N = N, typename = typename MSTD_STD_NAMESPACE::enable_if<(_N > 2 && _N == N)>::type>
		MSTD_CUDA_EXPR T b() const {
#endif
			return _values[2];
		}

#if _HAS_CXX20
		MSTD_CUDA_EXPR T& w() requires (N > 3) {
#else
		template<size_t _N = N, typename = typename MSTD_STD_NAMESPACE::enable_if<(_N > 3 && _N == N)>::type>
		MSTD_CUDA_EXPR T& w() {
#endif
			return _values[3];
		}
#if _HAS_CXX20
		MSTD_CUDA_EXPR T w() const requires (N > 3) {
#else
		template<size_t _N = N, typename = typename MSTD_STD_NAMESPACE::enable_if<(_N > 3 && _N == N)>::type>
		MSTD_CUDA_EXPR T w() const {
#endif
			return _values[3];
		}

#if _HAS_CXX20
		MSTD_CUDA_EXPR T& a() requires (N > 3) {
#else
		template<size_t _N = N, typename = typename MSTD_STD_NAMESPACE::enable_if<(_N > 3 && _N == N)>::type>
		MSTD_CUDA_EXPR T& a() {
#endif
			return _values[3];
		}
#if _HAS_CXX20
		MSTD_CUDA_EXPR T a() const requires (N > 3) {
#else
		template<size_t _N = N, typename = typename MSTD_STD_NAMESPACE::enable_if<(_N > 3 && _N == N)>::type>
		MSTD_CUDA_EXPR T a() const {
#endif
			return _values[3];
		}
#pragma endregion // VECTOR_GETTERS

#pragma region VECTOR_OPERATIONS
		MSTD_CUDA_EXPR T length_sq() const {
			T value = T(0);
			for (size_t i = 0; i != N; ++i) {
				value += _values[i] * _values[i];
			}
			return value;
		}

		MSTD_CUDA_EXPR T length() const {
#ifdef MSTD_USE_CUDA
			if constexpr (MSTD_STD_NAMESPACE::is_same_v<T, float>) {
				return __fdividef(1.0f, rsqrtf(length_sq()));
			}
			else {
				return static_cast<T>(1.0 / rsqrt(length_sq()));
			}
#else
			return static_cast<T>(MSTD_STD_NAMESPACE::sqrt(length_sq()));
#endif
		}

		MSTD_CUDA_EXPR vec<N, T>& normalize() {
			T len = length();
			if (len == T(0)) {
#ifdef MSTD_USE_CUDA
				return *this;
#else
				throw MSTD_STD_NAMESPACE::runtime_error("length was zero");
#endif
			}
			*this /= len;
			return *this;
		}
		MSTD_CUDA_EXPR vec<N, T> normalized() const {
			vec<N, T> res = *this;
			return res.normalize();
		}

		MSTD_CUDA_EXPR T dot(const vec<N, T>& other) const {
			T res = T(0);
			for (size_t i = 0; i != N; ++i) {
				res += _values[i] * other[i];
			}
			return res;
		}

		MSTD_CUDA_EXPR T angle_between(const vec<N, T>& other) const {
			T this_len = length_sq();
			T other_len = other.length_sq();

			if (this_len == T(0) || other_len == T(0)) {
#ifdef MSTD_USE_CUDA
				return T(0);
#else
				throw MSTD_STD_NAMESPACE::runtime_error("length was zero");
#endif
			}

#ifdef MSTD_USE_CUDA
			if constexpr (MSTD_STD_NAMESPACE::is_same_v<T, float>) {
				return acosf(dot(other) * rsqrtf(this_len * other_len));
			}
			else {
				return MSTD_STD_NAMESPACE::acos(dot(other) * rsqrt(this_len * other_len));
			}
#else
			return MSTD_STD_NAMESPACE::acos(dot(other) / MSTD_STD_NAMESPACE::sqrt(this_len * other_len));
#endif
		}

		MSTD_CUDA_EXPR vec<N, T>& reflect(const vec<N, T>& normal) noexcept {
			*this -= T(2) * this->dot(normal) * normal;
			return *this;
		}

		MSTD_CUDA_EXPR vec<N, T> reflected(const vec<N, T>& normal) const noexcept {
			vec<N, T> res = *this;
			return res.reflect(normal);
		}

		MSTD_CUDA_EXPR vec<N, T>& refract(const vec<N, T>& normal, const T& eta) {
			*this = this->refracted(normal, eta);
			return *this;
		}

		MSTD_CUDA_EXPR vec<N, T> refracted(const vec<N, T>& normal, const T& eta) const {
			float cos_theta = MSTD_STD_NAMESPACE::min((-(*this)).dot(normal), T(1));
			vec<N, T> r_out_perp = eta * (*this + cos_theta * normal);
			float length_sq = r_out_perp.length_sq();
#ifdef MSTD_USE_CUDA
			vec<N, T> r_out_parallel;
			if constexpr (MSTD_STD_NAMESPACE::is_same_v<T, float>) {
				r_out_parallel = -normal / rsqrtf(MSTD_STD_NAMESPACE::abs(1.0f - length_sq));
			}
			else {
				r_out_parallel = -normal / rsqrt(MSTD_STD_NAMESPACE::abs(T(1) - length_sq));
			}
#else
			const vec<N, T> r_out_parallel = -MSTD_STD_NAMESPACE::sqrt(MSTD_STD_NAMESPACE::abs(T(1) - length_sq)) * normal;
#endif
			return r_out_perp + r_out_parallel;
		}

		MSTD_CUDA_EXPR vec<N, T>& saturate() noexcept {
			for (size_t i = 0; i != N; ++i) {
#ifdef MSTD_USE_CUDA
				_values[i] = ::MSTD_NAMESPACE::saturate<T, true>(_values[i]);
#else
				_values[i] = ::MSTD_NAMESPACE::saturate(_values[i]);
#endif
			}
			return *this;
		}

		MSTD_CUDA_EXPR vec<N, T> saturated() const noexcept {
			vec<N, T> res = *this;
			return res.saturate();
		}

		MSTD_CUDA_EXPR vec<N, T>& fract() noexcept {
			for (size_t i = 0; i != N; ++i) {
				_values[i] = ::MSTD_NAMESPACE::fract(_values[i]);
			}
			return *this;
		}

		MSTD_CUDA_EXPR vec<N, T> fracted() const noexcept {
			vec<N, T> res = *this;
			return res.fract();
		}

		MSTD_CUDA_EXPR vec<N, T>& mod(const T& y) {
#ifdef MSTD_USE_CUDA
			if constexpr (MSTD_STD_NAMESPACE::is_same_v<T, float>) {
				float one_over_y = 1.0f / y;
				for (size_t i = 0; i != N; ++i) {
					_values[i] -= y * floorf(_values[i] * one_over_y);
				}
			}
			else {
				for (size_t i = 0; i != N; ++i) {
					_values[i] -= y * MSTD_STD_NAMESPACE::floor(_values[i] / y);
				}
			}
#else
			if constexpr (MSTD_STD_NAMESPACE::is_same_v<T, float>) {
				float one_over_y = 1.0f / y;
				for (size_t i = 0; i != N; ++i) {
					_values[i] -= y * MSTD_STD_NAMESPACE::floorf(_values[i] * one_over_y);
				}
			}
			else {
				for (size_t i = 0; i != N; ++i) {
					_values[i] -= y * MSTD_STD_NAMESPACE::floor(_values[i] / y);
				}
			}
#endif
			return *this;
		}

		MSTD_CUDA_EXPR vec<N, T> modded(const T& y) const {
			vec<N, T> res = *this;
			return res.mod(y);
		}

		MSTD_CUDA_EXPR vec<N, T>& mod(const vec<N, T>& other) {
			for (size_t i = 0; i != N; ++i) {
#ifdef MSTD_USE_CUDA
				if constexpr (MSTD_STD_NAMESPACE::is_same_v<T, float>) {
					_values[i] -= other[i] * floorf(__fdividef(_values[i], other[i]));
				}
				else {
					_values[i] -= other[i] * MSTD_STD_NAMESPACE::floor(_values[i] / other[i]);
				}
#else
				_values[i] -= other[i] * MSTD_STD_NAMESPACE::floor(_values[i] / other[i]);
#endif
			}
			return *this;
		}

		MSTD_CUDA_EXPR vec<N, T> modded(const vec<N, T>& other) const {
			vec<N, T> res = *this;
			return res.mod(other);
		}

		MSTD_CUDA_EXPR vec<N, T>& pow(const T& y) {
			for (size_t i = 0; i != N; ++i) {
				_values[i] = MSTD_STD_NAMESPACE::pow(_values[i], y);
			}
			return *this;
		}

		MSTD_CUDA_EXPR vec<N, T> powed(const T& y) const {
			vec<N, T> res = *this;
			return res.pow(y);
		}

		MSTD_CUDA_EXPR vec<N, T>& pow(const vec<N, T>& other) {
			for (size_t i = 0; i != N; ++i) {
				_values[i] = MSTD_STD_NAMESPACE::pow(_values[i], other[i]);
			}
			return *this;
		}

		MSTD_CUDA_EXPR vec<N, T> powed(const vec<N, T>& other) const {
			vec<N, T> res = *this;
			return res.pow(other);
		}

		MSTD_CUDA_EXPR vec<N, T>& clamp(const T& min_val, const T& max_val) {
			for (size_t i = 0; i != N; ++i) {
				_values[i] = MSTD_CLAMP(_values[i], min_val, max_val);
			}
			return *this;
		}

		MSTD_CUDA_EXPR vec<N, T> clampped(const T& min_val, const T& max_val) const {
			vec<N, T> res = *this;
			return res.clamp(min_val, max_val);
		}

		MSTD_CUDA_EXPR vec<N, T>& clamp(const vec<N, T>& min_val, const vec<N, T>& max_val) {
			for (size_t i = 0; i != N; ++i) {
				_values[i] = MSTD_CLAMP(_values[i], min_val[i], max_val[i]);
			}
			return *this;
		}

		MSTD_CUDA_EXPR vec<N, T> clampped(const vec<N, T>& min_val, const vec<N, T>& max_val) const {
			vec<N, T> res = *this;
			return res.clamp(min_val, max_val);
		}

#pragma region VECTOR_3_OPERATIONS
#if _HAS_CXX20
		MSTD_CUDA_EXPR vec<N, T> cross(const vec<N, T>& other) const requires (N == 3) {
#else
		template<size_t _N = N, typename = typename MSTD_STD_NAMESPACE::enable_if<(_N == 3 && _N == N)>::type>
		MSTD_CUDA_EXPR vec<N, T> cross(const vec<N, T>& other) const {
#endif
			return vec<N, T>(
				_values[1] * other[2] - _values[2] * other[1],
				_values[2] * other[0] - _values[0] * other[2],
				_values[0] * other[1] - _values[1] * other[0]
			);
		}

#if _HAS_CXX20
		MSTD_CUDA_EXPR vec<N, T>& rotate(const vec<N, T>& axis, const T& radians) requires (N == 3) {
#else
		template<size_t _N = N, typename = typename MSTD_STD_NAMESPACE::enable_if<(_N == 3 && _N == N)>::type>
		MSTD_CUDA_EXPR vec<N, T>& rotate(const vec<N, T>&axis, const T& radians) {
#endif
			const ::MSTD_NAMESPACE::quat<T> p(T(0), (*this));

			vec<N, T> norm_axis = axis;
			if (!norm_axis.is_zero()) norm_axis.normalize();

			const ::MSTD_NAMESPACE::quat<T>& q = ::MSTD_NAMESPACE::quat<T>::rotation(norm_axis, radians);

			const ::MSTD_NAMESPACE::quat<T>& invers_q = q.inverted();

			*this = (q * p * invers_q).v;
			return *this;
		}
#if _HAS_CXX20
		MSTD_CUDA_EXPR vec<N, T> rotated(const vec<N, T>& axis, const T& radians) requires (N == 3) {
#else
		template<size_t _N = N, typename = typename MSTD_STD_NAMESPACE::enable_if<(_N == 3 && _N == N)>::type>
		MSTD_CUDA_EXPR vec<N, T> rotated(const vec<N, T>& axis, const T& radians) {
#endif
			vec<N, T> res = *this;
			return res.rotate(axis, radians);
		}
#pragma endregion // VECTOR_3_OPERATIONS
#pragma endregion // VECTOR_OPERTATIONS

#pragma region OPERATORS
		MSTD_CUDA_EXPR vec<N, T>& operator+=(const vec<N, T>& other) {
			for (size_t i = 0; i != N; ++i) {
				_values[i] += other[i];
			}
			return *this;
		}
		MSTD_CUDA_EXPR vec<N, T>& operator-=(const vec<N, T>& other) {
			for (size_t i = 0; i != N; ++i) {
				_values[i] -= other[i];
			}
			return *this;
		}
		MSTD_CUDA_EXPR vec<N, T>& operator*=(const vec<N, T>& other) {
			for (size_t i = 0; i != N; ++i) {
				_values[i] *= other[i];
			}
			return *this;
		}
		MSTD_CUDA_EXPR vec<N, T>& operator/=(const vec<N, T>& other) {
			if (other.is_zero()) {
#ifdef MSTD_USE_CUDA
				return *this;
#else
				throw MSTD_STD_NAMESPACE::runtime_error("division by zero");
#endif
			}
			if constexpr (!MSTD_STD_NAMESPACE::is_same_v<T, float> && !MSTD_STD_NAMESPACE::is_same_v<T, double>) {
				for (size_t i = 0; i != N; ++i) {
					_values[i] /= other[i];
				}
			}
			else {
				T one_over_other[N];
				for (size_t i = 0; i != N; ++i) {
#ifdef MSTD_USE_CUDA
					if constexpr (MSTD_STD_NAMESPACE::is_same_v<T, float>) {
						one_over_other[i] = __fdividef(1.0f, other[i]);
					}
					else {
						one_over_other[i] = 1.0 / other[i];
					}
#else
					one_over_other[i] = 1.0 / other[i];
#endif
				}
				for (size_t i = 0; i != N; ++i) {
					_values[i] *= one_over_other[i];
				}
			}
			return *this;
		}

		MSTD_CUDA_EXPR vec<N, T>& operator+=(const T& y) {
			for (size_t i = 0; i != N; ++i) {
				_values[i] += y;
			}
			return *this;
		}
		MSTD_CUDA_EXPR vec<N, T>& operator-=(const T& y) {
			for (size_t i = 0; i != N; ++i) {
				_values[i] -= y;
			}
			return *this;
		}
		MSTD_CUDA_EXPR vec<N, T>& operator*=(const T& y) {
			for (size_t i = 0; i != N; ++i) {
				_values[i] *= y;
			}
			return *this;
		}
		MSTD_CUDA_EXPR vec<N, T>& operator/=(const T& y) {
			if (y == T(0)) {
#ifdef MSTD_USE_CUDA
				return *this;
#else
				throw MSTD_STD_NAMESPACE::runtime_error("division by zero");
#endif
			}
			if constexpr (!MSTD_STD_NAMESPACE::is_same_v<T, float> && !MSTD_STD_NAMESPACE::is_same_v<T, double>) {
				for (size_t i = 0; i != N; ++i) {
					_values[i] /= y;
				}
			}
			else {
#ifdef MSTD_USE_CUDA
				T one_over_y;
				if constexpr (MSTD_STD_NAMESPACE::is_same_v<T, float>) {
					one_over_y = 1.0f / y;
				}
				else {
					one_over_y = 1.0 / y;
				}
#else
				T one_over_y = 1.0 / y;
#endif
				for (size_t i = 0; i != N; ++i) {
					_values[i] *= one_over_y;
				}
			}
			return *this;
		}

		MSTD_CUDA_EXPR vec<N, T> operator+(const vec<N, T>& other) const {
			vec<N, T> res = *this;
			res += other;
			return res;
		}
		MSTD_CUDA_EXPR vec<N, T> operator-(const vec<N, T>& other) const {
			vec<N, T> res = *this;
			res -= other;
			return res;
		}
		MSTD_CUDA_EXPR vec<N, T> operator*(const vec<N, T>& other) const {
			vec<N, T> res = *this;
			res *= other;
			return res;
		}
		MSTD_CUDA_EXPR vec<N, T> operator/(const vec<N, T>& other) const {
			vec<N, T> res = *this;
			res /= other;
			return res;
		}

		MSTD_CUDA_EXPR vec<N, T> operator+(const T& y) const {
			vec<N, T> res = *this;
			res += y;
			return res;
		}
		MSTD_CUDA_EXPR vec<N, T> operator-(const T& y) const {
			vec<N, T> res = *this;
			res -= y;
			return res;
		}
		MSTD_CUDA_EXPR vec<N, T> operator*(const T& y) const {
			vec<N, T> res = *this;
			res *= y;
			return res;
		}
		MSTD_CUDA_EXPR MSTD_FRIEND vec<N, T> operator*(const T& y, const vec<N, T>& vector) {
			return vector * y;
		}
		MSTD_CUDA_EXPR vec<N, T> operator/(const T& y) const {
			vec<N, T> res = *this;
			res /= y;
			return res;
		}

		MSTD_CUDA_EXPR vec<N, T> operator+() const {
			return vec<N, T>(*this);
		}
		MSTD_CUDA_EXPR vec<N, T> operator-() const {
			return *this * -1;
		}

		MSTD_CUDA_EXPR vec<N, T>& operator++() {
			return *this += vec<N, T>::one();
		}
		MSTD_CUDA_EXPR vec<N, T>& operator--() {
			return *this -= vec<N, T>::one();
		}

		template<size_t ON>
		MSTD_CUDA_EXPR bool operator==(const vec<ON, T>& other) const {
			if constexpr (N != ON) {
				return false;
			}
			else {
#ifdef MSTD_USE_CUDA
				return ::thrust::equal(::thrust::device, _values, _values + N, static_cast<const T*>(other));
#else
				return MSTD_STD_NAMESPACE::memcmp(_values, static_cast<const T*>(other), N * sizeof(T)) == 0;
#endif
			}
		}
		template<size_t ON>
		MSTD_CUDA_EXPR bool operator!=(const vec<ON, T>& other) const {
			return !(*this == other);
		}

		MSTD_CUDA_EXPR operator const T* () const {
			return _values;
		}

		MSTD_CUDA_EXPR T& operator[](const size_t& idx) {
			return _values[idx];
		}
		MSTD_CUDA_EXPR T operator[](const size_t& idx) const {
			return _values[idx];
		}

		// ostream operators are not supported on cuda
#ifndef MSTD_USE_CUDA
		MSTD_FRIEND MSTD_STD_NAMESPACE::ostream& operator<<(MSTD_STD_NAMESPACE::ostream& str, const vec<N, T>& vector) {
			str << "[";
			for (size_t i = 0; i != N; ++i) {
				str << MSTD_STD_NAMESPACE::to_string(vector[i]);
				if (i != N - 1) str << ", ";
			}
			return str << "]";
		}
#else
		MSTD_CUDA_EXPR void print() const {
			printf("[");
			for (size_t i = 0; i != N; ++i) {

				if constexpr (MSTD_STD_NAMESPACE::is_same_v<T, float>) {
					printf("%f", _values[i]);
				}
				else if constexpr (MSTD_STD_NAMESPACE::is_same_v<T, bool>) {
					printf("%s", _values[i] ? "true" : "false");
				}
				else {
					printf("%d", _values[i]);
				}

				if (i != N - 1) printf(", ");
			}
			printf("]");
		}
#endif

#pragma endregion // OPERATORS
	};

#pragma region EXTRA_OPERATORS
	template<class T, size_t N>
	MSTD_CUDA_EXPR static vec<N, T> max(const vec<N, T>& a, const vec<N, T>& b) noexcept {
		vec<N, T> res;
		for (size_t i = 0; i != N; ++i) {
			res[i] = MSTD_STD_NAMESPACE::max(a[i], b[i]);
		}
		return res;
	}

	template<class T, size_t N>
	MSTD_CUDA_EXPR static vec<N, T> min(const vec<N, T>& a, const vec<N, T>& b) noexcept {
		vec<N, T> res;
		for (size_t i = 0; i != N; ++i) {
			res[i] = MSTD_STD_NAMESPACE::min(a[i], b[i]);
		}
		return res;
	}

	template<class T, size_t N>
	MSTD_CUDA_EXPR static T dot(const vec<N, T>& a, const vec<N, T>& b) {
		return a.dot(b);
	}

#if _HAS_CXX20
	template<class T, size_t N>
		requires (N == 3)
#else
	template<class T, size_t N, MSTD_STD_NAMESPACE::enable_if_t<(N == 3), bool> = true>
#endif
	MSTD_CUDA_EXPR static vec<N, T> cross(const vec<N, T>& a, const vec<N, T>& b) {
		return a.cross(b);
	}

	template<class T, size_t N>
	MSTD_CUDA_EXPR static T angle_between(const vec<N, T>& a, const vec<N, T>& b) {
		return a.angle_between(b);
	}

	template<class T, size_t N>
	MSTD_CUDA_EXPR static vec<N, T> reflect(const vec<N, T>& dir, const vec<N, T>& normal) {
		return dir.reflected(normal);
	}

	template<class T, size_t N>
	MSTD_CUDA_EXPR static vec<N, T> refract(const vec<N, T>& dir, const vec<N, T>& normal, const T& eta) {
		return dir.refracted(normal, eta);
	}

	template<class T, size_t N>
	MSTD_CUDA_EXPR static vec<N, T> saturate(const vec<N, T>& a) {
		return a.saturated();
	}

	template<class T, size_t N>
	MSTD_CUDA_EXPR static vec<N, T> fract(const vec<N, T>& a) {
		return a.fracted();
	}

	template<class T, size_t N>
	MSTD_CUDA_EXPR static vec<N, T> mod(const vec<N, T>& a, const T& y) {
		return a.modded(y);
	}

	template<class T, size_t N>
	MSTD_CUDA_EXPR static vec<N, T> mod(const vec<N, T>& a, const vec<N, T>& b) {
		return a.modded(b);
	}

	template<class T, size_t N>
	MSTD_CUDA_EXPR static vec<N, T> pow(const vec<N, T>& a, const T& y) {
		return a.powed(y);
	}

	template<class T, size_t N>
	MSTD_CUDA_EXPR static vec<N, T> pow(const vec<N, T>& a, const vec<N, T>& b) {
		return a.powed(b);
	}

	template<class T, size_t N>
	MSTD_CUDA_EXPR static vec<N, T> clamp(const vec<N, T>& a, const T& min_val, const T& max_val) {
		return a.clampped(min_val, max_val);
	}

	template<class T, size_t N>
	MSTD_CUDA_EXPR static vec<N, T> clamp(const vec<N, T>& a, const vec<N, T>& min_val, const vec<N, T>& max_val) {
		return a.clampped(min_val, max_val);
	}
#pragma endregion // EXTRA_OPERATORS

#pragma region PREDEFINED_TYPES
	using vec2 = vec<2ull, float>;
	using dvec2 = vec<2ull, double>;
	using ldvec2 = vec<2ull, long double>;
	using ivec2 = vec<2ull, int>;
	using uvec2 = vec<2ull, unsigned int>;
	using bvec2 = vec<2ull, bool>;
	using cvec2 = vec<2ull, char>;
	using ucvec2 = vec<2ull, unsigned char>;
	using scvec2 = vec<2ull, signed char>;
	using lvec2 = vec<2ull, long>;
	using ulvec2 = vec<2ull, unsigned long>;
	using llvec2 = vec<2ull, long long>;
	using ullvec2 = vec<2ull, unsigned long long>;

	using vec3 = vec<3ull, float>;
	using dvec3 = vec<3ull, double>;
	using ldvec3 = vec<3ull, long double>;
	using ivec3 = vec<3ull, int>;
	using uvec3 = vec<3ull, unsigned int>;
	using bvec3 = vec<3ull, bool>;
	using cvec3 = vec<3ull, char>;
	using ucvec3 = vec<3ull, unsigned char>;
	using scvec3 = vec<3ull, signed char>;
	using lvec3 = vec<3ull, long>;
	using ulvec3 = vec<3ull, unsigned long>;
	using llvec3 = vec<3ull, long long>;
	using ullvec3 = vec<3ull, unsigned long long>;

	using vec4 = vec<4ull, float>;
	using dvec4 = vec<4ull, double>;
	using ldvec4 = vec<4ull, long double>;
	using ivec4 = vec<4ull, int>;
	using uvec4 = vec<4ull, unsigned int>;
	using bvec4 = vec<4ull, bool>;
	using cvec4 = vec<4ull, char>;
	using ucvec4 = vec<4ull, unsigned char>;
	using scvec4 = vec<4ull, signed char>;
	using lvec4 = vec<4ull, long>;
	using ulvec4 = vec<4ull, unsigned long>;
	using llvec4 = vec<4ull, long long>;
	using ullvec4 = vec<4ull, unsigned long long>;
#pragma endregion // PREDEFINED_TYPES
}
#include "quat.hpp"