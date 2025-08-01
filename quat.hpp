#pragma once
#include "vec.hpp"
#include "math_functions.hpp"
#include "math_types.hpp"

namespace MSTD_NAMESPACE {
#if _HAS_CXX20
	template<arithmetic T>
#else
	template<class T, MSTD_STD_NAMESPACE::enable_if_t<MSTD_STD_NAMESPACE::is_arithmetic_v<T>, bool>>
#endif
	class quat {
	public:
		using value_type = T;
		using vec_type = ::MSTD_NAMESPACE::vec<3ull, T>;

		T s;
		vec_type v;

#pragma region CONSTRUCTORS
		MSTD_CUDA_EXPR quat() : s(0), v() {}
		MSTD_CUDA_EXPR quat(const T& scalar, const vec_type& vector) : s(scalar), v(vector) {}
		MSTD_CUDA_EXPR quat(const T& scalar, const T& x, const T& y, const T& z) : s(scalar), v(x, y, z) {}
#if _HAS_CXX20
		template<arithmetic OT>
#else
		template<class OT, MSTD_STD_NAMESPACE::enable_if_t<MSTD_STD_NAMESPACE::is_arithmetic_v<OT>, bool> = true>
#endif
		MSTD_CUDA_EXPR quat(const quat<OT>& other) : s(other.s), v(other.v) {}
#pragma endregion // CONSTRUCTORS

#pragma region DESTRUCTOR
#ifndef MSTD_USE_CUDA
		virtual ~quat() = default;
#endif
#pragma endregion // DESTRUCTOR

#pragma region ASSIGN
#if _HAS_CXX20
		template<arithmetic OT>
#else
		template<class OT, MSTD_STD_NAMESPACE::enable_if_t<MSTD_STD_NAMESPACE::is_arithmetic_v<OT>, bool> = true>
#endif
		MSTD_CUDA_EXPR quat<T>& operator=(const quat<OT>& other) {
			s = (T)other.s;
			v = other.v;
			return *this;
		}
#pragma endregion // ASSIGN

#pragma region PREDEFINED_QUATERNIONS
		MSTD_CUDA_EXPR static quat<T> rotation(const vec_type& axis, const T& radians) {
			quat<T> q;
			if (!axis.is_zero()) {
				q = quat<T>((T)MSTD_STD_NAMESPACE::cos(radians * 0.5), axis.normalized() * (T)MSTD_STD_NAMESPACE::sin(radians * 0.5));
			}
			else {
				q = quat<T>((T)MSTD_STD_NAMESPACE::cos(radians * 0.5), axis);
			}
			if (q.magnitude() != T(0)) q.normalize();
			return q;
		}

		MSTD_CUDA_EXPR static quat<T> from_euler_angels(const vec_type& euler_angels) {
			return from_radians({ ::MSTD_NAMESPACE::deg_to_rad(euler_angels[0]), ::MSTD_NAMESPACE::deg_to_rad(euler_angels[1]), ::MSTD_NAMESPACE::deg_to_rad(euler_angels[2]) });
		}

		MSTD_CUDA_EXPR static quat<T> from_radians(const vec_type& radians) {
			quat<T> qx = rotation(vec_type(T(1), T(0), T(0)), radians[0]);
			quat<T> qy = rotation(vec_type(T(0), T(1), T(0)), radians[1]);
			quat<T> qz = rotation(vec_type(T(0), T(0), T(1)), radians[2]);

			// ZYX convention
			quat<T> q = qz * qy * qx;
			if (q.magnitude() != T(0)) q.normalize();
			return q;
		}
#pragma endregion // PREDEFINED_QUATERNIONS

#pragma region QUATERNION_OPERATIONS
		MSTD_CUDA_EXPR T magnitude() const {
			return (T)MSTD_STD_NAMESPACE::sqrt(s * s + v.dot(v));
		}

		MSTD_CUDA_EXPR quat<T>& normalize() {
			T m = magnitude();
			*this /= m;
			return *this;
		}
		MSTD_CUDA_EXPR quat<T> normalized() const {
			quat<T> res = *this;
			return res.normalize();
		}

		MSTD_CUDA_EXPR quat<T>& conjugate() {
			v *= -1;
			return *this;
		}
		MSTD_CUDA_EXPR quat<T> conjugated() const {
			quat<T> res = *this;
			return res.conjugate();
		}

		MSTD_CUDA_EXPR quat<T>& invert() {
			T magnitudes = magnitude();
			magnitudes *= magnitudes;
			magnitudes = (T)(1.0 / magnitudes);

			conjugate();

			s *= magnitudes;
			v *= magnitudes;

			return *this;
		}
		MSTD_CUDA_EXPR quat<T> inverted() const {
			quat<T> res = *this;
			return res.invert();
		}

		MSTD_CUDA_EXPR vec_type to_radians() const {
			vec_type res;
			quat<T> q = *this;

			if (q.magnitude() != T(0)) q.normalize();

			// roll (x-axis rotation)
			T sinx_cosp = (T)(2.0 * (q.s * q.v[0] + q.v[1] * q.v[2]));
			T cosx_cosp = (T)(1.0 - 2.0 * (q.v[0] * q.v[0] + q.v[1] * q.v[1]));
			res[0] = (T)MSTD_STD_NAMESPACE::atan2(sinx_cosp, cosx_cosp);

			// pitch (y-axis rotation)
			T siny = (T)MSTD_STD_NAMESPACE::sqrt(1.0 + 2.0 * (q.s * q.v[1] - q.v[0] * q.v[2]));
			T cosy = (T)MSTD_STD_NAMESPACE::sqrt(1.0 - 2.0 * (q.s * q.v[1] - q.v[0] * q.v[2]));
			res[1] = (T)(2.0 * MSTD_STD_NAMESPACE::atan2(siny, cosy) - MSTD_PI / 2.0);

			// yaw (z-axis rotation)
			T sinz_cosp = (T)(2.0 * (q.s * q.v[2] + q.v[0] * q.v[1]));
			T cosz_cosp = (T)(1.0 - 2.0 * (q.v[1] * q.v[1] + q.v[2] * q.v[2]));
			res[2] = (T)MSTD_STD_NAMESPACE::atan2(sinz_cosp, cosz_cosp);

			return res;
		}

		MSTD_CUDA_EXPR vec_type to_euler_angles() const {
			vec_type res = to_radians();
			res[0] = ::MSTD_NAMESPACE::rad_to_deg(res[0]);
			res[1] = ::MSTD_NAMESPACE::rad_to_deg(res[1]);
			res[2] = ::MSTD_NAMESPACE::rad_to_deg(res[2]);
			return res;
		}

		MSTD_CUDA_EXPR T scalar(const quat<T>& other) {
			return s * other.s + v.dot(other.v);
		}
#pragma endregion // QUATERNION_OPERATIONS

#pragma region OPERATORS
		MSTD_CUDA_EXPR quat<T>& operator+=(const quat<T>& other) {
			s += other.s;
			v += other.v;
			return *this;
		}
		MSTD_CUDA_EXPR quat<T>& operator-=(const quat<T>& other) {
			s -= other.s;
			v -= other.v;
			return *this;
		}
		MSTD_CUDA_EXPR quat<T>& operator*=(const quat<T>& other) {
			T t = s;
			s = s * other.s - v.dot(other.v);
			v = other.v * t + v * other.s + v.cross(other.v);
			return *this;
		}
		MSTD_CUDA_EXPR quat<T>& operator*=(const vec_type& other) {
			quat<T> p(T(0), other);
			*this = p;
			return *this;
		}
		MSTD_CUDA_EXPR quat<T>& operator*=(const T& other) {
			s *= other;
			v *= other;
			return *this;
		}
		MSTD_CUDA_EXPR quat<T>& operator/=(const quat<T>& other) {
			*this *= other.inverted();
			return *this;
		}
		MSTD_CUDA_EXPR quat<T>& operator/=(const T& other) {
			if (other == T(0)) {
#ifdef MSTD_USE_CUDA
				return *this;
#else
				throw MSTD_STD_NAMESPACE::runtime_error("division by zero");
#endif
			}
			s /= other;
			v /= other;
			return *this;
		}

		MSTD_CUDA_EXPR quat<T> operator+(const quat<T>& other) const {
			quat<T> res = *this;
			return res += other;
		}
		MSTD_CUDA_EXPR quat<T> operator-(const quat<T>& other) const {
			quat<T> res = *this;
			return res -= other;
		}
		MSTD_CUDA_EXPR quat<T> operator*(const quat<T>& other) const {
			quat<T> res = *this;
			return res *= other;
		}
		MSTD_CUDA_EXPR quat<T> operator*(const vec_type& other) const {
			quat<T> res = *this;
			return res *= other;
		}
		MSTD_CUDA_EXPR MSTD_FRIEND quat<T> operator*(const vec_type& other, const quat<T>& quaternion) {
			return quaternion * other;
		}
		MSTD_CUDA_EXPR quat<T> operator*(const T& other) const {
			quat<T> res = *this;
			return res *= other;
		}
		MSTD_CUDA_EXPR MSTD_FRIEND quat<T> operator*(const T& other, const quat<T>& quaternion) {
			return quaternion * other;
		}
		MSTD_CUDA_EXPR quat<T> operator/(const quat<T>& other) const {
			quat<T> res = *this;
			return res /= other;
		}
		MSTD_CUDA_EXPR quat<T> operator/(const T& other) const {
			quat<T> res = *this;
			return res /= other;
		}

		MSTD_CUDA_EXPR quat<T> operator-() const {
			return *this * -1;
		}
		MSTD_CUDA_EXPR quat<T> operator+() const {
			return quat<T>(*this);
		}
		MSTD_CUDA_EXPR quat<T>& operator--() {
			s -= 1;
			v--;
			return *this;
		}
		MSTD_CUDA_EXPR quat<T>& operator++() {
			s += 1;
			v++;
			return *this;
		}

		MSTD_CUDA_EXPR bool operator==(const quat<T>& other) const {
			return s == other.s && v == other.v;
		}
		MSTD_CUDA_EXPR bool operator!=(const quat<T>& other) const {
			return !(*this == other);
		}

		// ostream operators are not supported on cuda
#ifndef MSTD_USE_CUDA
		MSTD_FRIEND MSTD_STD_NAMESPACE::ostream& operator<<(MSTD_STD_NAMESPACE::ostream& str, const quat<T>& quaternion) {
			return str << "(" << MSTD_STD_NAMESPACE::to_string(quaternion.s) << ", " << quaternion.v << ")";
		}
#else
		MSTD_CUDA_EXPR void print() const {
			printf("(");

			if constexpr (MSTD_STD_NAMESPACE::is_same_v<T, float>) {
				printf("%f, ", s);
			}
			else if constexpr (MSTD_STD_NAMESPACE::is_same_v<T, bool>) {
				printf("%s, ", s ? "true" : "false");
			}
			else {
				printf("%d, ", s);
			}

			v.print();

			printf(")");
		}
#endif

#pragma endregion // OPERATORS
	};

#pragma region PREDEFINED_TYPES
	using fquat = quat<float>;
	using dquat = quat<double>;
	using ldquat = quat<long double>;
	using iquat = quat<int>;
	using uquat = quat<unsigned int>;
	using bquat = quat<bool>;
	using cquat = quat<char>;
	using ucquat = quat<unsigned char>;
	using scquat = quat<signed char>;
	using lquat = quat<long>;
	using ulquat = quat<unsigned long>;
	using llquat = quat<long long>;
	using ullquat = quat<unsigned long long>;
#pragma endregion // PREDEFINED_TYPES
}