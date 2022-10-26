#pragma once
#include <immintrin.h>

#define FINLINE __forceinline

typedef __m256 floatx8;
typedef __m256i uint8x32;
typedef __m256i int8x32;
typedef __m256i uint16x16;
typedef __m256i int16x16;
typedef __m256i uint32x8;
typedef __m256i int32x8;
typedef __m256i uint64x4;
typedef __m256i int64x4;

typedef __m128 floatx4;
typedef __m128i uint8x16;
typedef __m128i uint32x4;

#pragma pack(push, 1)

struct alignas(32) vec3fx8 {
	floatx8 x;
	floatx8 y;
	floatx8 z;

	FINLINE vec3fx8() : x{ _mm256_setzero_ps() }, y{ _mm256_setzero_ps() }, z{ _mm256_setzero_ps() } {
	}

	FINLINE vec3fx8(const floatx8& v) : x{ v }, y{ v }, z{ v } {
	}

	FINLINE vec3fx8(const floatx8& x, const floatx8& y, const floatx8& z) : x{ x }, y{ y }, z{ z } {
	}

	floatx8& get(uint32_t idx) {
		return reinterpret_cast<floatx8*>(this)[idx];
	}

	FINLINE void set(const floatx8& f) {
		x = f;
		y = f;
		z = f;
	}

	FINLINE void set(float f) {
		floatx8 fx8 = _mm256_set1_ps(f);
		x = fx8;
		y = fx8;
		z = fx8;
	}

	FINLINE void setzero() {
		x = _mm256_setzero_ps();
		y = _mm256_setzero_ps();
		z = _mm256_setzero_ps();
	}

	FINLINE floatx8 equal_zero() const {
		floatx8 zero = _mm256_setzero_ps();
		floatx8 cmpx = _mm256_cmp_ps(x, zero, _CMP_EQ_UQ);
		floatx8 cmpy = _mm256_cmp_ps(y, zero, _CMP_EQ_UQ);
		floatx8 cmpz = _mm256_cmp_ps(z, zero, _CMP_EQ_UQ);
		return _mm256_and_ps(cmpx, _mm256_and_ps(cmpy, cmpz));
	}

	FINLINE vec3fx8 operator+(const vec3fx8& other) const {
		return vec3fx8{ _mm256_add_ps(x, other.x), _mm256_add_ps(y, other.y), _mm256_add_ps(z, other.z) };
	}

	FINLINE vec3fx8 operator-(const vec3fx8& other) const {
		return vec3fx8{ _mm256_sub_ps(x, other.x), _mm256_sub_ps(y, other.y), _mm256_sub_ps(z, other.z) };
	}

	FINLINE vec3fx8 operator*(const vec3fx8& other) const {
		return vec3fx8{ _mm256_mul_ps(x, other.x), _mm256_mul_ps(y, other.y), _mm256_mul_ps(z, other.z) };
	}

	FINLINE vec3fx8 operator/(const vec3fx8& other) const {
		return vec3fx8{ _mm256_div_ps(x, other.x), _mm256_div_ps(y, other.y), _mm256_div_ps(z, other.z) };
	}

	FINLINE vec3fx8 operator+(const floatx8& other) const {
		return vec3fx8{ _mm256_add_ps(x, other), _mm256_add_ps(y, other), _mm256_add_ps(z, other) };
	}

	FINLINE vec3fx8 operator-(const floatx8& other) const {
		return vec3fx8{ _mm256_sub_ps(x, other), _mm256_sub_ps(y, other), _mm256_sub_ps(z, other) };
	}

	FINLINE vec3fx8 operator*(const floatx8& other) const {
		return vec3fx8{ _mm256_mul_ps(x, other), _mm256_mul_ps(y, other), _mm256_mul_ps(z, other) };
	}

	FINLINE vec3fx8 operator/(const floatx8& other) const {
		const floatx8 lrcp = _mm256_rcp_ps(other);
		return vec3fx8{ _mm256_mul_ps(x, lrcp), _mm256_mul_ps(y, lrcp), _mm256_mul_ps(z, lrcp) };
	}

	FINLINE vec3fx8 rcp() const {
		return vec3fx8{ _mm256_rcp_ps(x), _mm256_rcp_ps(y), _mm256_rcp_ps(z) };
	}

	FINLINE floatx8 operator==(const vec3fx8& other) const {
		floatx8 cmpx = _mm256_cmp_ps(x, other.x, _CMP_EQ_UQ);
		floatx8 cmpy = _mm256_cmp_ps(y, other.y, _CMP_EQ_UQ);
		floatx8 cmpz = _mm256_cmp_ps(z, other.z, _CMP_EQ_UQ);
		return _mm256_and_ps(cmpx, _mm256_and_ps(cmpy, cmpz));
	}
};

#pragma pack(pop)

FINLINE vec3fx8 vec3fx8_load(float* f) {
	return vec3fx8{ _mm256_load_ps(f), _mm256_load_ps(f + 8), _mm256_load_ps(f + 16) };
}

FINLINE vec3fx8 operator+(const floatx8& left, const vec3fx8& right) {
	return right + left;
}

FINLINE vec3fx8 operator-(const floatx8& left, const vec3fx8& right) {
	return vec3fx8{ _mm256_sub_ps(left, right.x), _mm256_sub_ps(left, right.y), _mm256_sub_ps(left, right.z) };
}

FINLINE vec3fx8 operator*(const floatx8& left, const vec3fx8& right) {
	return right * left;
}

FINLINE vec3fx8 operator/(const floatx8& left, const vec3fx8& right) {
	return vec3fx8{ _mm256_div_ps(left, right.x), _mm256_div_ps(left, right.y), _mm256_div_ps(left, right.z) };
}

FINLINE vec3fx8 fmadd(const vec3fx8& m1, const vec3fx8& m2, const vec3fx8& a) {
	return vec3fx8{ _mm256_fmadd_ps(m1.x, m2.x, a.x), _mm256_fmadd_ps(m1.y, m2.y, a.y), _mm256_fmadd_ps(m1.z, m2.z, a.z) };
}

FINLINE vec3fx8 fmadd(const vec3fx8& m1, const vec3fx8& m2, const floatx8& a) {
	return vec3fx8{ _mm256_fmadd_ps(m1.x, m2.x, a), _mm256_fmadd_ps(m1.y, m2.y, a), _mm256_fmadd_ps(m1.z, m2.z, a) };
}

FINLINE vec3fx8 fmadd(const floatx8& m1, const vec3fx8& m2, const vec3fx8& a) {
	return vec3fx8{ _mm256_fmadd_ps(m1, m2.x, a.x), _mm256_fmadd_ps(m1, m2.y, a.y), _mm256_fmadd_ps(m1, m2.z, a.z) };
}

FINLINE vec3fx8 fmadd(const floatx8& m1, const vec3fx8& m2, const floatx8& a) {
	return vec3fx8{ _mm256_fmadd_ps(m1, m2.x, a), _mm256_fmadd_ps(m1, m2.y, a), _mm256_fmadd_ps(m1, m2.z, a) };
}

#pragma pack(push, 1)

struct alignas(32) vec4fx8 {
	floatx8 x;
	floatx8 y;
	floatx8 z;
	floatx8 w;

	FINLINE vec4fx8() : x{ _mm256_setzero_ps() }, y{ _mm256_setzero_ps() }, z{ _mm256_setzero_ps() }, w{ _mm256_setzero_ps() } {
	}

	FINLINE vec4fx8(const floatx8& v) : x{ v }, y{ v }, z{ v }, w{ v } {
	}

	FINLINE vec4fx8(const floatx8& x, const floatx8& y, const floatx8& z, const floatx8& w) : x{ x }, y{ y }, z{ z }, w{ w } {
	}

	FINLINE vec4fx8(const vec3fx8& v, const floatx8 w) : x{ v.x }, y{ v.y }, z{ v.z }, w{ w } {
	}

	floatx8& get(uint32_t idx) {
		return reinterpret_cast<floatx8*>(this)[idx];
	}

	FINLINE void set(floatx8 f) {
		x = f;
		y = f;
		z = f;
		w = f;
	}

	FINLINE void set(float f) {
		floatx8 fx8 = _mm256_set1_ps(f);
		x = fx8;
		y = fx8;
		z = fx8;
		w = fx8;
	}

	FINLINE void setzero() {
		x = _mm256_setzero_ps();
		y = _mm256_setzero_ps();
		z = _mm256_setzero_ps();
		w = _mm256_setzero_ps();
	}

	FINLINE floatx8 equal_zero() const {
		floatx8 zero = _mm256_setzero_ps();
		floatx8 cmpx = _mm256_cmp_ps(x, zero, _CMP_EQ_UQ);
		floatx8 cmpy = _mm256_cmp_ps(y, zero, _CMP_EQ_UQ);
		floatx8 cmpz = _mm256_cmp_ps(z, zero, _CMP_EQ_UQ);
		floatx8 cmpw = _mm256_cmp_ps(w, zero, _CMP_EQ_UQ);
		return _mm256_and_ps(cmpx, _mm256_and_ps(cmpy, _mm256_and_ps(cmpz, cmpw)));
	}

	FINLINE vec3fx8 xyz() const {
		return vec3fx8{ x, y, z };
	}

	FINLINE vec4fx8 operator+(const vec4fx8& other) const {
		return vec4fx8{ _mm256_add_ps(x, other.x), _mm256_add_ps(y, other.y), _mm256_add_ps(z, other.z), _mm256_add_ps(w, other.w) };
	}

	FINLINE vec4fx8 operator-(const vec4fx8& other) const {
		return vec4fx8{ _mm256_sub_ps(x, other.x), _mm256_sub_ps(y, other.y), _mm256_sub_ps(z, other.z), _mm256_sub_ps(w, other.w) };
	}

	FINLINE vec4fx8 operator*(const vec4fx8& other) const {
		return vec4fx8{ _mm256_mul_ps(x, other.x), _mm256_mul_ps(y, other.y), _mm256_mul_ps(z, other.z), _mm256_mul_ps(w, other.w) };
	}

	FINLINE vec4fx8 operator/(const vec4fx8& other) const {
		return vec4fx8{ _mm256_div_ps(x, other.x), _mm256_div_ps(y, other.y), _mm256_div_ps(z, other.z), _mm256_div_ps(w, other.w) };
	}

	FINLINE vec4fx8 operator+(const floatx8& other) const {
		return vec4fx8{ _mm256_add_ps(x, other), _mm256_add_ps(y, other), _mm256_add_ps(z, other), _mm256_add_ps(w, other) };
	}

	FINLINE vec4fx8 operator-(const floatx8& other) const {
		return vec4fx8{ _mm256_sub_ps(x, other), _mm256_sub_ps(y, other), _mm256_sub_ps(z, other), _mm256_sub_ps(w, other) };
	}

	FINLINE vec4fx8 operator*(const floatx8& other) const {
		return vec4fx8{ _mm256_mul_ps(x, other), _mm256_mul_ps(y, other), _mm256_mul_ps(z, other), _mm256_mul_ps(w, other) };
	}

	FINLINE vec4fx8 operator/(const floatx8& other) const {
		const floatx8 lrcp = _mm256_rcp_ps(other);
		return vec4fx8{ _mm256_mul_ps(x, lrcp), _mm256_mul_ps(y, lrcp), _mm256_mul_ps(z, lrcp), _mm256_mul_ps(w, lrcp) };
	}

	FINLINE vec4fx8 rcp() const {
		return vec4fx8{ _mm256_rcp_ps(x), _mm256_rcp_ps(y), _mm256_rcp_ps(z), _mm256_rcp_ps(w) };
	}

	FINLINE floatx8 operator==(const vec4fx8& other) const {
		floatx8 cmpx = _mm256_cmp_ps(x, other.x, _CMP_EQ_UQ);
		floatx8 cmpy = _mm256_cmp_ps(y, other.y, _CMP_EQ_UQ);
		floatx8 cmpz = _mm256_cmp_ps(z, other.z, _CMP_EQ_UQ);
		floatx8 cmpw = _mm256_cmp_ps(w, other.w, _CMP_EQ_UQ);
		return _mm256_and_ps(cmpx, _mm256_and_ps(cmpy, _mm256_and_ps(cmpz, cmpw)));
	}
};

#pragma pack(pop)

FINLINE vec4fx8 vec4fx8_load(float* f) {
	return vec4fx8{ _mm256_load_ps(f), _mm256_load_ps(f + 8), _mm256_load_ps(f + 16), _mm256_load_ps(f + 24) };
}

FINLINE vec4fx8 operator+(const floatx8& left, const vec4fx8& right) {
	return right + left;
}

FINLINE vec4fx8 operator-(const floatx8& left, const vec4fx8& right) {
	return vec4fx8{ _mm256_sub_ps(left, right.x), _mm256_sub_ps(left, right.y), _mm256_sub_ps(left, right.z), _mm256_sub_ps(left, right.w) };
}

FINLINE vec4fx8 operator*(const floatx8& left, const vec4fx8& right) {
	return right * left;
}

FINLINE vec4fx8 operator/(const floatx8& left, const vec4fx8& right) {
	return vec4fx8{ _mm256_div_ps(left, right.x), _mm256_div_ps(left, right.y), _mm256_div_ps(left, right.z), _mm256_div_ps(left, right.w) };
}

FINLINE vec4fx8 fmadd(const vec4fx8& m1, const vec4fx8& m2, const vec4fx8& a) {
	return vec4fx8{ _mm256_fmadd_ps(m1.x, m2.x, a.x), _mm256_fmadd_ps(m1.y, m2.y, a.y), _mm256_fmadd_ps(m1.z, m2.z, a.z), _mm256_fmadd_ps(m1.w, m2.w, a.w) };
}

FINLINE vec4fx8 fmadd(const vec4fx8& m1, const vec4fx8& m2, const floatx8& a) {
	return vec4fx8{ _mm256_fmadd_ps(m1.x, m2.x, a), _mm256_fmadd_ps(m1.y, m2.y, a), _mm256_fmadd_ps(m1.z, m2.z, a), _mm256_fmadd_ps(m1.w, m2.w, a) };
}

FINLINE vec4fx8 fmadd(const floatx8& m1, const vec4fx8& m2, const vec4fx8& a) {
	return vec4fx8{ _mm256_fmadd_ps(m1, m2.x, a.x), _mm256_fmadd_ps(m1, m2.y, a.y), _mm256_fmadd_ps(m1, m2.z, a.z), _mm256_fmadd_ps(m1, m2.w, a.w) };
}

FINLINE vec4fx8 fmadd(const floatx8& m1, const vec4fx8& m2, const floatx8& a) {
	return vec4fx8{ _mm256_fmadd_ps(m1, m2.x, a), _mm256_fmadd_ps(m1, m2.y, a), _mm256_fmadd_ps(m1, m2.z, a), _mm256_fmadd_ps(m1, m2.w, a) };
}



FINLINE floatx8 dot(const vec3fx8& a, const vec3fx8& b) {
	return _mm256_fmadd_ps(a.z, b.z, _mm256_fmadd_ps(a.y, b.y, _mm256_mul_ps(a.x, b.x)));
}

FINLINE vec3fx8 cross(const vec3fx8& a, const vec3fx8& b) {
	floatx8 x = _mm256_fmsub_ps(a.y, b.z, _mm256_mul_ps(b.y, a.z));
	floatx8 y = _mm256_fmsub_ps(a.x, b.z, _mm256_mul_ps(b.x, a.z));
	floatx8 z = _mm256_fmsub_ps(a.x, b.y, _mm256_mul_ps(b.x, a.y));
	return vec3fx8{ x, y, z };
}

FINLINE floatx8 length(const vec3fx8& v) {
	return _mm256_sqrt_ps(dot(v, v));
}

FINLINE floatx8 length_sq(const vec3fx8& v) {
	return dot(v, v);
}

FINLINE vec3fx8 normalize(const vec3fx8& v) {
	return v / length(v);
}

FINLINE vec3fx8 floor(const vec3fx8& v) {
	return vec3fx8{ _mm256_round_ps(v.x, _MM_FROUND_FLOOR), _mm256_round_ps(v.y, _MM_FROUND_FLOOR), _mm256_round_ps(v.z, _MM_FROUND_FLOOR) };
}

FINLINE vec3fx8 clamp01(const vec3fx8& v) {
	return vec3fx8{
		_mm256_max_ps(_mm256_min_ps(v.x, _mm256_set1_ps(1.0F)), _mm256_setzero_ps()),
		_mm256_max_ps(_mm256_min_ps(v.y, _mm256_set1_ps(1.0F)), _mm256_setzero_ps()),
		_mm256_max_ps(_mm256_min_ps(v.z, _mm256_set1_ps(1.0F)), _mm256_setzero_ps())
	};
}

FINLINE vec3fx8 trunc(const vec3fx8& v) {
	return vec3fx8{ _mm256_round_ps(v.x, _MM_FROUND_TRUNC), _mm256_round_ps(v.y, _MM_FROUND_TRUNC), _mm256_round_ps(v.z, _MM_FROUND_TRUNC) };
}

FINLINE vec3fx8 max(const vec3fx8& a, const vec3fx8& b) {
	return vec3fx8{ _mm256_max_ps(a.x, b.x), _mm256_max_ps(a.y, b.y), _mm256_max_ps(a.z, b.z) };
}

FINLINE vec3fx8 max(const vec3fx8& v, const floatx8& f) {
	return vec3fx8{ _mm256_max_ps(v.x, f), _mm256_max_ps(v.y, f), _mm256_max_ps(v.z, f) };
}

FINLINE vec3fx8 min(const vec3fx8& a, const vec3fx8& b) {
	return vec3fx8{ _mm256_min_ps(a.x, b.x), _mm256_min_ps(a.y, b.y), _mm256_min_ps(a.z, b.z) };
}

FINLINE vec3fx8 min(const vec3fx8& v, const floatx8& f) {
	return vec3fx8{ _mm256_min_ps(v.x, f), _mm256_min_ps(v.y, f), _mm256_min_ps(v.z, f) };
}

FINLINE vec3fx8 blend(const vec3fx8& a, const vec3fx8& b, const floatx8& weight) {
	return vec3fx8{ _mm256_blendv_ps(a.x, b.x, weight), _mm256_blendv_ps(a.y, b.y, weight), _mm256_blendv_ps(a.z, b.z, weight) };
}


FINLINE floatx8 dot(const vec4fx8& a, const vec4fx8& b) {
	return _mm256_fmadd_ps(a.w, b.w, _mm256_fmadd_ps(a.z, b.z, _mm256_fmadd_ps(a.y, b.y, _mm256_mul_ps(a.x, b.x))));
}

FINLINE floatx8 length(const vec4fx8& v) {
	return _mm256_sqrt_ps(dot(v, v));
}

FINLINE floatx8 length_sq(const vec4fx8& v) {
	return dot(v, v);
}

FINLINE vec4fx8 normalize(const vec4fx8& v) {
	return v / length(v);
}

FINLINE vec4fx8 floor(const vec4fx8& v) {
	return vec4fx8{ _mm256_round_ps(v.x, _MM_FROUND_FLOOR), _mm256_round_ps(v.y, _MM_FROUND_FLOOR), _mm256_round_ps(v.z, _MM_FROUND_FLOOR), _mm256_round_ps(v.w, _MM_FROUND_FLOOR) };
}

FINLINE vec4fx8 clamp01(const vec4fx8& v) {
	return vec4fx8{
		_mm256_max_ps(_mm256_min_ps(v.x, _mm256_set1_ps(1.0F)), _mm256_setzero_ps()),
		_mm256_max_ps(_mm256_min_ps(v.y, _mm256_set1_ps(1.0F)), _mm256_setzero_ps()),
		_mm256_max_ps(_mm256_min_ps(v.z, _mm256_set1_ps(1.0F)), _mm256_setzero_ps()),
		_mm256_max_ps(_mm256_min_ps(v.w, _mm256_set1_ps(1.0F)), _mm256_setzero_ps())
	};
}

FINLINE vec4fx8 trunc(const vec4fx8& v) {
	return vec4fx8{ _mm256_round_ps(v.x, _MM_FROUND_TRUNC), _mm256_round_ps(v.y, _MM_FROUND_TRUNC), _mm256_round_ps(v.z, _MM_FROUND_TRUNC), _mm256_round_ps(v.w, _MM_FROUND_TRUNC) };
}

FINLINE vec4fx8 max(const vec4fx8& a, const vec4fx8& b) {
	return vec4fx8{ _mm256_max_ps(a.x, b.x), _mm256_max_ps(a.y, b.y), _mm256_max_ps(a.z, b.z), _mm256_max_ps(a.w, b.w) };
}

FINLINE vec4fx8 max(const vec4fx8& v, const floatx8& f) {
	return vec4fx8{ _mm256_max_ps(v.x, f), _mm256_max_ps(v.y, f), _mm256_max_ps(v.z, f), _mm256_max_ps(v.w, f) };
}

FINLINE vec4fx8 min(const vec4fx8& a, const vec4fx8& b) {
	return vec4fx8{ _mm256_min_ps(a.x, b.x), _mm256_min_ps(a.y, b.y), _mm256_min_ps(a.z, b.z), _mm256_min_ps(a.w, b.w) };
}

FINLINE vec4fx8 min(const vec4fx8& v, const floatx8& f) {
	return vec4fx8{ _mm256_min_ps(v.x, f), _mm256_min_ps(v.y, f), _mm256_min_ps(v.z, f), _mm256_min_ps(v.w, f) };
}

FINLINE vec4fx8 blend(const vec4fx8& a, const vec4fx8& b, const floatx8& weight) {
	return vec4fx8{ _mm256_blendv_ps(a.x, b.x, weight), _mm256_blendv_ps(a.y, b.y, weight), _mm256_blendv_ps(a.z, b.z, weight), _mm256_blendv_ps(a.w, b.w, weight) };
}


FINLINE floatx8 clamp01(const floatx8& f) {
	return _mm256_max_ps(_mm256_min_ps(f, _mm256_set1_ps(1.0F)), _mm256_setzero_ps());
}

FINLINE floatx8 fmadd(const floatx8& m1, const floatx8& m2, const floatx8& a) {
	return _mm256_fmadd_ps(m1, m2, a);
}

FINLINE floatx8 blend(const floatx8& a, const floatx8& b, const floatx8& weight) {
	return _mm256_blendv_ps(a, b, weight);
}

FINLINE floatx8 operator==(const floatx8& a, const floatx8& b) {
	return _mm256_cmp_ps(a, b, _CMP_EQ_UQ);
}

FINLINE uint32x4 cvt_int64x4_int32x4(uint64x4 vec) {
	floatx4 low = _mm_castsi128_ps(_mm256_castsi256_si128(vec));
	floatx4 high = _mm_castsi128_ps(_mm256_extracti128_si256(vec, 1));
	return _mm_castps_si128(_mm_shuffle_ps(low, high, _MM_SHUFFLE(2, 0, 2, 0)));
}

FINLINE float horizontal_sum(const floatx8& f) {
	floatx4 bestErrorLow = _mm256_castps256_ps128(f);
	floatx4 bestErrorHigh = _mm256_extractf128_ps(f, 1);
	floatx4 sum = _mm_add_ps(bestErrorLow, bestErrorHigh);
	sum = _mm_add_ps(sum, _mm_permute_ps(sum, _MM_PERM_CDCD));
	sum = _mm_add_ps(sum, _mm_permute_ps(sum, _MM_PERM_BBBB));
	return _mm_cvtss_f32(sum);
}


// Function alternatives for the operator overloads.
FINLINE vec4fx8 add(const vec4fx8& a, const vec4fx8& b) {
	return a + b;
}

FINLINE vec3fx8 add(const vec3fx8& a, const vec3fx8& b) {
	return a + b;
}

FINLINE vec3fx8 add(const vec3fx8& a, const floatx8& b) {
	return a + b;
}

FINLINE vec4fx8 add(const vec4fx8& a, const floatx8& b) {
	return a + b;
}

FINLINE vec4fx8 add(const floatx8& a, const vec4fx8& b) {
	return a + b;
}

FINLINE vec3fx8 add(const floatx8& a, const vec3fx8& b) {
	return a + b;
}

FINLINE floatx8 add(const floatx8& a, const floatx8& b) {
	return _mm256_add_ps(a, b);
}

FINLINE vec4fx8 sub(const vec4fx8& a, const vec4fx8& b) {
	return a - b;
}

FINLINE vec3fx8 sub(const vec3fx8& a, const vec3fx8& b) {
	return a - b;
}

FINLINE vec4fx8 sub(const vec4fx8& a, const floatx8& b) {
	return a - b;;
}

FINLINE vec3fx8 sub(const vec3fx8& a, const floatx8& b) {
	return a - b;
}

FINLINE vec4fx8 sub(const floatx8& a, const vec4fx8& b) {
	return a - b;
}

FINLINE vec3fx8 sub(const floatx8& a, const vec3fx8& b) {
	return a - b;
}

FINLINE floatx8 sub(const floatx8& a, const floatx8& b) {
	return _mm256_sub_ps(a, b);
}

FINLINE vec4fx8 mul(const vec4fx8& a, const vec4fx8& b) {
	return a * b;
}

FINLINE vec3fx8 mul(const vec3fx8& a, const vec3fx8& b) {
	return a * b;
}

FINLINE vec4fx8 mul(const vec4fx8& a, const floatx8& b) {
	return a * b;
}

FINLINE vec3fx8 mul(const vec3fx8& a, const floatx8& b) {
	return a * b;
}

FINLINE vec4fx8 mul(const floatx8& a, const vec4fx8& b) {
	return a * b;
}

FINLINE vec3fx8 mul(const floatx8& a, const vec3fx8& b) {
	return a * b;
}

FINLINE floatx8 mul(const floatx8& a, const floatx8& b) {
	return _mm256_mul_ps(a, b);
}

FINLINE vec4fx8 div(const vec4fx8& a, const vec4fx8& b) {
	return a / b;
}

FINLINE vec3fx8 div(const vec3fx8& a, const vec3fx8& b) {
	return a / b;
}

FINLINE vec4fx8 div(const vec4fx8& a, const floatx8& b) {
	return a / b;
}

FINLINE vec3fx8 div(const vec3fx8& a, const floatx8& b) {
	return a / b;
}

FINLINE vec4fx8 div(const floatx8& a, const vec4fx8& b) {
	return a / b;
}

FINLINE vec3fx8 div(const floatx8& a, const vec3fx8& b) {
	return a / b;
}

FINLINE floatx8 div(const floatx8& a, const floatx8& b) {
	return _mm256_div_ps(a, b);
}

FINLINE vec4fx8 rcp(const vec4fx8& v) {
	return v.rcp();
}

FINLINE vec3fx8 rcp(const vec3fx8& v) {
	return v.rcp();
}

FINLINE floatx8 rcp(const floatx8& f) {
	return _mm256_rcp_ps(f);
}

FINLINE floatx8 trunc(const floatx8& f) {
	return _mm256_round_ps(f, _MM_FROUND_TRUNC);
}

// https://stackoverflow.com/questions/41315420/how-to-implement-sign-function-with-sse3
FINLINE floatx8 signum(const floatx8& f) {
	floatx8 nonZero = _mm256_cmp_ps(f, _mm256_setzero_ps(), _CMP_NEQ_UQ);
	floatx8 signBit = _mm256_and_ps(f, _mm256_set1_ps(-0.0F));
	floatx8 zeroOne = _mm256_and_ps(nonZero, _mm256_set1_ps(1.0F));
	return _mm256_or_ps(zeroOne, signBit);
}

FINLINE floatx8 normalize(const floatx8& f) {
	return signum(f);
}

FINLINE floatx8 equals_zero(const vec4fx8& v) {
	return v.equal_zero();
}

FINLINE floatx8 equals_zero(const vec3fx8& v) {
	return v.equal_zero();
}

FINLINE floatx8 equals_zero(const floatx8& v) {
	return _mm256_cmp_ps(v, _mm256_setzero_ps(), _CMP_EQ_UQ);
}

FINLINE floatx8 length_sq(const floatx8& v) {
	return _mm256_mul_ps(v, v);
}

FINLINE uint32x8 blend(const uint32x8& a, const uint32x8& b, const floatx8& weight) {
	return _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b), weight));
}

FINLINE void extract_lo_hi_masks(const floatx8& mask, floatx8* lo, floatx8* hi) {
	*lo = _mm256_castsi256_ps(_mm256_cvtepi32_epi64(_mm256_castsi256_si128(_mm256_castps_si256(mask))));
	*hi = _mm256_castsi256_ps(_mm256_cvtepi32_epi64(_mm256_extracti128_si256(_mm256_castps_si256(mask), 1)));
}