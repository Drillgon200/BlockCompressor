#pragma once

#include <intrin.h>
#include "BC7Tables.h"
#include "JobSystem.h"

#pragma pack(push, 1)

struct BC7Block {
	uint64_t data[2];
};

struct alignas(32) BC7Blockx4 {
	uint64x4 data[2];
};

#pragma pack(pop)

//Useful resources
//https://registry.khronos.org/DataFormat/specs/1.3/dataformat.1.3.html#bptc_bc7
//https://docs.microsoft.com/en-us/windows/win32/direct3d11/bc7-format
//https://docs.microsoft.com/en-us/windows/win32/direct3d11/bc7-format-mode-reference

//Alright I got tired of doing vector math without a vector class, so I'll just write a couple short implementations here
struct vec3f {
	union {
		float components[3];
		struct {
			float x;
			float y;
			float z;
		};
	};
	vec3f() {}

	constexpr vec3f(float v) : x{ v }, y{ v }, z{ v } {
	}

	constexpr vec3f(float x, float y, float z) : x{ x }, y{ y }, z{ z } {
	}

	vec3f operator+(vec3f other) {
		return vec3f{ x + other.x, y + other.y, z + other.z };
	}
	vec3f operator-(vec3f other) {
		return vec3f{ x - other.x, y - other.y, z - other.z };
	}
	vec3f operator*(vec3f other) {
		return vec3f{ x * other.x, y * other.y, z * other.z };
	}
	vec3f operator/(vec3f other) {
		return vec3f{ x / other.x, y / other.y, z / other.z };
	}
	vec3f operator+(float other) {
		return vec3f{ x + other, y + other, z + other };
	}
	vec3f operator-(float other) {
		return vec3f{ x - other, y - other, z - other };
	}
	vec3f operator-() {
		return vec3f{ -x, -y, -z };
	}
	vec3f operator*(float other) {
		return vec3f{ x * other, y * other, z * other };
	}
	vec3f operator/(float other) {
		return vec3f{ x / other, y / other, z / other };
	}

	void operator+=(vec3f other) {
		x += other.x; y += other.y; z += other.z;
	}
	void operator-=(vec3f other) {
		x -= other.x; y -= other.y; z -= other.z;
	}
	void operator*=(vec3f other) {
		x *= other.x; y *= other.y; z *= other.z;
	}
	void operator/=(vec3f other) {
		x /= other.x; y /= other.y; z /= other.z;
	}
	void operator+=(float other) {
		x += other; y += other; z += other;
	}
	void operator-=(float other) {
		x -= other; y -= other; z -= other;
	}
	void operator*=(float other) {
		x *= other; y *= other; z *= other;
	}
	void operator/=(float other) {
		x /= other; y /= other; z /= other;
	}

	bool operator==(vec3f other) {
		return (x == other.x) & (y == other.y) & (z == other.z);
	}
	bool operator!=(vec3f other) {
		return (x != other.x) | (y != other.y) | (z != other.z);
	}
};

struct vec4f {
	union {
		float components[4];
		struct {
			float x;
			float y;
			float z;
			float w;
		};
	};

	vec4f(){}

	constexpr vec4f(const vec3f& vec, float w) : x{ vec.x }, y{ vec.y }, z{ vec.z }, w{ w } {
	}

	constexpr vec4f(float v) : x{ v }, y{ v }, z{ v }, w{ v } {
	}

	constexpr vec4f(float x, float y, float z, float w) : x{ x }, y{ y }, z{ z }, w{ w } {
	}

	vec4f operator+(vec4f other) {
		return vec4f{ x + other.x, y + other.y, z + other.z, w + other.w };
	}
	vec4f operator-(vec4f other) {
		return vec4f{ x - other.x, y - other.y, z - other.z, w - other.w };
	}
	vec4f operator*(vec4f other) {
		return vec4f{ x * other.x, y * other.y, z * other.z, w * other.w };
	}
	vec4f operator/(vec4f other) {
		return vec4f{ x / other.x, y / other.y, z / other.z, w / other.w };
	}
	vec4f operator+(float other) {
		return vec4f{ x + other, y + other, z + other, w + other };
	}
	vec4f operator-(float other) {
		return vec4f{ x - other, y - other, z - other, w - other };
	}
	vec4f operator-() {
		return vec4f{ -x, -y, -z, -w };
	}
	vec4f operator*(float other) {
		return vec4f{ x * other, y * other, z * other, w * other };
	}
	vec4f operator/(float other) {
		return vec4f{ x / other, y / other, z / other, w / other };
	}

	void operator+=(float other) {
		x += other;
		y += other;
		z += other;
		w += other;
	}

	void operator+=(vec4f other) {
		x += other.x;
		y += other.y;
		z += other.z;
		w += other.w;
	}

	vec3f xyz() {
		return vec3f{ x, y, z };
	}

	bool operator==(vec4f other) {
		return x == other.x && y == other.y && z == other.z && w == other.w;
	}

	bool operator!=(vec4f other) {
		return x != other.x || y != other.y || z != other.z || w != other.w;
	}
};

vec3f operator+(float l, vec3f r) {
	return r + l;
}

vec3f operator-(float l, vec3f r) {
	return vec3f{ l - r.x, l - r.y, l - r.z };
}

vec3f operator*(float l, vec3f r) {
	return r * l;
}

vec3f operator/(float l, vec3f r) {
	return vec3f{ l / r.x, l / r.y, l / r.z };
}

vec4f operator+(float l, vec4f r) {
	return r + l;
}

vec4f operator-(float l, vec4f r) {
	return vec4f{ l - r.x, l - r.y, l - r.z, l - r.z };
}

vec4f operator*(float l, vec4f r) {
	return r * l;
}

vec4f operator/(float l, vec4f r) {
	return vec4f{ l / r.x, l / r.y, l / r.z, l / r.w };
}

float dot(vec3f a, vec3f b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

float dot(vec4f a, vec4f b) {
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * a.w;
}

vec3f cross(vec3f a, vec3f b) {
	return vec3f{ a.y * b.z - b.y * a.z, a.x * b.z - b.x * a.z, a.x * b.y - b.x * a.y };
}

float length(vec3f v) {
	return std::sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

float length_sq(vec3f v) {
	return v.x * v.x + v.y * v.y + v.z * v.z;
}

float length(vec4f v) {
	return std::sqrtf(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
}

float length_sq(vec4f v) {
	return v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
}

float length(float v) {
	return fabs(v);
}

float length_sq(float v) {
	return v * v;
}

vec3f normalize(vec3f v) {
	float invLen = 1.0F / std::sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
	return vec3f{ v.x * invLen, v.y * invLen, v.z * invLen };
}

vec3f floor(vec3f v) {
	return vec3f{ floor(v.x), floor(v.y), floor(v.z) };
}

vec3f clamp01(vec3f v) {
	return vec3f{ clamp01(v.x), clamp01(v.y), clamp01(v.z) };
}

vec3f truncf(vec3f v) {
	return vec3f{ truncf(v.x), truncf(v.y), truncf(v.z) };
}

vec4f normalize(vec4f v) {
	float invLen = 1.0F / std::sqrtf(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
	return vec4f{ v.x * invLen, v.y * invLen, v.z * invLen, v.w * invLen };
}

vec4f floor(vec4f v) {
	return vec4f{ floor(v.x), floor(v.y), floor(v.z), floor(v.w) };
}

vec4f clamp01(vec4f v) {
	return vec4f{ clamp01(v.x), clamp01(v.y), clamp01(v.z), clamp01(v.w) };
}

vec4f truncf(vec4f v) {
	return vec4f{ truncf(v.x), truncf(v.y), truncf(v.z), truncf(v.w) };
}

int32_t signum(float val) {
	return (0.0F < val) - (0.0F > val);
}

float normalize(float f) {
	return signum(f);
}

namespace std {
	vec3f min(vec3f a, vec3f b) {
		return vec3f{ std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z) };
	}
	vec3f max(vec3f a, vec3f b) {
		return vec3f{ std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z) };
	}
	vec3f min(vec3f a, float b) {
		return vec3f{ std::min(a.x, b), std::min(a.y, b), std::min(a.z, b) };
	}
	vec3f max(vec3f a, float b) {
		return vec3f{ std::max(a.x, b), std::max(a.y, b), std::max(a.z, b) };
	}
}

FINLINE uint32x8 bc7_interpolate(const uint32x8& e0, const uint32x8& e1, const uint32x8& interpolationFactor) {
	const uint32x8 leftInterp = _mm256_mullo_epi32(_mm256_sub_epi32(_mm256_set1_epi32(64), interpolationFactor), e0);
	const uint32x8 rightInterp = _mm256_mullo_epi32(interpolationFactor, e1);
	return _mm256_srli_epi32(_mm256_add_epi32(_mm256_add_epi32(leftInterp, rightInterp), _mm256_set1_epi32(32)), 6);
}

template<typename T>
T bc7_interpolatex8(const T& a, const T& b, const uint32x8& interpolationFactor, const uint32x8& interpolationFactorAlpha) {
	static_assert(typeid(T) == typeid(floatx8) || typeid(T) == typeid(vec3fx8) || typeid(T) == typeid(vec4fx8), "Implement type");
}

template<>
FINLINE floatx8 bc7_interpolatex8(const floatx8& a, const floatx8& b, const uint32x8& interpolationFactor, const uint32x8& interpolationFactorAlpha) {
	return _mm256_cvtepi32_ps(bc7_interpolate(_mm256_cvtps_epi32(a), _mm256_cvtps_epi32(b), interpolationFactor));
}

template<>
FINLINE vec3fx8 bc7_interpolatex8(const vec3fx8& a, const vec3fx8& b, const uint32x8& interpolationFactor, const uint32x8& interpolationFactorAlpha) {
	return vec3fx8{
		_mm256_cvtepi32_ps(bc7_interpolate(_mm256_cvtps_epi32(a.x), _mm256_cvtps_epi32(b.x), interpolationFactor)),
		_mm256_cvtepi32_ps(bc7_interpolate(_mm256_cvtps_epi32(a.y), _mm256_cvtps_epi32(b.y), interpolationFactor)),
		_mm256_cvtepi32_ps(bc7_interpolate(_mm256_cvtps_epi32(a.z), _mm256_cvtps_epi32(b.z), interpolationFactor))
	};
}

template<>
FINLINE vec4fx8 bc7_interpolatex8(const vec4fx8& a, const vec4fx8& b, const uint32x8& interpolationFactor, const uint32x8& interpolationFactorAlpha) {
	return vec4fx8{
		_mm256_cvtepi32_ps(bc7_interpolate(_mm256_cvtps_epi32(a.x), _mm256_cvtps_epi32(b.x), interpolationFactor)),
		_mm256_cvtepi32_ps(bc7_interpolate(_mm256_cvtps_epi32(a.y), _mm256_cvtps_epi32(b.y), interpolationFactor)),
		_mm256_cvtepi32_ps(bc7_interpolate(_mm256_cvtps_epi32(a.z), _mm256_cvtps_epi32(b.z), interpolationFactor)),
		_mm256_cvtepi32_ps(bc7_interpolate(_mm256_cvtps_epi32(a.w), _mm256_cvtps_epi32(b.w), interpolationFactorAlpha))
	};
}

template<typename Endpoint>
FINLINE floatx8 get_distance_between_endpoints(const Endpoint endpoints[2], const Endpoint& pixel) {
	Endpoint endpointVector = endpoints[1] - endpoints[0];
	Endpoint pixelVector = pixel - endpoints[0];
	floatx8 project = dot(endpointVector, pixelVector);
	return clamp01(_mm256_div_ps(project, length_sq(endpointVector)));
}

template<uint32_t bits>
FINLINE floatx8 quantize_to_index(const vec4fx8 endpoints[2], const vec4fx8& pixel) {
	constexpr float scale = static_cast<float>((1 << bits) - 1);
	floatx8 normalizedProject = get_distance_between_endpoints(endpoints, pixel);
	// trunc((normalizedProject * scale + 0.5)) / scale
	return _mm256_mul_ps(_mm256_round_ps(_mm256_add_ps(_mm256_mul_ps(normalizedProject, _mm256_set1_ps(scale)), _mm256_set1_ps(0.5F)), _MM_FROUND_TRUNC), _mm256_set1_ps(1.0F / scale));
}

template<uint32_t bits>
FINLINE floatx8 quantize_to_index(const vec3fx8 endpoints[2], const vec3fx8& pixel) {
	constexpr float scale = static_cast<float>((1 << bits) - 1);
	floatx8 normalizedProject = get_distance_between_endpoints(endpoints, pixel);
	// trunc((normalizedProject * scale + 0.5)) / scale
	return _mm256_mul_ps(_mm256_round_ps(_mm256_add_ps(_mm256_mul_ps(normalizedProject, _mm256_set1_ps(scale)), _mm256_set1_ps(0.5F)), _MM_FROUND_TRUNC), _mm256_set1_ps(1.0F / scale));
}

template<uint32_t bits>
FINLINE floatx8 quantize_to_index(const floatx8 endpoints[2], const floatx8& pixel) {
	constexpr float scale = static_cast<float>((1 << bits) - 1);
	floatx8 normalizedProject = _mm256_div_ps(_mm256_sub_ps(pixel, endpoints[0]), _mm256_sub_ps(endpoints[1], endpoints[0]));
	// trunc((normalizedProject * scale + 0.5)) / scale
	return _mm256_mul_ps(_mm256_round_ps(_mm256_add_ps(_mm256_mul_ps(normalizedProject, _mm256_set1_ps(scale)), _mm256_set1_ps(0.5F)), _MM_FROUND_TRUNC), _mm256_set1_ps(1.0F / scale));
}

template<uint32_t bits>
FINLINE uint32x8 find_index(const vec4fx8 endpoints[2], const vec4fx8& pixel) {
	constexpr float scale = static_cast<float>((1 << bits) - 1);
	floatx8 normalizedProject = get_distance_between_endpoints(endpoints, pixel);
	// (uint32_t) trunc(normalizedProject * scale * 0.5)
	return _mm256_cvtps_epi32(_mm256_round_ps(_mm256_add_ps(_mm256_mul_ps(normalizedProject, _mm256_set1_ps(scale)), _mm256_set1_ps(0.5F)), _MM_FROUND_TRUNC));
}

template<uint32_t bits>
FINLINE uint32x8 find_index(const vec3fx8 endpoints[2], const vec3fx8& pixel) {
	constexpr float scale = static_cast<float>((1 << bits) - 1);
	floatx8 normalizedProject = get_distance_between_endpoints(endpoints, pixel);
	// (uint32_t) trunc(normalizedProject * scale * 0.5)
	return _mm256_cvtps_epi32(_mm256_round_ps(_mm256_add_ps(_mm256_mul_ps(normalizedProject, _mm256_set1_ps(scale)), _mm256_set1_ps(0.5F)), _MM_FROUND_TRUNC));
}

template<uint32_t bits>
FINLINE uint32x8 find_index(const floatx8 endpoints[2], const floatx8& pixel) {
	constexpr float scale = static_cast<float>((1 << bits) - 1);
	floatx8 normalizedProject = _mm256_div_ps(_mm256_sub_ps(pixel, endpoints[0]), _mm256_sub_ps(endpoints[1], endpoints[0]));
	// (uint32_t) trunc(normalizedProject * scale * 0.5)
	return _mm256_cvtps_epi32(_mm256_round_ps(_mm256_add_ps(_mm256_mul_ps(normalizedProject, _mm256_set1_ps(scale)), _mm256_set1_ps(0.5F)), _MM_FROUND_TRUNC));
}


uint8_t bc7_interpolate(uint8_t e0, uint8_t e1, uint32_t interpolationFactor) {
	return ((64 - interpolationFactor) * e0 + interpolationFactor * e1 + 32) >> 6;
}

template<typename T>
T bc7_interpolate(T a, T b, uint32_t interpolationFactor, uint32_t interpolationFactorAlpha) {
	static_assert(typeid(T) == typeid(float) || typeid(T) == typeid(vec3f) || typeid(T) == typeid(vec4f), "Implement type");
}

template<>
float bc7_interpolate(float a, float b, uint32_t interpolationFactor, uint32_t interpolationFactorAlpha) {
	return static_cast<float>(bc7_interpolate(static_cast<uint8_t>(a), static_cast<uint8_t>(b), interpolationFactor));
}

template<>
vec3f bc7_interpolate(vec3f a, vec3f b, uint32_t interpolationFactor, uint32_t interpolationFactorAlpha) {
	return vec3f{
		bc7_interpolate(a.x, b.x, interpolationFactor, 0),
		bc7_interpolate(a.y, b.y, interpolationFactor, 0),
		bc7_interpolate(a.z, b.z, interpolationFactor, 0)
	};
}

template<>
vec4f bc7_interpolate(vec4f a, vec4f b, uint32_t interpolationFactor, uint32_t interpolationFactorAlpha) {
	return vec4f{
		bc7_interpolate(a.x, b.x, interpolationFactor, 0),
		bc7_interpolate(a.y, b.y, interpolationFactor, 0),
		bc7_interpolate(a.z, b.z, interpolationFactor, 0),
		bc7_interpolate(a.w, b.w, interpolationFactorAlpha, 0)
	};
}

template<typename Vec>
float get_distance_between_endpoints(Vec endpoints[2], Vec pixel) {
	Vec endpointVector = endpoints[1] - endpoints[0];
	Vec pixelVector = pixel - endpoints[0];
	float project = dot(endpointVector, pixelVector);
	return clamp01(project / length_sq(endpointVector));
}

template<uint32_t bits>
float quantize_to_index(vec4f endpoints[2], vec4f pixel) {
	constexpr float scale = static_cast<float>((1 << bits) - 1);
	constexpr float invScale = 1.0F / scale;
	float normalizedProject = get_distance_between_endpoints(endpoints, pixel);
	return truncf(normalizedProject * scale + 0.5) * invScale;
}

template<uint32_t bits>
float quantize_to_index(vec3f endpoints[2], vec3f pixel) {
	constexpr float scale = static_cast<float>((1 << bits) - 1);
	constexpr float invScale = 1.0F / scale;
	float normalizedProject = get_distance_between_endpoints(endpoints, pixel);
	return truncf(normalizedProject * scale + 0.5) * invScale;
}

template<uint32_t bits>
float quantize_to_index(float endpoints[2], float pixel) {
	constexpr float scale = static_cast<float>((1 << bits) - 1);
	constexpr float invScale = 1.0F / scale;
	float normalizedProject = clamp01((pixel - endpoints[0]) / (endpoints[1] - endpoints[0]));
	return truncf(normalizedProject * scale + 0.5) * invScale;
}

template<uint32_t bits>
uint32_t find_index(vec4f endpoints[2], vec4f pixel) {
	constexpr float scale = static_cast<float>((1 << bits) - 1);
	float normalizedProject = get_distance_between_endpoints(endpoints, pixel);
	return static_cast<uint32_t>(truncf(normalizedProject * scale + 0.5F));
}

template<uint32_t bits>
uint32_t find_index(vec3f endpoints[2], vec3f pixel) {
	constexpr float scale = static_cast<float>((1 << bits) - 1);
	float normalizedProject = get_distance_between_endpoints(endpoints, pixel);
	return static_cast<uint32_t>(truncf(normalizedProject * scale + 0.5F));
}

template<uint32_t bits>
uint32_t find_index(float endpoints[2], float pixel) {
	constexpr float scale = static_cast<float>((1 << bits) - 1);
	float normalizedProject = clamp01((pixel - endpoints[0]) / (endpoints[1] - endpoints[0]));
	return static_cast<uint32_t>(truncf(normalizedProject * scale + 0.5F));
}

void quantize_bc7_endpoints_mode0(vec3f endpoints[6]) {
	for (uint32_t endpointIndex = 0; endpointIndex < 6; endpointIndex++) {
		vec3f& endpoint = endpoints[endpointIndex];
		//Find best p bits to compress this endpoint
		float bestError = FLT_MAX;
		vec3f bestQuantizedEndpoint;
		for (float pBit = 0; pBit < 2; pBit++) {
			float error = 0;
			vec3f quantizedEndpoint;
			for (uint32_t component = 0; component < 3; component++) {
				//4 bits
				float quantized = floor(clamp01(endpoint.components[component]) * 15.0F + 0.5F);
				//Add p bit
				quantized = quantized * 2.0F + pBit;
				//Put bottom 3 bits in
				quantized = quantized * 8.0F + floor(quantized / 4.0F);

				float err = endpoint.components[component] - quantized / 255.0F;
				error += err * err;
				quantizedEndpoint.components[component] = quantized;
			}
			if (error < bestError) {
				bestQuantizedEndpoint = quantizedEndpoint;
				bestError = error;
			}
		}
		endpoints[endpointIndex] = bestQuantizedEndpoint;
	}
}

float error_mode0(vec4f pixels[16], uint32_t partitionTable, vec3f endpoints[6], uint64_t indices) {
	const BC7PartitionTable& table = bc7PartitionTable3Subsets[partitionTable];
	float error = 0.0F;
	for (uint32_t pixel = 0; pixel < 16; pixel++) {
		uint32_t partition = table.partitionNumbers[pixel];
		for (uint32_t component = 0; component < 3; component++) {
			float compressedComponent = static_cast<float>(bc7_interpolate(static_cast<uint8_t>(endpoints[partition * 2 + 0].components[component]), static_cast<uint8_t>(endpoints[partition * 2 + 1].components[component]), bc7InterpolationFactors3[(indices >> (pixel * 3)) & 0b111])) / 255.0F;
			float err = compressedComponent - pixels[pixel].components[component];
			error += err * err;
		}
	}
	return error;
}

float ray_cast_unit_box(float pos, float dir) {
	float invDir = 1.0F / dir;
	float tMin = (-pos) * invDir;
	float tMax = (1.0F - pos) * invDir;
	return std::max(tMin, tMax);
}

float ray_cast_unit_box(vec3f pos, vec3f dir) {
	vec3f invDir = 1.0F / dir;

	float xTMin = (-pos.x) * invDir.x;
	float xTMax = (1.0F - pos.x) * invDir.x;
	float yTMin = (-pos.y) * invDir.y;
	float yTMax = (1.0F - pos.y) * invDir.y;
	float zTMin = (-pos.z) * invDir.z;
	float zTMax = (1.0F - pos.z) * invDir.z;

	return std::min(std::max(xTMax, xTMin), std::min(std::max(yTMax, yTMin), std::max(zTMax, zTMin)));
}

float ray_cast_unit_box(vec4f pos, vec4f dir) {
	vec4f invDir = 1.0F / dir;

	float xTMin = (-pos.x) * invDir.x;
	float xTMax = (1.0F - pos.x) * invDir.x;
	float yTMin = (-pos.y) * invDir.y;
	float yTMax = (1.0F - pos.y) * invDir.y;
	float zTMin = (-pos.z) * invDir.z;
	float zTMax = (1.0F - pos.z) * invDir.z;
	float wTMin = (-pos.w) * invDir.w;
	float wTMax = (1.0F - pos.w) * invDir.w;

	return std::min(std::max(xTMax, xTMin), std::min(std::max(yTMax, yTMin), std::min(std::max(zTMax, zTMin), std::max(wTMax, wTMin))));
}

floatx8 ray_cast_unit_box(const floatx8& pos, const floatx8& dir) {
	floatx8 invDir = _mm256_rcp_ps(dir);
	floatx8 tMin = _mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), pos), invDir);
	floatx8 tMax = _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(1.0F), pos), invDir);
	return _mm256_max_ps(tMin, tMax);
}

floatx8 ray_cast_unit_box(const vec3fx8& pos, const vec3fx8& dir) {
	vec3fx8 invDir = dir.rcp();

	floatx8 xTMin = _mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), pos.x), invDir.x);
	floatx8 xTMax = _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(1.0F), pos.x), invDir.x);
	floatx8 yTMin = _mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), pos.y), invDir.y);
	floatx8 yTMax = _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(1.0F), pos.y), invDir.y);
	floatx8 zTMin = _mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), pos.z), invDir.z);
	floatx8 zTMax = _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(1.0F), pos.z), invDir.z);

	return _mm256_min_ps(_mm256_max_ps(xTMax, xTMin), _mm256_min_ps(_mm256_max_ps(yTMax, yTMin), _mm256_max_ps(zTMax, zTMin)));
}

floatx8 ray_cast_unit_box(const vec4fx8& pos, const vec4fx8& dir) {
	vec4fx8 invDir = dir.rcp();

	floatx8 xTMin = _mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), pos.x), invDir.x);
	floatx8 xTMax = _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(1.0F), pos.x), invDir.x);
	floatx8 yTMin = _mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), pos.y), invDir.y);
	floatx8 yTMax = _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(1.0F), pos.y), invDir.y);
	floatx8 zTMin = _mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), pos.z), invDir.z);
	floatx8 zTMax = _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(1.0F), pos.z), invDir.z);
	floatx8 wTMin = _mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), pos.w), invDir.w);
	floatx8 wTMax = _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(1.0F), pos.w), invDir.w);

	return _mm256_min_ps(_mm256_max_ps(xTMax, xTMin), _mm256_min_ps(_mm256_max_ps(yTMax, yTMin), _mm256_min_ps(_mm256_max_ps(zTMax, zTMin), _mm256_max_ps(wTMax, wTMin))));
}

// This one does one partition at a time
template<typename Endpoint, uint32_t indexResolution>
void least_squares_optimize_endpoints(vec4fx8 pixels[16], Endpoint endpoints[2], const BC7PartitionTable& table, uint32_t partition) {
	floatx8 endpointsEqual = endpoints[0] == endpoints[1];
	if (_mm256_movemask_ps(endpointsEqual) == 0xFF) {
		// All of them have no axis to optimize along
		return;
	}
	floatx8 alphaSq{};
	floatx8 alphaBeta{};
	floatx8 betaSq{};
	Endpoint alphaX{};
	Endpoint betaX{};
	for (uint32_t i = 0; i < 16; i++) {
		if (table.partitionNumbers[i] != partition) {
			continue;
		}
		vec3fx8 pixel;
		if constexpr (std::is_same<vec3fx8, Endpoint>::value) {
			pixel = pixels[i].xyz();
		} else if constexpr (std::is_same<vec4fx8, Endpoint>::value) {
			pixel = pixels[i];
		} else {
			pixel = pixels[i].w;
		}

		floatx8 alpha = quantize_to_index<indexResolution>(endpoints, pixel);
		floatx8 beta = _mm256_sub_ps(_mm256_set1_ps(1.0F), alpha);

		alphaSq = _mm256_fmadd_ps(alpha, alpha, alphaSq);
		alphaBeta = _mm256_fmadd_ps(alpha, beta, alphaBeta);
		betaSq = _mm256_fmadd_ps(beta, beta, betaSq);

		alphaX = fmadd(alpha, pixel, alphaX);
		betaX = fmadd(beta, pixel, betaX);
	}

	floatx8 inverseDeterminant = _mm256_rcp_ps(_mm256_fmsub_ps(alphaSq, betaSq, _mm256_mul_ps(alphaBeta, alphaBeta)));
	alphaBeta = _mm256_sub_ps(_mm256_setzero_ps(), _mm256_mul_ps(alphaBeta, inverseDeterminant));
	alphaSq = _mm256_mul_ps(alphaSq, inverseDeterminant);
	betaSq = _mm256_mul_ps(betaSq, inverseDeterminant);

	endpoints[0] = blend(fmadd(betaSq, alphaX, alphaBeta * betaX), endpoints[0], endpointsEqual);
	endpoints[1] = blend(fmadd(alphaBeta, alphaX, alphaSq * betaX), endpoints[1], endpointsEqual);
}

// This one does all partitions at once
template<typename Endpoint, uint32_t indexResolution, uint32_t partitions>
void least_squares_optimize_endpointsx8(vec4fx8 pixels[16], Endpoint endpoints[6], const BC7PartitionTable& table) {
	floatx8 endpointsEqual[3]{ endpoints[0] == endpoints[1], endpoints[2] == endpoints[3], endpoints[4] == endpoints[5] };
	uint32_t endpointsEqualMasks[3]{ _mm256_movemask_ps(endpointsEqual[0]), _mm256_movemask_ps(endpointsEqual[1]), _mm256_movemask_ps(endpointsEqual[2]) };
	if ((endpointsEqualMasks[0] & endpointsEqualMasks[1] & endpointsEqualMasks[2]) == 0xFF) {
		// All of them have no axis to optimize along
		return;
	}
	floatx8 alphaSq[partitions]{};
	floatx8 alphaBeta[partitions]{};
	floatx8 betaSq[partitions]{};
	Endpoint alphaX[partitions]{};
	Endpoint betaX[partitions]{};
	for (uint32_t i = 0; i < 16; i++) {
		uint32_t partition = table.partitionNumbers[i];
		if (endpointsEqualMasks[partition] == 0xFF) {
			continue;
		}
		Endpoint pixel;
		if constexpr (std::is_same<vec3fx8, Endpoint>::value) {
			pixel = pixels[i].xyz();
		} else if constexpr (std::is_same<vec4fx8, Endpoint>::value) {
			pixel = pixels[i];
		} else {
			pixel = pixels[i].w;
		}

		floatx8 alpha = quantize_to_index<indexResolution>(&endpoints[partition * 2], pixel);
		floatx8 beta = _mm256_sub_ps(_mm256_set1_ps(1.0F), alpha);

		alphaSq[partition] = _mm256_fmadd_ps(alpha, alpha, alphaSq[partition]);
		alphaBeta[partition] = _mm256_fmadd_ps(alpha, beta, alphaBeta[partition]);
		betaSq[partition] = _mm256_fmadd_ps(beta, beta, betaSq[partition]);

		alphaX[partition] = fmadd(alpha, pixel, alphaX[partition]);
		betaX[partition] = fmadd(beta, pixel, betaX[partition]);
	}

	for (uint32_t i = 0; i < partitions; i++) {
		if (endpointsEqualMasks[i] == 0xFF) {
			continue;
		}
		floatx8 inverseDeterminant = _mm256_rcp_ps(_mm256_fmsub_ps(alphaSq[i], betaSq[i], _mm256_mul_ps(alphaBeta[i], alphaBeta[i])));
		floatx8 alphaBetaInv = _mm256_sub_ps(_mm256_setzero_ps(), _mm256_mul_ps(alphaBeta[i], inverseDeterminant));
		floatx8 alphaSqInv = _mm256_mul_ps(alphaSq[i], inverseDeterminant);
		floatx8 betaSqInv = _mm256_mul_ps(betaSq[i], inverseDeterminant);

		endpoints[i * 2 + 0] = blend(fmadd(betaSqInv, alphaX[i], mul(alphaBetaInv, betaX[i])), endpoints[i * 2 + 0], endpointsEqual[i]);
		endpoints[i * 2 + 1] = blend(fmadd(alphaBetaInv, alphaX[i], mul(alphaSqInv, betaX[i])), endpoints[i * 2 + 1], endpointsEqual[i]);
	}
}

template<uint32_t indexBits>
void least_squares_optimize_endpoints_rgba(vec4f pixels[16], vec4f endpoints[2], const BC7PartitionTable& table, uint32_t partition) {
	if (endpoints[0] == endpoints[1]) {
		//No starting axis to optimize along, the endpoint will be the exact color
		return;
	}
	float alphaSq = 0.0F;
	float alphaBeta = 0.0F;
	float betaSq = 0.0F;
	vec4f alphaX{ 0.0F };
	vec4f betaX{ 0.0F };
	//Find least squares best fit for indices
	for (uint32_t i = 0; i < 16; i++) {
		if (table.partitionNumbers[i] != partition) {
			continue;
		}
		float alpha = quantize_to_index<indexBits>(endpoints, pixels[i]);
		float beta = 1.0F - alpha;

		alphaSq += alpha * alpha;
		alphaBeta += alpha * beta;
		betaSq += beta * beta;

		alphaX += alpha * pixels[i];
		betaX += beta * pixels[i];
	}

	//Inverse matrix
	float inverseDeterminant = 1.0F / (alphaSq * betaSq - alphaBeta * alphaBeta);
	alphaBeta = -alphaBeta * inverseDeterminant;
	alphaSq *= inverseDeterminant;
	betaSq *= inverseDeterminant;

	endpoints[0] = betaSq * alphaX + alphaBeta * betaX;
	endpoints[1] = alphaBeta * alphaX + alphaSq * betaX;
}

template<uint32_t indexBits>
void least_squares_optimize_endpoints_rgb(vec4f pixels[16], vec3f endpoints[2], const BC7PartitionTable& table, uint32_t partition) {
	if (endpoints[0] == endpoints[1]) {
		//No starting axis to optimize along, the endpoint will be the exact color
		return;
	}
	float alphaSq = 0.0F;
	float alphaBeta = 0.0F;
	float betaSq = 0.0F;
	vec3f alphaX{ 0.0F, 0.0F, 0.0F };
	vec3f betaX{ 0.0F, 0.0F, 0.0F };
	//Find least squares best fit for indices
	for (uint32_t i = 0; i < 16; i++) {
		if (table.partitionNumbers[i] != partition) {
			continue;
		}
		float alpha = quantize_to_index<indexBits>(endpoints, pixels[i].xyz());
		float beta = 1.0F - alpha;

		alphaSq += alpha * alpha;
		alphaBeta += alpha * beta;
		betaSq += beta * beta;

		alphaX += alpha * pixels[i].xyz();
		betaX += beta * pixels[i].xyz();
	}

	//Inverse matrix
	float inverseDeterminant = 1.0F / (alphaSq * betaSq - alphaBeta * alphaBeta);
	alphaBeta = -alphaBeta * inverseDeterminant;
	alphaSq *= inverseDeterminant;
	betaSq *= inverseDeterminant;

	endpoints[0] = betaSq * alphaX + alphaBeta * betaX;
	endpoints[1] = alphaBeta * alphaX + alphaSq * betaX;
}

template<uint32_t indexBits>
void least_squares_optmize_endpoints_alpha(vec4f pixels[16], float endpoints[2]) {
	if (endpoints[0] == endpoints[1]) {
		//No starting axis to optimize along, the endpoint will be the exact color
		return;
	}
	float alphaSq = 0.0F;
	float alphaBeta = 0.0F;
	float betaSq = 0.0F;
	float alphaX{ 0.0F };
	float betaX{ 0.0F };
	//Find least squares best fit for indices
	for (uint32_t i = 0; i < 16; i++) {
		float alpha = quantize_to_index<indexBits>(endpoints, pixels[i].w);
		float beta = 1.0F - alpha;

		alphaSq += alpha * alpha;
		alphaBeta += alpha * beta;
		betaSq += beta * beta;

		alphaX += alpha * pixels[i].w;
		betaX += beta * pixels[i].w;
	}

	//Inverse matrix
	float inverseDeterminant = 1.0F / (alphaSq * betaSq - alphaBeta * alphaBeta);
	alphaBeta = -alphaBeta * inverseDeterminant;
	alphaSq *= inverseDeterminant;
	betaSq *= inverseDeterminant;

	endpoints[0] = betaSq * alphaX + alphaBeta * betaX;
	endpoints[1] = alphaBeta * alphaX + alphaSq * betaX;
}

const uint32_t numPartitionTablesPerSubset = 64;
const uint32_t numPartitionsFor3Subsets = numPartitionTablesPerSubset * 3;
const uint32_t numPartitionsFor2Subsets = numPartitionTablesPerSubset * 2;
const uint32_t totalNumPartitions = numPartitionsFor3Subsets + numPartitionsFor2Subsets;

float distance_to_line_sq(vec3f point, vec3f line[2]) {
	//float length = length(cross(point - line[0], point - line[1])) / length(line[1] - line[0]);
	//return length * length;
	vec3f lineVector = line[1] - line[0];
	float proj = dot(point - line[0], lineVector) / length_sq(lineVector);
	return length_sq(point - (line[0] + proj * lineVector));
}

float pixels_dist_to_line_sq(vec4f pixels[16], vec4f line[2]) {
	float dist = 0;
	vec4f lineVector = line[1] - line[0];
	for (uint32_t i = 0; i < 16; i++) {
		vec4f point = pixels[i];
		float proj = dot(point - line[0], lineVector) / length_sq(lineVector);
		return length_sq(point - (line[0] + proj * lineVector));
	}
	return dist;
}

template<uint32_t numPartitions>
void choose_best_diagonals_rgba(vec4f pixels[16], vec4f boundingBoxes[numPartitions * 2], const BC7PartitionTable& table) {
	for (uint32_t part = 0; part < numPartitions; part++) {
		vec4f& min = boundingBoxes[part * 2 + 0];
		vec4f& max = boundingBoxes[part * 2 + 1];

		vec4f bestDiag[2]{ min, max };
		float bestError = pixels_dist_to_line_sq(pixels, bestDiag);

		vec4f diag[2]{ min, max };
		float error;

#define CHECK_DIAG error = pixels_dist_to_line_sq(pixels, diag);\
		if (error < bestError) {\
			bestError = error;\
			memcpy(bestDiag, diag, 2 * sizeof(vec4f));\
		}

		std::swap(diag[0].x, diag[1].x);
		CHECK_DIAG;

		std::swap(diag[0].y, diag[1].y);
		CHECK_DIAG;

		std::swap(diag[0].x, diag[1].x);
		CHECK_DIAG;

		std::swap(diag[0].z, diag[1].z);
		CHECK_DIAG;

		std::swap(diag[0].x, diag[1].x);
		CHECK_DIAG;

		std::swap(diag[0].y, diag[1].y);
		CHECK_DIAG;

		std::swap(diag[0].x, diag[1].x);
		CHECK_DIAG;

#undef CHECK_DIAG

		memcpy(&boundingBoxes[part * 2], bestDiag, 2 * sizeof(vec4f));
	}
}

template<uint32_t numPartitions>
void choose_best_diagonals(vec4f pixels[16], vec3f boundingBoxes[6], const BC7PartitionTable& table) {
	const uint32_t diagonalSelectors[3 * 3] = { 0, 0, 0, 1, 0, 0, 0, 1, 0 };
	//3 endpoint pairs, 3 diagonals for each one
	float distancesSq[numPartitions][3]{};
	for (uint32_t i = 0; i < 16; i++) {
		uint32_t partition = table.partitionNumbers[i];
		vec3f* endpoints = &boundingBoxes[partition * 2];
		// I JUST LOOKED AT THIS AGAIN AND IT TURNS OUT THERE ARE 4 DIAGONALS ACCROSS A 3D BOX
		// I'm stupid
		for (uint32_t diag = 0; diag < 3; diag++) {
			vec3f diagonal[]{
				vec3f{ endpoints[diagonalSelectors[diag * 3 + 0]].x, endpoints[diagonalSelectors[diag * 3 + 1]].y, endpoints[diagonalSelectors[diag * 3 + 2]].z },
				vec3f{ endpoints[1 - diagonalSelectors[diag * 3 + 0]].x, endpoints[1 - diagonalSelectors[diag * 3 + 1]].y, endpoints[1 - diagonalSelectors[diag * 3 + 2]].z }
			};
			distancesSq[partition][diag] += distance_to_line_sq(pixels[i].xyz(), diagonal);
		}
	}
	for (uint32_t partition = 0; partition < numPartitions; partition++) {
		if (boundingBoxes[partition * 2] == boundingBoxes[partition * 2 + 1]) {
			continue;
		}
		uint32_t minDistanceDiag;
		float minDistance = FLT_MAX;
		for (uint32_t diag = 0; diag < 3; diag++) {
			float dist = distancesSq[partition][diag];
			if (dist < minDistance) {
				minDistanceDiag = diag;
				minDistance = dist;
			}
		}
		vec3f* endpoints = &boundingBoxes[partition * 2];
		vec3f diagonal[]{
				vec3f{ endpoints[diagonalSelectors[minDistanceDiag * 3 + 0]].x, endpoints[diagonalSelectors[minDistanceDiag * 3 + 1]].y, endpoints[diagonalSelectors[minDistanceDiag * 3 + 2]].z },
				vec3f{ endpoints[1 - diagonalSelectors[minDistanceDiag * 3 + 0]].x, endpoints[1 - diagonalSelectors[minDistanceDiag * 3 + 1]].y, endpoints[1 - diagonalSelectors[minDistanceDiag * 3 + 2]].z }
		};
		boundingBoxes[partition * 2 + 0] = diagonal[0];
		boundingBoxes[partition * 2 + 1] = diagonal[1];
	}
}

template<typename Endpoint>
FINLINE floatx8 pixels_dist_to_line_sq(const vec4fx8 pixels[16], const Endpoint line[2], const uint32_t partition, const BC7PartitionTable& table) {
	floatx8 distSq = _mm256_setzero_ps();
	for (uint32_t i = 0; i < 16; i++) {
		if (table.partitionNumbers[i] != partition) {
			continue;
		}
		Endpoint lineVector = line[1] - line[0];
		Endpoint point;
		if constexpr (std::is_same<Endpoint, vec3fx8>::value) {
			point = pixels[i].xyz();
		} else {
			point = pixels[i];
		}
		floatx8 proj = _mm256_div_ps(dot(point - line[0], lineVector), length_sq(lineVector));
		distSq = _mm256_add_ps(length_sq(point - (line[0] + proj * lineVector)), distSq);
	}
	return distSq;
}

template<uint32_t numPartitions>
FINLINE void choose_best_diagonals_rgba(const vec4fx8 pixels[16], vec4fx8 boundingBoxes[numPartitions], const BC7PartitionTable& table) {
	for (uint32_t part = 0; part < numPartitions; part++) {
		vec4fx8& min = boundingBoxes[part * 2 + 0];
		vec4fx8& max = boundingBoxes[part * 2 + 1];
		floatx8 blendMaskX = _mm256_setzero_ps();
		floatx8 blendMaskY = _mm256_setzero_ps();
		floatx8 blendMaskZ = _mm256_setzero_ps();

		//Check all 4 3d box diagonalconfigurations

		// (minx, miny, minz, minw), (maxx, maxy, maxz, maxw)
		vec4fx8 diagLine[2]{ min, max };
		floatx8 bestDistSq = pixels_dist_to_line_sq(pixels, diagLine, part, table);

		floatx8 all = _mm256_cmp_ps(_mm256_setzero_ps(), _mm256_setzero_ps(), _CMP_EQ_UQ);
		floatx8 none = _mm256_setzero_ps();

		floatx8 distSq;
		floatx8 less;
#define CHECK_DIAG(x, y, z) distSq = pixels_dist_to_line_sq(pixels, diagLine, part, table);\
		less = _mm256_cmp_ps(distSq, bestDistSq, _CMP_LT_OQ);\
		blendMaskX = _mm256_blendv_ps(blendMaskX, x, less);\
		blendMaskY = _mm256_blendv_ps(blendMaskY, y, less);\
		blendMaskZ = _mm256_blendv_ps(blendMaskZ, z, less);\
		bestDistSq = _mm256_min_ps(bestDistSq, distSq);

		// (maxx, miny, minz, minw), (minx, maxy, maxz, maxw)
		std::swap(diagLine[0].x, diagLine[1].x);
		CHECK_DIAG(all, none, none);

		// (maxx, maxy, minz, minw), (minx, miny, maxz, maxw)
		std::swap(diagLine[0].y, diagLine[1].y);
		CHECK_DIAG(all, all, none);

		// (minx, maxy, minz, minw), (maxx, miny, maxz, maxw)
		std::swap(diagLine[0].x, diagLine[1].x);
		CHECK_DIAG(none, all, none);

		// (minx, maxy, maxz, minw), (maxx, miny, minz, maxw)
		std::swap(diagLine[0].z, diagLine[1].z);
		CHECK_DIAG(none, all, all);

		// (maxx, maxy, maxz, minw), (minx, miny, minz, maxw)
		std::swap(diagLine[0].x, diagLine[1].x);
		CHECK_DIAG(all, all, all);

		// (maxx, miny, maxz, minw), (minx, maxy, minz, maxw)
		std::swap(diagLine[0].y, diagLine[1].y);
		CHECK_DIAG(all, none, all);

		// (minx, miny, maxz, minw), (maxx, maxy, minz, maxw)
		std::swap(diagLine[0].x, diagLine[1].x);
		CHECK_DIAG(none, none, all);

		// Blend together final values, write to out.
		// Perhaps I could combine this with the above code and output values directly instead of a mask
		floatx8 x0 = _mm256_blendv_ps(min.x, max.x, blendMaskX);
		floatx8 y0 = _mm256_blendv_ps(min.y, max.y, blendMaskY);
		floatx8 z0 = _mm256_blendv_ps(min.z, max.z, blendMaskZ);
		floatx8 x1 = _mm256_blendv_ps(max.x, min.x, blendMaskX);
		floatx8 y1 = _mm256_blendv_ps(max.y, min.y, blendMaskY);
		floatx8 z1 = _mm256_blendv_ps(max.z, min.z, blendMaskZ);
		boundingBoxes[part * 2 + 0] = vec4fx8{ x0, y0, z0, min.w };
		boundingBoxes[part * 2 + 1] = vec4fx8{ x1, y1, z1, max.w };
	}
}

template<uint32_t numPartitions>
FINLINE void choose_best_diagonals(const vec4fx8 pixels[16], vec3fx8 boundingBoxes[numPartitions], const BC7PartitionTable& table) {
	for (uint32_t part = 0; part < numPartitions; part++) {
		vec3fx8& min = boundingBoxes[part * 2 + 0];
		vec3fx8& max = boundingBoxes[part * 2 + 1];
		floatx8 blendMaskX = _mm256_setzero_ps();
		floatx8 blendMaskY = _mm256_setzero_ps();

		//Check all 4 3d box diagonalconfigurations

		// (minx, miny, minz), (maxx, maxy, maxz)
		vec3fx8 diagLine[2]{ min, max };
		floatx8 bestDistSq = pixels_dist_to_line_sq(pixels, diagLine, part, table);

		floatx8 all = _mm256_cmp_ps(_mm256_setzero_ps(), _mm256_setzero_ps(), _CMP_EQ_UQ);
		floatx8 none = _mm256_setzero_ps();

		floatx8 distSq;
		floatx8 less;
#define CHECK_DIAG(x, y) distSq = pixels_dist_to_line_sq(pixels, diagLine, part, table);\
		less = _mm256_cmp_ps(distSq, bestDistSq, _CMP_LT_OQ);\
		blendMaskX = _mm256_blendv_ps(blendMaskX, x, less);\
		blendMaskY = _mm256_blendv_ps(blendMaskY, y, less);\
		bestDistSq = _mm256_min_ps(bestDistSq, distSq);

		// (maxx, miny, minz), (minx, maxy, maxz)
		std::swap(diagLine[0].x, diagLine[1].x);
		CHECK_DIAG(all, none);

		// (maxx, maxy, minz), (minx, miny, maxz)
		std::swap(diagLine[0].y, diagLine[1].y);
		CHECK_DIAG(all, all);

		// (minx, maxy, minz), (maxx, miny, maxz)
		std::swap(diagLine[0].x, diagLine[1].x);
		CHECK_DIAG(none, all);

		// Blend together final values, write to out.
		// Perhaps I could combine this with the above code and output values directly instead of a mask
		floatx8 x0 = _mm256_blendv_ps(min.x, max.x, blendMaskX);
		floatx8 y0 = _mm256_blendv_ps(min.y, max.y, blendMaskY);
		floatx8 x1 = _mm256_blendv_ps(max.x, min.x, blendMaskX);
		floatx8 y1 = _mm256_blendv_ps(max.y, min.y, blendMaskY);
		boundingBoxes[part * 2 + 0] = vec3fx8{ x0, y0, min.z };
		boundingBoxes[part * 2 + 1] = vec3fx8{ x1, y1, max.z };
	}
}

float quantize_bc7_endpoints3_mode0(vec4f pixels[16], vec3f endpoints[6], const BC7PartitionTable& table, uint64_t* outIndices) {
	//Do most of the quantization work beforehand so we don't repeat it 4 times
	vec3f preQuantizedEndpoints[6];
	for (uint32_t end = 0; end < 6; end++) {
		preQuantizedEndpoints[end] = truncf(clamp01(endpoints[end]) * 15.0F + 0.5F);
	}
	for (uint32_t end = 0; end < 6; end++) {
		//Ray cast against the quantized space to extend the endpoints, that way we benefit more from the precision of the interpolation
		vec3f* localEndpoints = &endpoints[end & 0b110];
		uint32_t endpointIndex = end & 1;
		vec3f endpointDirection;
		if (localEndpoints[0] == localEndpoints[1]) {
			endpointDirection = localEndpoints[0] - preQuantizedEndpoints[end] / 15.0F;
			if (end) {
				endpointDirection = -endpointDirection;
			}
		} else {
			endpointDirection = localEndpoints[endpointIndex] - localEndpoints[1 - endpointIndex];
		}
		endpointDirection = normalize(endpointDirection);

		vec3f endpointScaled = (localEndpoints[endpointIndex] + 0.5F / 255.0F) * 15.0F;
		float t = ray_cast_unit_box(endpointScaled - truncf(endpointScaled), endpointDirection);
		vec3f extendedEndpoint = localEndpoints[endpointIndex];
		if (endpointDirection != vec3f{ 0.0F, 0.0F, 0.0F }) {
			extendedEndpoint += endpointDirection * t / 15.0F;
		}

		//4 bits
		vec3f quantized = truncf(clamp01(extendedEndpoint) * 15.0F + 0.5F);
		//Put bottom 3 bits in
		preQuantizedEndpoints[end] = quantized * 16.0F + truncf(quantized / 2.0F);
	}
	//Find best p bits to compress this endpoint
	float bestError[3] = { FLT_MAX, FLT_MAX, FLT_MAX };
	vec3f bestQuantizedEndpoints[6];
	//uint32_t bestPBits[3];
	uint64_t bestIndices[3];
	//Try every combination of p bits and check the error each time. Not fast, but it should provide much better results for gradients where the endpoints would normally quantize to the same value
	for (uint32_t pBits = 0; pBits < 4; pBits++) {
		//Quantize all endpoints with this set of p bits
		vec3f quantizedEndpoints[6];
		for (uint32_t end = 0; end < 6; end++) {
			float fPBit = static_cast<float>(((pBits >> (end & 1)) & 1)) * 8.0F;
			quantizedEndpoints[end] = preQuantizedEndpoints[end] + fPBit;
		}

		//Find indices for each pixel for the quantized endpoints, and find the error for each set of endpoints
		float error[3] = { 0.0F, 0.0F, 0.0F };
		uint64_t indices[3] = { 0, 0, 0 };
		for (uint32_t pixel = 0; pixel < 16; pixel++) {
			uint32_t partition = table.partitionNumbers[pixel];
			uint64_t index = static_cast<uint64_t>(find_index<3>(&quantizedEndpoints[partition * 2], pixels[pixel].xyz() * 255.0F));
			indices[partition] |= index << (pixel * 3);
			for (uint32_t component = 0; component < 3; component++) {
				float compressedComponent = static_cast<float>(bc7_interpolate(static_cast<uint8_t>(quantizedEndpoints[partition * 2 + 0].components[component]), static_cast<uint8_t>(quantizedEndpoints[partition * 2 + 1].components[component]), bc7InterpolationFactors3[index])) / 255.0F;
				float err = compressedComponent - pixels[pixel].components[component];
				error[partition] += err * err;
			}
		}
		for (uint32_t partition = 0; partition < 3; partition++) {
			if (error[partition] < bestError[partition]) {
				memcpy(&bestQuantizedEndpoints[partition * 2], &quantizedEndpoints[partition * 2], 2 * sizeof(vec3f));
				//bestPBits[partition] = pBits;
				bestError[partition] = error[partition];
				bestIndices[partition] = indices[partition];
			}
		}
	}
	//Write all the indices found so we don't have to recalculate them, and output the best quantized endpoints we found
	*outIndices = bestIndices[0] | bestIndices[1] | bestIndices[2];
	memcpy(endpoints, bestQuantizedEndpoints, 6 * sizeof(vec3f));
	return bestError[0] + bestError[1] + bestError[2];
}

// What did the 3 mean??? I can't remember and I didn't think to comment it
float quantize_bc7_endpoints3_mode1(vec4f pixels[16], vec3f endpoints[4], const BC7PartitionTable& table, uint64_t* outIndices) {
	//Do most of the quantization work beforehand so we don't repeat it 4 times
	vec3f preQuantizedEndpoints[4];
	for (uint32_t end = 0; end < 4; end++) {
		preQuantizedEndpoints[end] = truncf(clamp01(endpoints[end]) * 63.0F + 0.5F);
	}
	for (uint32_t end = 0; end < 4; end++) {
		//Ray cast against the quantized space to extend the endpoints, that way we benefit more from the precision of the interpolation
		vec3f* localEndpoints = &endpoints[end & 0b10];
		uint32_t endpointIndex = end & 1;
		vec3f endpointDirection;
		if (localEndpoints[0] == localEndpoints[1]) {
			endpointDirection = localEndpoints[0] - preQuantizedEndpoints[end] / 63.0F;
			if (end) {
				endpointDirection = -endpointDirection;
			}
		} else {
			endpointDirection = localEndpoints[endpointIndex] - localEndpoints[1 - endpointIndex];
		}
		vec3f normalizedEndpointDirection = normalize(endpointDirection);

		vec3f endpointScaled = (localEndpoints[endpointIndex] + 0.5F / 255.0F) * 63.0F;
		float t = ray_cast_unit_box(endpointScaled - truncf(endpointScaled), normalizedEndpointDirection);
		vec3f extendedEndpoint = localEndpoints[endpointIndex];
		if (endpointDirection != vec3f{ 0.0F, 0.0F, 0.0F }) {
			extendedEndpoint += normalizedEndpointDirection * t / 63.0F;
		}
		//6 bits
		vec3f quantized = truncf(clamp01(extendedEndpoint) * 63.0F + 0.5F);
		//Put bottom bit in
		preQuantizedEndpoints[end] = quantized * 4.0F + truncf(quantized / 32.0F);
	}
	//Find best p bits to compress this endpoint
	float bestError[2] = { FLT_MAX, FLT_MAX };
	vec3f bestQuantizedEndpoints[4];
	uint64_t bestIndices[2];
	//Try both p bits and check the error each time to get the best result. Could try optimizing this to find the p bit without brute forcing both
	for (uint32_t pBit = 0; pBit < 2; pBit++) {
		//Quantize all endpoints with this p bit
		vec3f quantizedEndpoints[4];
		float fPBit = static_cast<float>(pBit) * 2.0F;
		for (uint32_t end = 0; end < 4; end++) {
			quantizedEndpoints[end] = preQuantizedEndpoints[end] + fPBit;
		}

		//Find indices for each pixel for the quantized endpoints, and find the error for each set of endpoints
		float error[2] = { 0.0F, 0.0F };
		uint64_t indices[3] = { 0, 0, 0 };
		for (uint32_t pixel = 0; pixel < 16; pixel++) {
			uint32_t partition = table.partitionNumbers[pixel];
			uint64_t index = static_cast<uint64_t>(find_index<3>(&quantizedEndpoints[partition * 2], pixels[pixel].xyz() * 255.0F));
			indices[partition] |= index << (pixel * 3);
			for (uint32_t component = 0; component < 3; component++) {
				float compressedComponent = static_cast<float>(bc7_interpolate(static_cast<uint8_t>(quantizedEndpoints[partition * 2 + 0].components[component]), static_cast<uint8_t>(quantizedEndpoints[partition * 2 + 1].components[component]), bc7InterpolationFactors3[index])) / 255.0F;
				float err = compressedComponent - pixels[pixel].components[component];
				error[partition] += err * err;
			}
		}
		for (uint32_t partition = 0; partition < 2; partition++) {
			if (error[partition] < bestError[partition]) {
				memcpy(&bestQuantizedEndpoints[partition * 2], &quantizedEndpoints[partition * 2], 2 * sizeof(vec3f));
				bestError[partition] = error[partition];
				bestIndices[partition] = indices[partition];
			}
		}
	}
	//Write all the indices found so we don't have to recalculate them, and output the best quantized endpoints we found
	*outIndices = bestIndices[0] | bestIndices[1];
	memcpy(endpoints, bestQuantizedEndpoints, 4 * sizeof(vec3f));
	return bestError[0] + bestError[1];
}

template<uint32_t partitions, uint32_t componentBits, uint32_t numPBits, uint32_t indexResolution, typename Endpoint = vec3f, uint32_t alphaBits = 0>
float quantize_bc7_endpoints(vec4f pixels[16], Endpoint endpoints[partitions * 2], const BC7PartitionTable& table, uint64_t* outIndices) {

	constexpr uint32_t numEndpoints = partitions * 2;
	constexpr float quantizeScaleXYZ = static_cast<float>((1 << componentBits) - 1);
	constexpr float quantizeScaleAlpha = static_cast<float>((1 << alphaBits) - 1);
	constexpr Endpoint quantizeScale = [] { if constexpr (alphaBits > 0) { return Endpoint{ quantizeScaleXYZ, quantizeScaleXYZ, quantizeScaleXYZ, quantizeScaleAlpha }; } else { return Endpoint{ quantizeScaleXYZ }; } }();
	constexpr float pBitShiftXYZ = static_cast<float>(1 << (8 - componentBits - 1));
	constexpr float pBitShiftAlpha = static_cast<float>(1 << (8 - alphaBits - 1));
	constexpr Endpoint pBitShift = [] { if constexpr (alphaBits > 0) { return Endpoint{ pBitShiftXYZ, pBitShiftXYZ, pBitShiftXYZ, pBitShiftAlpha }; } else { return Endpoint{ pBitShiftXYZ }; } }();
	constexpr float dataShiftXYZ = static_cast<float>(1 << (8 - componentBits));
	constexpr float dataShiftAlpha = static_cast<float>(1 << (8 - alphaBits));
	constexpr Endpoint dataShift = [] { if constexpr (alphaBits > 0) { return Endpoint{ dataShiftXYZ, dataShiftXYZ, dataShiftXYZ, dataShiftAlpha }; } else { return Endpoint{ dataShiftXYZ }; } }();
	constexpr float bottomDataShiftXYZ = numPBits > 0 ? static_cast<float>(1 << (componentBits - (8 - componentBits - 1))) : static_cast<float>(1 << (componentBits - (8 - componentBits)));
	constexpr float bottomDataShiftAlpha = numPBits > 0 ? static_cast<float>(1 << (alphaBits - (8 - alphaBits - 1))) : static_cast<float>(1 << (alphaBits - (8 - alphaBits)));
	constexpr Endpoint bottomDataShift = [] { if constexpr (alphaBits > 0) { return Endpoint{ bottomDataShiftXYZ, bottomDataShiftXYZ, bottomDataShiftXYZ, bottomDataShiftAlpha }; } else { return Endpoint{ bottomDataShiftXYZ }; } }();

	//Do most of the quantization work beforehand so we don't repeat it 4 times
	Endpoint preQuantizedEndpoints[numEndpoints];
	for (uint32_t end = 0; end < numEndpoints; end++) {
		preQuantizedEndpoints[end] = truncf(clamp01(endpoints[end]) * quantizeScale + 0.5F);
	}
	for (uint32_t end = 0; end < numEndpoints; end++) {
		//Ray cast against the quantized space to extend the endpoints, that way we benefit more from the precision of the interpolation
		Endpoint* localEndpoints = &endpoints[end & (~1ui32)];
		uint32_t endpointIndex = end & 1;
		Endpoint endpointDirection;
		if (localEndpoints[0] == localEndpoints[1]) {
			endpointDirection = localEndpoints[0] - preQuantizedEndpoints[end] / quantizeScale;
			if (end & 1) {
				endpointDirection = -endpointDirection;
			}
		} else {
			endpointDirection = localEndpoints[endpointIndex] - localEndpoints[1 - endpointIndex];
		}
		Endpoint normalizedEndpointDirection = normalize(endpointDirection);

		Endpoint endpointScaled = (localEndpoints[endpointIndex] + 0.5F / 255.0F) * quantizeScale;
		float raycastIntersectTime = ray_cast_unit_box(endpointScaled - truncf(endpointScaled), normalizedEndpointDirection);
		Endpoint extendedEndpoint = localEndpoints[endpointIndex];
		if (endpointDirection != Endpoint{ 0.0F }) {
			extendedEndpoint += normalizedEndpointDirection * raycastIntersectTime / quantizeScale;
		}
		//6 bits
		Endpoint quantized = truncf(clamp01(extendedEndpoint) * quantizeScale + 0.5F);
		//Put bottom bits in
		preQuantizedEndpoints[end] = quantized * dataShift + truncf(quantized / bottomDataShift);
	}
	//Find best p bits to compress this endpoint
	float bestError[partitions];
	for (uint32_t i = 0; i < partitions; i++) {
		bestError[i] = FLT_MAX;
	}
	Endpoint bestQuantizedEndpoints[numEndpoints];
	uint64_t bestIndices[3]{};
	//Try both p bits and check the error each time to get the best result. Could try optimizing this to find the p bit without brute forcing both
	for (uint32_t pBit = 0; pBit < (1 << numPBits); pBit++) {
		//Quantize all endpoints with this p bit
		Endpoint quantizedEndpoints[numEndpoints];
		Endpoint fPBit0 = static_cast<float>(pBit & 0b01) * pBitShift;
		Endpoint fPBit1 = numPBits == 2 ? static_cast<float>((pBit & 0b10) >> 1) * pBitShift : fPBit0;
		Endpoint fPBits[2]{ fPBit0, fPBit1 };
		for (uint32_t end = 0; end < numEndpoints; end++) {
			quantizedEndpoints[end] = preQuantizedEndpoints[end] + fPBits[end & 1];
		}

		//Find indices for each pixel for the quantized endpoints, and find the error for each set of endpoints
		float error[partitions]{};
		uint64_t indices[partitions]{};
		for (uint32_t pixel = 0; pixel < 16; pixel++) {
			Endpoint pixelEndpoint;
			if constexpr (std::is_same<Endpoint, float>::value) {
				pixelEndpoint = pixels[pixel].w;
			} else if constexpr (std::is_same<Endpoint, vec3f>::value) {
				pixelEndpoint = pixels[pixel].xyz();
			} else if constexpr (std::is_same<Endpoint, vec4f>::value) {
				pixelEndpoint = pixels[pixel];
			}
			uint32_t partition = table.partitionNumbers[pixel];
			uint64_t index = static_cast<uint64_t>(find_index<indexResolution>(&quantizedEndpoints[partition * 2], pixelEndpoint * 255.0F));;
			uint32_t interpolationFactor;
			static_assert(indexResolution == 2 || indexResolution == 3 || indexResolution == 4, "Index resolution wrong");
			if constexpr (indexResolution == 2) {
				interpolationFactor = bc7InterpolationFactors2[index];
			} else if constexpr (indexResolution == 3) {
				interpolationFactor = bc7InterpolationFactors3[index];
			} else if constexpr (indexResolution == 4) {
				interpolationFactor = bc7InterpolationFactors4[index];
			}
			indices[partition] |= index << (pixel * indexResolution);
			Endpoint compressed = bc7_interpolate(quantizedEndpoints[partition * 2 + 0], quantizedEndpoints[partition * 2 + 1], interpolationFactor, 0) / 255.0F;
			error[partition] += length_sq(compressed - pixelEndpoint);
		}
		for (uint32_t partition = 0; partition < partitions; partition++) {
			if (error[partition] < bestError[partition]) {
				memcpy(&bestQuantizedEndpoints[partition * 2], &quantizedEndpoints[partition * 2], 2 * sizeof(Endpoint));
				bestError[partition] = error[partition];
				bestIndices[partition] = indices[partition];
			}
		}
	}
	//Write all the indices found so we don't have to recalculate them, and output the best quantized endpoints we found
	*outIndices = bestIndices[0] | bestIndices[1] | bestIndices[2];
	memcpy(endpoints, bestQuantizedEndpoints, numEndpoints * sizeof(Endpoint));
	float finalError = 0;
	for (uint32_t i = 0; i < partitions; i++) {
		finalError += bestError[i];
	}
	return finalError;
}

template<uint32_t partitions, uint32_t componentBits, uint32_t numPBits, uint32_t indexResolution, typename Endpoint = vec3fx8, uint32_t alphaBits = 0>
floatx8 quantize_bc7_endpointsx8(vec4fx8 pixels[16], Endpoint endpoints[partitions * 2], const BC7PartitionTable& table, uint64x4 outIndices[2]) {
	// Bunch of constants to use in the function
	// Compiler should optimize all this stuff away
	constexpr uint32_t numEndpoints = partitions * 2;
	const floatx8 quantizeScaleXYZ = _mm256_set1_ps(static_cast<float>((1 << componentBits) - 1));
	const floatx8 quantizeScaleAlpha = _mm256_set1_ps(static_cast<float>((1 << alphaBits) - 1));
	Endpoint quantizeScale; if constexpr (alphaBits > 0) { quantizeScale = Endpoint{ quantizeScaleXYZ, quantizeScaleXYZ, quantizeScaleXYZ, quantizeScaleAlpha }; } else { quantizeScale = Endpoint{ quantizeScaleXYZ }; }
	Endpoint invQuantizeScale = rcp(quantizeScale);
	const floatx8 pBitShiftXYZ = _mm256_set1_ps(static_cast<float>(1 << (8 - componentBits - 1)));
	const floatx8 pBitShiftAlpha = _mm256_set1_ps(static_cast<float>(1 << (8 - alphaBits - 1)));
	Endpoint pBitShift; if constexpr (alphaBits > 0) { pBitShift = Endpoint{ pBitShiftXYZ, pBitShiftXYZ, pBitShiftXYZ, pBitShiftAlpha }; } else { pBitShift = Endpoint{ pBitShiftXYZ }; }
	const floatx8 dataShiftXYZ = _mm256_set1_ps(static_cast<float>(1 << (8 - componentBits)));
	const floatx8 dataShiftAlpha = _mm256_set1_ps(static_cast<float>(1 << (8 - alphaBits)));
	Endpoint dataShift; if constexpr (alphaBits > 0) { dataShift = Endpoint{ dataShiftXYZ, dataShiftXYZ, dataShiftXYZ, dataShiftAlpha }; } else { dataShift = Endpoint{ dataShiftXYZ }; }
	const floatx8 bottomDataShiftXYZ = _mm256_set1_ps(numPBits > 0 ? static_cast<float>(1 << (componentBits - (8 - componentBits - 1))) : static_cast<float>(1 << (componentBits - (8 - componentBits))));
	const floatx8 bottomDataShiftAlpha = _mm256_set1_ps(numPBits > 0 ? static_cast<float>(1 << (alphaBits - (8 - alphaBits - 1))) : static_cast<float>(1 << (alphaBits - (8 - alphaBits))));
	Endpoint bottomDataShift; if constexpr (alphaBits > 0) { bottomDataShift = Endpoint{ bottomDataShiftXYZ, bottomDataShiftXYZ, bottomDataShiftXYZ, bottomDataShiftAlpha }; } else { bottomDataShift = Endpoint{ bottomDataShiftXYZ }; }
	Endpoint invBottomDataShift = rcp(bottomDataShift);


	// Do most of the quantization work beforehand so we don't repeat it 4 times
	Endpoint preQuantizedEndpoints[numEndpoints];
	for (uint32_t end = 0; end < numEndpoints; end++) {
		preQuantizedEndpoints[end] = trunc(fmadd(clamp01(endpoints[end]), quantizeScale, _mm256_set1_ps(0.5F)));
	}
	for (uint32_t end = 0; end < numEndpoints; end++) {
		// Ray cast against the quantized space to extend the endpoints, that way we benefit more from the precision of the interpolation
		Endpoint* localEndpoints = &endpoints[end & (~1ui32)];
		uint32_t endpointIndex = end & 1;

		floatx8 endpointsSame = localEndpoints[0] == localEndpoints[1];
		Endpoint endpointDirectionWhenSame = sub(localEndpoints[0], mul(preQuantizedEndpoints[end], invQuantizeScale));
		if (end & 1) {
			endpointDirectionWhenSame = sub(_mm256_setzero_ps(), endpointDirectionWhenSame);
		}
		Endpoint endpointDirectionWhenDifferent = sub(localEndpoints[endpointIndex], localEndpoints[1 - endpointIndex]);
		Endpoint endpointDirection = blend(endpointDirectionWhenDifferent, endpointDirectionWhenSame, endpointsSame);
		Endpoint endpointDirectionNormalized = normalize(endpointDirection);

		Endpoint endpointScaled = mul(add(localEndpoints[endpointIndex], _mm256_set1_ps(0.5F / 255.0F)), quantizeScale);
		// This raycast *will* break with fast math turned on. Don't turn on fast math!
		floatx8 raycastIntersectTime = ray_cast_unit_box(sub(endpointScaled, trunc(endpointScaled)), endpointDirectionNormalized);
		Endpoint regularEndpoint = localEndpoints[endpointIndex];

		floatx8 endpointDirectionIsZero = equals_zero(endpointDirection);
		Endpoint endpointExtension = fmadd(mul(endpointDirectionNormalized, invQuantizeScale), raycastIntersectTime, regularEndpoint);

		Endpoint extendedEndpoint = blend(endpointExtension, regularEndpoint, endpointDirectionIsZero);
		Endpoint quantized = trunc(fmadd(clamp01(extendedEndpoint), quantizeScale, _mm256_set1_ps(0.5F)));

		// Put bottom bits in
		preQuantizedEndpoints[end] = fmadd(quantized, dataShift, trunc(mul(quantized, invBottomDataShift)));
	}

	//Find best p bits to compress this endpoint
	floatx8 bestError[partitions];
	for (uint32_t i = 0; i < partitions; i++) {
		bestError[i] = _mm256_set1_ps(FLT_MAX);
	}
	Endpoint bestQuantizedEndpoints[numEndpoints];
	uint64x4 bestIndices[3][2]{};
	//Try both p bits and check the error each time to get the best result. Could try optimizing this to find the p bit without brute forcing both
	for (uint32_t pBit = 0; pBit < (1 << numPBits); pBit++) {
		//Quantize all endpoints with this p bit
		Endpoint quantizedEndpoints[numEndpoints];
		Endpoint fPBit0 = mul(_mm256_set1_ps(static_cast<float>(pBit & 0b01)), pBitShift);
		Endpoint fPBit1 = numPBits == 2 ? mul(_mm256_set1_ps(static_cast<float>((pBit & 0b10) >> 1)), pBitShift) : fPBit0;
		Endpoint fPBits[2]{ fPBit0, fPBit1 };
		for (uint32_t end = 0; end < numEndpoints; end++) {
			quantizedEndpoints[end] = add(preQuantizedEndpoints[end], fPBits[end & 1]);
		}

		//Find indices for each pixel for the quantized endpoints, and find the error for each set of endpoints
		floatx8 error[partitions];
		uint64x4 indices[partitions][2];
		for (uint32_t i = 0; i < partitions; i++) {
			error[i] = _mm256_setzero_ps();
			indices[i][0] = _mm256_setzero_si256();
			indices[i][1] = _mm256_setzero_si256();
		}
		for (uint32_t pixel = 0; pixel < 16; pixel++) {
			uint32_t partition = table.partitionNumbers[pixel];
			Endpoint pixelEndpoint;
			if constexpr (std::is_same<vec4fx8, Endpoint>::value) {
				pixelEndpoint = pixels[pixel];
			} else if constexpr (std::is_same<vec3fx8, Endpoint>::value) {
				pixelEndpoint = pixels[pixel].xyz();
			} else {
				pixelEndpoint = pixels[pixel].w;
			}
			uint32x8 index = find_index<indexResolution>(&quantizedEndpoints[partition * 2], mul(pixelEndpoint, _mm256_set1_ps(255.0F)));
			uint32x8 interpolationFactor;
			static_assert(indexResolution == 2 || indexResolution == 3 || indexResolution == 4, "Index resolution wrong");
			if constexpr (indexResolution == 2) {
				// (index * 86) >> 2, yields the bc7InterpolationFactors2 table
				// I just used linear regression to find some linear equations I could turn into integer math
				// Doing this in math is probably better than as a gather
				interpolationFactor = _mm256_srli_epi32(_mm256_mullo_epi32(index, _mm256_set1_epi32(86)), 2);
			} else if constexpr (indexResolution == 3) {
				// (index * 37) >> 2, same as above but for bc7InterpolationFactors3
				interpolationFactor = _mm256_srli_epi32(_mm256_mullo_epi32(index, _mm256_set1_epi32(37)), 2);
			} else if constexpr (indexResolution == 4) {
				// (index * 68 + 8) >> 4, same as above but for bc7InterpolationFactors4
				interpolationFactor = _mm256_srli_epi32(_mm256_add_epi32(_mm256_mullo_epi32(index, _mm256_set1_epi32(68)), _mm256_set1_epi32(8)), 4);
			}
			uint64x4 indicesLow = _mm256_slli_epi64(_mm256_cvtepi32_epi64(_mm256_extracti128_si256(index, 0)), pixel * indexResolution);
			uint64x4 indicesHigh = _mm256_slli_epi64(_mm256_cvtepi32_epi64(_mm256_extracti128_si256(index, 1)), pixel * indexResolution);
			indices[partition][0] = _mm256_or_si256(indices[partition][0], indicesLow);
			indices[partition][1] = _mm256_or_si256(indices[partition][1], indicesHigh);

			Endpoint compressed = mul(bc7_interpolatex8(quantizedEndpoints[partition * 2 + 0], quantizedEndpoints[partition * 2 + 1], interpolationFactor, _mm256_setzero_si256()), _mm256_set1_ps(1.0F / 255.0F));
			error[partition] = _mm256_add_ps(error[partition], length_sq(sub(compressed, pixelEndpoint)));
		}
		for (uint32_t partition = 0; partition < partitions; partition++) {
			floatx8 errorLessThan = _mm256_cmp_ps(error[partition], bestError[partition], _CMP_LT_OQ);
			// Set current best error
			bestError[partition] = _mm256_min_ps(bestError[partition], error[partition]);
			// Copy any better endpoints to the output
			bestQuantizedEndpoints[partition * 2 + 0] = blend(bestQuantizedEndpoints[partition * 2 + 0], quantizedEndpoints[partition * 2 + 0], errorLessThan);
			bestQuantizedEndpoints[partition * 2 + 1] = blend(bestQuantizedEndpoints[partition * 2 + 1], quantizedEndpoints[partition * 2 + 1], errorLessThan);
			// Copy any better indices to the output
			uint32x8 errorLessThanimask = _mm256_castps_si256(errorLessThan);
			uint32x8 indicesLowMask = _mm256_permutevar8x32_epi32(errorLessThanimask, _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3));
			uint32x8 indicesHighMask = _mm256_permutevar8x32_epi32(errorLessThanimask, _mm256_setr_epi32(4, 4, 5, 5, 6, 6, 7, 7));
			bestIndices[partition][0] = _mm256_blendv_epi8(bestIndices[partition][0], indices[partition][0], indicesLowMask);
			bestIndices[partition][1] = _mm256_blendv_epi8(bestIndices[partition][1], indices[partition][1], indicesHighMask);
		}
	}
	//Write all the indices found so we don't have to recalculate them, and output the best quantized endpoints we found
	outIndices[0] = _mm256_or_si256(bestIndices[0][0], _mm256_or_si256(bestIndices[1][0], bestIndices[2][0]));
	outIndices[1] = _mm256_or_si256(bestIndices[0][1], _mm256_or_si256(bestIndices[1][1], bestIndices[2][1]));
	memcpy(endpoints, bestQuantizedEndpoints, numEndpoints * sizeof(Endpoint));

	floatx8 finalError = _mm256_setzero_ps();
	for (uint32_t i = 0; i < partitions; i++) {
		finalError = _mm256_add_ps(finalError, bestError[i]);
	}
	return finalError;
}

void write_bc7_block_mode0(BC7Block& block, uint32_t bestPartition, vec3f bestEndpoints[6], uint64_t bestIndices) {
	//mode 0
	block.data[0] = 0b1;
	block.data[1] = 0;
	block.data[0] |= bestPartition << 1;
	uint64_t endpointReds = 0;
	uint64_t endpointGreens = 0;
	uint64_t endpointBlues = 0;
	uint64_t endpointPBits = 0;
	for (uint32_t endpoint = 0; endpoint < 6; endpoint++) {
		uint32_t r = static_cast<uint32_t>(bestEndpoints[endpoint].x);
		uint32_t g = static_cast<uint32_t>(bestEndpoints[endpoint].y);
		uint32_t b = static_cast<uint32_t>(bestEndpoints[endpoint].z);
		endpointReds |= ((r >> 4) & 0b1111) << (endpoint * 4);
		endpointGreens |= ((g >> 4) & 0b1111) << (endpoint * 4);
		endpointBlues |= ((b >> 4) & 0b1111) << (endpoint * 4);
		endpointPBits |= ((r >> 3) & 1) << endpoint;
	}
	block.data[0] |= endpointReds << 5;
	block.data[0] |= endpointGreens << 29;
	block.data[0] |= endpointBlues << 53;
	block.data[1] |= endpointBlues >> 11;
	block.data[1] |= endpointPBits << 13;

	uint32_t anchor2nd = bc7PartitionTable3Anchors2ndSubset[bestPartition];
	uint32_t anchor3rd = bc7PartitionTable3Anchors3rdSubset[bestPartition];

	uint64_t encodedIndices = 0;
	uint32_t shift = 0;
	for (uint32_t i = 0; i < 16; i++) {
		//3 bit indices
		encodedIndices |= ((bestIndices >> (i * 3)) & 0b111) << shift;
		if ((i == 0) | (i == anchor2nd) | (i == anchor3rd)) {
			shift += 2;
		} else {
			shift += 3;
		}
	}

	block.data[1] |= encodedIndices << 19;
}

void decompress_bc7_mode1(BC7Block& block, RGBA pixels[16]);

void write_bc7_block_mode1(BC7Block& block, uint32_t bestPartition, vec3f bestEndpoints[6], uint64_t bestIndices) {
	//mode 1
	block.data[0] = 0b10;
	block.data[1] = 0;
	block.data[0] |= bestPartition << 2;
	uint64_t endpointReds = 0;
	uint64_t endpointGreens = 0;
	uint64_t endpointBlues = 0;
	uint64_t endpointPBits = 0;
	for (uint32_t endpoint = 0; endpoint < 4; endpoint++) {
		uint32_t r = static_cast<uint32_t>(bestEndpoints[endpoint].x);
		uint32_t g = static_cast<uint32_t>(bestEndpoints[endpoint].y);
		uint32_t b = static_cast<uint32_t>(bestEndpoints[endpoint].z);
		endpointReds |= ((r >> 2) & 0b111111) << (endpoint * 6);
		endpointGreens |= ((g >> 2) & 0b111111) << (endpoint * 6);
		endpointBlues |= ((b >> 2) & 0b111111) << (endpoint * 6);
		endpointPBits |= ((r >> 1) & 1) << (endpoint >> 1);
	}
	block.data[0] |= endpointReds << 8;
	block.data[0] |= endpointGreens << 32;
	block.data[0] |= endpointBlues << 56;
	block.data[1] |= endpointBlues >> 8;
	block.data[1] |= endpointPBits << 16;

	uint32_t anchor2nd = bc7PartitionTable2Anchors2ndSubset[bestPartition];

	uint64_t encodedIndices = 0;
	uint32_t shift = 0;
	for (uint32_t i = 0; i < 16; i++) {
		//3 bit indices
		encodedIndices |= ((bestIndices >> (i * 3)) & 0b111) << shift;
		if ((i == 0) | (i == anchor2nd)) {
			shift += 2;
		} else {
			shift += 3;
		}
	}

	block.data[1] |= encodedIndices << 18;
}

template<typename Endpoint, uint32_t partitions, uint32_t indexResolution>
void check_flip_indices(Endpoint endpoints[partitions * 2], uint64_t* outIndices, const uint32_t partitionTable, const BC7PartitionTable& table) {
	uint64_t indices = *outIndices;
	//For each subset, if the anchor most significant bit is 1, flip its partition around
	constexpr uint64_t indexMask = (1 << indexResolution) - 1;
	constexpr uint64_t highBitCheck = indexMask >> 1;

	bool shouldFlip[partitions];
	shouldFlip[0] = (indices & indexMask) > highBitCheck;
	if constexpr (partitions == 2) {
		shouldFlip[1] = ((indices >> (bc7PartitionTable2Anchors2ndSubset[partitionTable] * indexResolution)) & indexMask) > highBitCheck;
	} else if constexpr (partitions == 3) {
		shouldFlip[1] = ((indices >> (bc7PartitionTable3Anchors2ndSubset[partitionTable] * indexResolution)) & indexMask) > highBitCheck;
		shouldFlip[2] = ((indices >> (bc7PartitionTable3Anchors3rdSubset[partitionTable] * indexResolution)) & indexMask) > highBitCheck;
	}

	for (uint32_t i = 0; i < partitions; i++) {
		if (shouldFlip[i]) {
			std::swap(endpoints[i * 2], endpoints[i * 2 + 1]);
		}
	}
	for (uint32_t i = 0; i < 16; i++) {
		if (shouldFlip[table.partitionNumbers[i]]) {
			indices ^= indexMask << (i * indexResolution);
		}
	}
	*outIndices = indices;
}

template<typename Endpoint, uint32_t partitions, uint32_t indexResolution>
void check_flip_indices(Endpoint endpoints[partitions * 2], uint64x4 outIndices[2], const uint32x8 partitionTable, const uint32x8 tableData[16]) {
	uint64x4 indices[2]{ outIndices[0], outIndices[1] };
	//For each subset, if the anchor most significant bit is 1, flip its partition around
	constexpr uint64_t indexMask = (1 << indexResolution) - 1;
	const uint64x4 indexMask4 = _mm256_set1_epi64x(indexMask);
	constexpr uint32_t highBitCheck = indexMask >> 1;
	const uint32x4 highBitCheck4 = _mm_set1_epi32(highBitCheck);

	uint32x8 shouldFlip[partitions];
	// (indices & indexMask) > highBitCheck
	uint32x4 cmpA = _mm_cmpgt_epi32(cvt_int64x4_int32x4(_mm256_and_si256(indices[0], indexMask4)), highBitCheck4);
	uint32x4 cmpB = _mm_cmpgt_epi32(cvt_int64x4_int32x4(_mm256_and_si256(indices[1], indexMask4)), highBitCheck4);
	shouldFlip[0] = _mm256_inserti128_si256(_mm256_castsi128_si256(cmpA), cmpB, 1);

	if constexpr (partitions == 2) {
		//shouldFlip[1] = ((indices >> (bc7PartitionTable2Anchors2ndSubset[partitionTable] * indexResolution)) & indexMask) > highBitCheck;

		// bc7PartitionTable2Anchors2ndSubset[partitionTable] * indexResolution
		uint32x8 tableAnchors = _mm256_mullo_epi32(_mm256_i32gather_epi32(reinterpret_cast<const int*>(bc7PartitionTable2Anchors2ndSubset), partitionTable, 4), _mm256_set1_epi32(indexResolution));
		// Extract low and high halves, casting to 64 bit integers
		uint64x4 tableAnchorsLow = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(tableAnchors));
		uint64x4 tableAnchorsHigh = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(tableAnchors, 1));
		// (indices >> anchor) & indexMask > highBitCheck
		cmpA = _mm_cmpgt_epi32(cvt_int64x4_int32x4(_mm256_and_si256(_mm256_srlv_epi64(indices[0], tableAnchorsLow), indexMask4)), highBitCheck4);
		cmpB = _mm_cmpgt_epi32(cvt_int64x4_int32x4(_mm256_and_si256(_mm256_srlv_epi64(indices[1], tableAnchorsHigh), indexMask4)), highBitCheck4);
		// Combine low and high halves again, we're back in 32 bits
		shouldFlip[1] = _mm256_inserti128_si256(_mm256_castsi128_si256(cmpA), cmpB, 1);
	} else if constexpr (partitions == 3) {
		//shouldFlip[1] = ((indices >> (bc7PartitionTable3Anchors2ndSubset[partitionTable] * indexResolution)) & indexMask) > highBitCheck;
		//shouldFlip[2] = ((indices >> (bc7PartitionTable3Anchors3rdSubset[partitionTable] * indexResolution)) & indexMask) > highBitCheck;

		// bc7PartitionTable3Anchors2ndSubset[partitionTable] * indexResolution
		uint32x8 tableAnchors = _mm256_mullo_epi32(_mm256_i32gather_epi32(reinterpret_cast<const int*>(bc7PartitionTable3Anchors2ndSubset), partitionTable, 4), _mm256_set1_epi32(indexResolution));
		// Extract low and high halves, casting to 64 bit integers
		uint64x4 tableAnchorsLow = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(tableAnchors));
		uint64x4 tableAnchorsHigh = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(tableAnchors, 1));
		// (indices >> anchor) & indexMask > highBitCheck
		cmpA = _mm_cmpgt_epi32(cvt_int64x4_int32x4(_mm256_and_si256(_mm256_srlv_epi64(indices[0], tableAnchorsLow), indexMask4)), highBitCheck4);
		cmpB = _mm_cmpgt_epi32(cvt_int64x4_int32x4(_mm256_and_si256(_mm256_srlv_epi64(indices[1], tableAnchorsHigh), indexMask4)), highBitCheck4);
		// Combine low and high halves again, we're back in 32 bits
		shouldFlip[1] = _mm256_inserti128_si256(_mm256_castsi128_si256(cmpA), cmpB, 1);

		// bc7PartitionTable3Anchors3rdSubset[partitionTable] * indexResolution
		tableAnchors = _mm256_mullo_epi32(_mm256_i32gather_epi32(reinterpret_cast<const int*>(bc7PartitionTable3Anchors3rdSubset), partitionTable, 4), _mm256_set1_epi32(indexResolution));
		// Extract low and high halves, casting to 64 bit integers
		tableAnchorsLow = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(tableAnchors));
		tableAnchorsHigh = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(tableAnchors, 1));
		// (indices >> anchor) & indexMask > highBitCheck
		cmpA = _mm_cmpgt_epi32(cvt_int64x4_int32x4(_mm256_and_si256(_mm256_srlv_epi64(indices[0], tableAnchorsLow), indexMask4)), highBitCheck4);
		cmpB = _mm_cmpgt_epi32(cvt_int64x4_int32x4(_mm256_and_si256(_mm256_srlv_epi64(indices[1], tableAnchorsHigh), indexMask4)), highBitCheck4);
		// Combine low and high halves again, we're back in 32 bits
		shouldFlip[2] = _mm256_inserti128_si256(_mm256_castsi128_si256(cmpA), cmpB, 1);
	}

	for (uint32_t i = 0; i < partitions; i++) {
		Endpoint endA{ endpoints[i * 2 + 0] };
		Endpoint endB{ endpoints[i * 2 + 1] };
		// If should flip, swap the endpoints with a blend
		endpoints[i * 2 + 0] = blend(endA, endB, _mm256_castsi256_ps(shouldFlip[i]));
		endpoints[i * 2 + 1] = blend(endB, endA, _mm256_castsi256_ps(shouldFlip[i]));
	}

	for (uint32_t i = 0; i < 16; i++) {
		uint32x8 tableRowi = tableData[i];
		// shouldFlipMask = blend(blend(shouldFlip[2], shouldlFlip[1], tableRowi == 1), shouldFlip[0], tableRowi == 0)
		// AKA shouldFlipMask = tableRowi == 0 ? shouldFlip[0] : tableRowi == 1 ? shouldFlip[1] : shouldFlip[2]
		floatx8 equals0 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(tableRowi, _mm256_setzero_si256()));
		floatx8 equals1 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(tableRowi, _mm256_set1_epi32(1)));
		floatx8 equals2 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(tableRowi, _mm256_set1_epi32(2)));
		// equals0 ? shouldFlip[0] : 
		// equals1 ? shouldFlip[1] : 
		// equals2 ? shouldFlip[2] : 
		// zero
		uint32x8 shouldFlipMask = _mm256_castps_si256(
			_mm256_blendv_ps(
			_mm256_blendv_ps(
			_mm256_blendv_ps(
			_mm256_setzero_ps(),
			_mm256_castsi256_ps(shouldFlip[2]), equals2),
			_mm256_castsi256_ps(shouldFlip[1]), equals1),
			_mm256_castsi256_ps(shouldFlip[0]), equals0));

		// If shouldFlipMask or flipMask is 0, the index won't be flipped
		uint64x4 flipMask = _mm256_set1_epi64x(indexMask << (i * indexResolution));
		// indices[0] ^= flipMask & shouldFlipMask[0-3]
		indices[0] = _mm256_xor_si256(indices[0], _mm256_and_si256(flipMask, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(shouldFlipMask))));
		// indices[1] ^= flipMask & shouldFlipMask[4-7]
		indices[1] = _mm256_xor_si256(indices[1], _mm256_and_si256(flipMask, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(shouldFlipMask, 1))));
	}
	outIndices[0] = indices[0];
	outIndices[1] = indices[1];
}

void decompress_bc7_block(BC7Block& block, RGBA pixels[16]);

FINLINE void write_to_bc7_block(BC7Block& block, uint64_t data, uint32_t dataSize, uint32_t& blockIndex, uint32_t& dataOffset) {
	block.data[blockIndex] |= data << dataOffset;
	dataOffset += dataSize;
	if (dataOffset >= 64) {
		// This branch won't be hit more than once
		blockIndex = 1;
		dataOffset -= 64;
		block.data[1] |= data >> (dataSize - dataOffset);
	}
}

void write_bc7_block(BC7Block& block, uint32_t mode, uint32_t bestPartition, vec4f bestEndpoints[6], uint64_t bestIndices, uint64_t bestIndices2 = 0, uint32_t indexSelection = 0, uint32_t rotation = 0) {
	uint32_t numSubsets = bc7NumSubsets[mode];
	uint32_t partitionBits = bc7PartitionBitCounts[mode];
	uint32_t rotationBits = bc7RotationBitCounts[mode];
	uint32_t indexSelectionBit = bc7IndexSelectionBit[mode];
	uint32_t colorBits = bc7ColorBits[mode];
	uint32_t alphaBits = bc7AlphaBits[mode];
	uint32_t endpointPBits = bc7EndpointPBits[mode];
	uint32_t sharedPBits = bc7SharedPBits[mode];
	uint32_t numPBits = (endpointPBits << 1) + sharedPBits;
	uint32_t indexBits = bc7IndexBits[mode];
	uint32_t indexBits2 = bc7SecondaryIndexBits[mode];

	block.data[0] = 1 << mode;
	block.data[1] = 0;
	block.data[0] |= bestPartition << (mode + 1);

	uint32_t dataOffset = mode + 1 + partitionBits;
	block.data[0] |= rotation << dataOffset;
	dataOffset += rotationBits;
	block.data[0] |= indexSelection << dataOffset;
	dataOffset += indexSelectionBit;

	uint64_t endpointRGBA[4]{};
	uint32_t pBits = 0;

	uint64_t colorMask = (1 << colorBits) - 1;
	uint64_t alphaMask = (1 << alphaBits) - 1;
	uint64_t cutoffBits = 8 - colorBits;
	uint64_t alphaCutoffBits = 8 - alphaBits;
	for (uint32_t endpoint = 0; endpoint < (numSubsets * 2); endpoint++) {
		uint32_t r = static_cast<uint32_t>(bestEndpoints[endpoint].x);
		uint32_t g = static_cast<uint32_t>(bestEndpoints[endpoint].y);
		uint32_t b = static_cast<uint32_t>(bestEndpoints[endpoint].z);
		uint32_t a = static_cast<uint32_t>(bestEndpoints[endpoint].w);
		endpointRGBA[0] |= ((r >> cutoffBits) & colorMask) << (endpoint * colorBits);
		endpointRGBA[1] |= ((g >> cutoffBits) & colorMask) << (endpoint * colorBits);
		endpointRGBA[2] |= ((b >> cutoffBits) & colorMask) << (endpoint * colorBits);
		endpointRGBA[3] |= ((a >> alphaCutoffBits) & alphaMask) << (endpoint * alphaBits);
		if (numPBits == 2 || endpoint & numPBits) {
			uint32_t pShift = numPBits == 2 ? endpoint : endpoint >> 1;
			pBits |= ((r >> (cutoffBits - 1)) & 1) << pShift;
		}
	}

	uint32_t blockIndex = 0;
	for (uint32_t i = 0; i < (3 + (alphaBits > 0)); i++) {
		uint32_t dataSize = ((i == 3) ? alphaBits : colorBits) * numSubsets * 2;
		write_to_bc7_block(block, endpointRGBA[i], dataSize, blockIndex, dataOffset);
	}
	if (numPBits) {
		write_to_bc7_block(block, pBits, numSubsets * numPBits, blockIndex, dataOffset);
	}


	uint32_t anchor2nd = (numSubsets == 3) ? bc7PartitionTable3Anchors2ndSubset[bestPartition] :
		(numSubsets == 2) ? bc7PartitionTable2Anchors2ndSubset[bestPartition] :
		UINT32_MAX;
	uint32_t anchor3rd = (numSubsets == 3) ? bc7PartitionTable3Anchors3rdSubset[bestPartition] : UINT32_MAX;

	uint64_t encodedIndices = 0;
	uint32_t shift = 0;
	uint32_t indexMask = (1 << indexBits) - 1;
	for (uint32_t i = 0; i < 16; i++) {
		encodedIndices |= ((bestIndices >> (i * indexBits)) & indexMask) << shift;
		if ((i == 0) | (i == anchor2nd) | (i == anchor3rd)) {
			shift += indexBits - 1;
		} else {
			shift += indexBits;
		}
	}

	write_to_bc7_block(block, encodedIndices, shift, blockIndex, dataOffset);

	if (indexBits2) {
		encodedIndices = 0;
		shift = 0;
		indexMask = (1 << indexBits2) - 1;
		for (uint32_t i = 0; i < 16; i++) {
			encodedIndices |= ((bestIndices2 >> (i * indexBits2)) & indexMask) << shift;
			if ((i == 0) | (i == anchor2nd) | (i == anchor3rd)) {
				shift += indexBits2 - 1;
			} else {
				shift += indexBits2;
			}
		}
		write_to_bc7_block(block, encodedIndices, shift, blockIndex, dataOffset);
	}
}

FINLINE void write_to_bc7_blockx8(BC7Blockx4 blocks[2], uint64x4& data0, uint64x4& data1, uint32_t dataSize, uint32_t& blockIndex, uint32_t& dataOffset) {
	blocks[0].data[blockIndex] = _mm256_or_si256(blocks[0].data[blockIndex], _mm256_slli_epi64(data0, dataOffset));
	blocks[1].data[blockIndex] = _mm256_or_si256(blocks[1].data[blockIndex], _mm256_slli_epi64(data1, dataOffset));
	dataOffset += dataSize;
	if (dataOffset >= 64) {
		// This branch won't be hit more than once
		blockIndex = 1;
		dataOffset -= 64;
		blocks[0].data[1] = _mm256_or_si256(blocks[0].data[1], _mm256_srli_epi64(data0, dataSize - dataOffset));
		blocks[1].data[1] = _mm256_or_si256(blocks[1].data[1], _mm256_srli_epi64(data1, dataSize - dataOffset));
	}
}

FINLINE void encode_bc7_block_indicesx8(BC7Blockx4 blocks[2], uint64x4 bestIndices[2], uint32_t indexBits, uint32_t numSubsets, uint32x8& anchor2nd, uint32x8& anchor3rd, uint32_t& blockIndex, uint32_t& dataOffset) {
	uint32x8 indexBitsx8 = _mm256_set1_epi32(indexBits);

	uint64x4 encodedIndices[2]{ _mm256_setzero_si256(), _mm256_setzero_si256() };
	uint32x8 shift = _mm256_setzero_si256();
	uint64x4 indexMask = _mm256_set1_epi64x((1 << indexBits) - 1);
	for (uint32_t i = 0; i < 16; i++) {
		// encodedIndices |= ((bestIndices >> (i * indexBits)) & indexMask) << shift;
		uint64x4 shiftLow = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(shift));
		uint64x4 shiftHigh = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(shift, 1));
		encodedIndices[0] = _mm256_or_si256(encodedIndices[0], _mm256_sllv_epi64(_mm256_and_si256(_mm256_srli_epi64(bestIndices[0], i * indexBits), indexMask), shiftLow));
		encodedIndices[1] = _mm256_or_si256(encodedIndices[1], _mm256_sllv_epi64(_mm256_and_si256(_mm256_srli_epi64(bestIndices[1], i * indexBits), indexMask), shiftHigh));

		// Update shift
		uint32x8 ix8 = _mm256_set1_epi32(i);
		uint32x8 cmp1st = _mm256_cmpeq_epi32(ix8, _mm256_setzero_si256());
		uint32x8 cmp2nd = _mm256_cmpeq_epi32(ix8, anchor2nd);
		uint32x8 cmp3rd = _mm256_cmpeq_epi32(ix8, anchor3rd);
		uint32x8 anchor = _mm256_or_si256(_mm256_or_si256(cmp1st, cmp2nd), cmp3rd);
		uint32x8 tmp = _mm256_sub_epi32(indexBitsx8, _mm256_and_si256(anchor, _mm256_set1_epi32(1)));
		shift = _mm256_add_epi32(shift, tmp);
	}

	write_to_bc7_blockx8(blocks, encodedIndices[0], encodedIndices[1], indexBits * 16 - numSubsets, blockIndex, dataOffset);
}

template<typename Endpoint, uint32_t mode>
void write_bc7_blockx8(BC7Blockx4 blocks[2], uint32x8 bestPartition, Endpoint bestEndpoints[6], uint64x4 bestIndices[2], uint64x4 bestIndices2[2] = nullptr, uint32x8 indexSelection = _mm256_setzero_si256(), uint32x8 rotation = _mm256_setzero_si256()) {
	constexpr uint32_t numSubsets = bc7NumSubsets[mode];
	constexpr uint32_t partitionBits = bc7PartitionBitCounts[mode];
	constexpr uint32_t rotationBits = bc7RotationBitCounts[mode];
	constexpr uint32_t indexSelectionBit = bc7IndexSelectionBit[mode];
	constexpr uint32_t colorBits = bc7ColorBits[mode];
	constexpr uint32_t alphaBits = bc7AlphaBits[mode];
	constexpr uint32_t endpointPBits = bc7EndpointPBits[mode];
	constexpr uint32_t sharedPBits = bc7SharedPBits[mode];
	constexpr uint32_t numPBits = (endpointPBits << 1) + sharedPBits;
	constexpr uint32_t indexBits = bc7IndexBits[mode];
	constexpr uint32_t indexBits2 = bc7SecondaryIndexBits[mode];

	// Write out the first easy parts: mode, partition, rotation (if present), index selection (if present)
	uint32x8 startData = _mm256_or_si256(_mm256_or_si256(_mm256_or_si256(
		_mm256_set1_epi32(1 << mode),
		_mm256_slli_epi32(bestPartition, mode + 1)),
		_mm256_slli_epi32(rotation, mode + 1 + partitionBits)),
		_mm256_slli_epi32(indexSelection, mode + 1 + partitionBits + rotationBits));
	blocks[0].data[0] = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(startData));
	blocks[1].data[0] = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(startData, 1));
	blocks[0].data[1] = _mm256_setzero_si256();
	blocks[1].data[1] = _mm256_setzero_si256();

	uint32_t blockIndex = 0;
	uint32_t dataOffset = mode + 1 + partitionBits + rotationBits + indexSelectionBit;

	// Brace initialization is probably good enough, but just to be sure...
	uint32x8 endpointRGBA[4]{ _mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256() };
	uint32x8 pBits = _mm256_setzero_si256();

	// Gather the final endpoint color data in bit packed format
	uint32x8 colorMask = _mm256_set1_epi32((1 << colorBits) - 1);
	uint32x8 alphaMask = _mm256_set1_epi32((1 << alphaBits) - 1);
	uint32_t cutoffBits = 8 - colorBits;
	uint32_t alphaCutoffBits = 8 - alphaBits;
	for (uint32_t endpoint = 0; endpoint < (numSubsets * 2); endpoint++) {
		uint32x8 r = _mm256_cvtps_epi32(bestEndpoints[endpoint].x);
		uint32x8 g = _mm256_cvtps_epi32(bestEndpoints[endpoint].y);
		uint32x8 b = _mm256_cvtps_epi32(bestEndpoints[endpoint].z);
		endpointRGBA[0] = _mm256_or_si256(endpointRGBA[0], _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(r, cutoffBits), colorMask), endpoint * colorBits));
		endpointRGBA[1] = _mm256_or_si256(endpointRGBA[1], _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(g, cutoffBits), colorMask), endpoint * colorBits));
		endpointRGBA[2] = _mm256_or_si256(endpointRGBA[2], _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(b, cutoffBits), colorMask), endpoint * colorBits));
		if constexpr (alphaBits != 0) {
			uint32x8 a = _mm256_cvtps_epi32(bestEndpoints[endpoint].w);
			endpointRGBA[3] = _mm256_or_si256(endpointRGBA[3], _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(a, alphaCutoffBits), alphaMask), endpoint * alphaBits));
		}
		if (numPBits == 2 || endpoint & numPBits) {
			uint32_t pShift = numPBits == 2 ? endpoint : endpoint >> 1;
			pBits = _mm256_or_si256(pBits, _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(r, cutoffBits - 1), _mm256_set1_epi32(1)), pShift));
		}
	}

	// Write the endpoint data
	uint32_t colorFinalDataIndex = 0;
	for (uint32_t i = 0; i < (3 + (alphaBits > 0)); i++) {
		uint32_t componentSize = ((i == 3) ? alphaBits : colorBits) * numSubsets * 2;
		uint64x4 endpointCompLow = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(endpointRGBA[i]));
		uint64x4 endpointCompHigh = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(endpointRGBA[i], 1));
		write_to_bc7_blockx8(blocks, endpointCompLow, endpointCompHigh, componentSize, blockIndex, dataOffset);
	}
	if constexpr (numPBits) {
		uint32_t pbitsSize = numSubsets * numPBits;
		uint64x4 pbitsLow = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(pBits));
		uint64x4 pbitsHigh = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(pBits, 1));
		write_to_bc7_blockx8(blocks, pbitsLow, pbitsHigh, pbitsSize, blockIndex, dataOffset);
	}

	// Gather anchor indices
	uint32x8 anchor2nd;
	uint32x8 anchor3rd;
	if constexpr (numSubsets == 2) {
		anchor2nd = _mm256_i32gather_epi32(reinterpret_cast<const int*>(bc7PartitionTable2Anchors2ndSubset), bestPartition, 4);
		anchor3rd = _mm256_set1_epi32(UINT32_MAX);
	} else if constexpr (numSubsets == 3) {
		anchor2nd = _mm256_i32gather_epi32(reinterpret_cast<const int*>(bc7PartitionTable3Anchors2ndSubset), bestPartition, 4);
		anchor3rd = _mm256_i32gather_epi32(reinterpret_cast<const int*>(bc7PartitionTable3Anchors3rdSubset), bestPartition, 4);
	} else {
		anchor3rd = anchor2nd = _mm256_set1_epi32(UINT32_MAX);
	}

	// Encode indices into blocks
	encode_bc7_block_indicesx8(blocks, bestIndices, indexBits, numSubsets, anchor2nd, anchor3rd, blockIndex, dataOffset);
	if constexpr (indexBits2) {
		encode_bc7_block_indicesx8(blocks, bestIndices2, indexBits2, numSubsets, anchor2nd, anchor3rd, blockIndex, dataOffset);
	}
}

// Transforms random access partition data into something more usable by SIMD
// 16x8 bytewise transpose
// Each uint8x8 in the the tabledata contains one int for the index at that position in that partition table
FINLINE void transpose_partition_tables(uint32x8 tableIndices, uint32x8 tableData[16], const BC7PartitionTable tables[64]) {
	// gather all our tables
	uint8x16 t0 = _mm_load_si128(reinterpret_cast<const uint8x16*>(&tables[_mm256_extract_epi32(tableIndices, 0)].partitionNumbers));
	uint8x16 t1 = _mm_load_si128(reinterpret_cast<const uint8x16*>(&tables[_mm256_extract_epi32(tableIndices, 1)].partitionNumbers));
	uint8x16 t2 = _mm_load_si128(reinterpret_cast<const uint8x16*>(&tables[_mm256_extract_epi32(tableIndices, 2)].partitionNumbers));
	uint8x16 t3 = _mm_load_si128(reinterpret_cast<const uint8x16*>(&tables[_mm256_extract_epi32(tableIndices, 3)].partitionNumbers));
	uint8x16 t4 = _mm_load_si128(reinterpret_cast<const uint8x16*>(&tables[_mm256_extract_epi32(tableIndices, 4)].partitionNumbers));
	uint8x16 t5 = _mm_load_si128(reinterpret_cast<const uint8x16*>(&tables[_mm256_extract_epi32(tableIndices, 5)].partitionNumbers));
	uint8x16 t6 = _mm_load_si128(reinterpret_cast<const uint8x16*>(&tables[_mm256_extract_epi32(tableIndices, 6)].partitionNumbers));
	uint8x16 t7 = _mm_load_si128(reinterpret_cast<const uint8x16*>(&tables[_mm256_extract_epi32(tableIndices, 7)].partitionNumbers));

	// Transpose them to SoA

	/*
	// Do the smaller 4x4 transpose fisrt
	uint8x16 transpose4x4 = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
	t0 = _mm_shuffle_epi8(t0, transpose4x4);
	t1 = _mm_shuffle_epi8(t1, transpose4x4);
	t2 = _mm_shuffle_epi8(t2, transpose4x4);
	t3 = _mm_shuffle_epi8(t3, transpose4x4);
	t4 = _mm_shuffle_epi8(t4, transpose4x4);
	t5 = _mm_shuffle_epi8(t5, transpose4x4);
	t6 = _mm_shuffle_epi8(t6, transpose4x4);
	t7 = _mm_shuffle_epi8(t7, transpose4x4);*/

	// Pair up to groups of 2 (t0-3 each contain 2 lowest bytes)
	// What do t and p stand for? I don't know man, I just needed some short temp variable names. Maybe they stand for TransPose.

	// a0 a1 c0 c1...
	uint8x16 p0 = _mm_unpacklo_epi8(t0, t1);
	// a2 a3 c2 c3...
	uint8x16 p1 = _mm_unpacklo_epi8(t2, t3);
	// a4 a5 c4 c5...
	uint8x16 p2 = _mm_unpacklo_epi8(t4, t5);
	// a6 a7 c6 c7...
	uint8x16 p3 = _mm_unpacklo_epi8(t6, t7);
	// b0 b1 d0 d1...
	uint8x16 p4 = _mm_unpackhi_epi8(t0, t1);
	uint8x16 p5 = _mm_unpackhi_epi8(t2, t3);
	uint8x16 p6 = _mm_unpackhi_epi8(t4, t5);
	uint8x16 p7 = _mm_unpackhi_epi8(t6, t7);
	// Pair up to groups of 4 (t0-1 each contain 4 lowest bytes)
	t0 = _mm_unpacklo_epi16(p0, p1);
	t1 = _mm_unpacklo_epi16(p2, p3);
	t2 = _mm_unpacklo_epi16(p4, p5);
	t3 = _mm_unpacklo_epi16(p6, p7);
	t4 = _mm_unpackhi_epi16(p0, p1);
	t5 = _mm_unpackhi_epi16(p2, p3);
	t6 = _mm_unpackhi_epi16(p4, p5);
	t7 = _mm_unpackhi_epi16(p6, p7);
	// Pair up to groups of 8 (t0 now contains all 8 low bytes, ready to be converted)
	p0 = _mm_unpacklo_epi32(t0, t1);
	p1 = _mm_unpackhi_epi32(t0, t1);
	p2 = _mm_unpacklo_epi32(t4, t5);
	p3 = _mm_unpackhi_epi32(t4, t5);
	p4 = _mm_unpacklo_epi32(t2, t3);
	p5 = _mm_unpackhi_epi32(t2, t3);
	p6 = _mm_unpacklo_epi32(t6, t7);
	p7 = _mm_unpackhi_epi32(t6, t7);

	tableData[0] = _mm256_cvtepi8_epi32(p0);
	tableData[1] = _mm256_cvtepi8_epi32(_mm_srli_si128(p0, 8));
	tableData[2] = _mm256_cvtepi8_epi32(p1);
	tableData[3] = _mm256_cvtepi8_epi32(_mm_srli_si128(p1, 8));
	tableData[4] = _mm256_cvtepi8_epi32(p2);
	tableData[5] = _mm256_cvtepi8_epi32(_mm_srli_si128(p2, 8));
	tableData[6] = _mm256_cvtepi8_epi32(p3);
	tableData[7] = _mm256_cvtepi8_epi32(_mm_srli_si128(p3, 8));
	tableData[8] = _mm256_cvtepi8_epi32(p4);
	tableData[9] = _mm256_cvtepi8_epi32(_mm_srli_si128(p4, 8));
	tableData[10] = _mm256_cvtepi8_epi32(p5);
	tableData[11] = _mm256_cvtepi8_epi32(_mm_srli_si128(p5, 8));
	tableData[12] = _mm256_cvtepi8_epi32(p6);
	tableData[13] = _mm256_cvtepi8_epi32(_mm_srli_si128(p6, 8));
	tableData[14] = _mm256_cvtepi8_epi32(p7);
	tableData[15] = _mm256_cvtepi8_epi32(_mm_srli_si128(p7, 8));
}

constexpr bool mode0Enable = false;
constexpr bool mode1Enable = true;
constexpr bool mode2Enable = false;
constexpr bool mode3Enable = false;
constexpr bool mode4Enable = false;
constexpr bool mode5Enable = false;
constexpr bool mode6Enable = true;
constexpr bool mode7Enable = false;

constexpr bool modeEnables[]{ mode0Enable, mode1Enable, mode2Enable, mode3Enable, mode4Enable, mode5Enable, mode6Enable, mode7Enable };

template<uint32_t mode, typename Endpoint>
floatx8 compress_bc7_block_mode01237x8(vec4fx8 pixels[16], BC7Blockx4 blocks[2], vec3fx8* mins, vec3fx8* maxes, floatx8* alphaMins, floatx8* alphaMaxes) {
	if (!modeEnables[mode]) {
		return _mm256_set1_ps(FLT_MAX);
	}

	constexpr uint32_t modePartitions = bc7NumSubsets[mode];
	constexpr uint32_t modeComponentBits = bc7ColorBits[mode];
	// If has endpoint pbits, 2, else if has shared endpoint pbits, 1, else 0
	constexpr uint32_t modeNumPBits = bc7EndpointPBits[mode] * 2 + bc7SharedPBits[mode];
	constexpr uint32_t modeIndexResolution = bc7IndexBits[mode];

	//Check each possible partition and find the one with lowest error
	floatx8 bestError = _mm256_set1_ps(FLT_MAX);
	uint32x8 bestPartition{};
	Endpoint bestEndpoints[modePartitions * 2]{};
	uint64x4 bestIndices[2]{};
	//It might be much better for speed to try to find the best pattern or several patterns first with a basic distance from points to line rather than checking them all.
	constexpr const BC7PartitionTable* partitionTables = modePartitions == 2 ? bc7PartitionTable2Subsets : bc7PartitionTable3Subsets;
	constexpr uint32_t partitionTablesToCheck = 1 << bc7PartitionBitCounts[mode];
	for (uint32_t partitionTable = 0; partitionTable < partitionTablesToCheck; partitionTable++) {
		const BC7PartitionTable& table = partitionTables[partitionTable];
		//Original endpoints can be min and max
		Endpoint endpoints[modePartitions * 2];
		if constexpr (mode == 7) {
			endpoints[0] = Endpoint{ mins[partitionTable * modePartitions + 0], alphaMins[partitionTable * modePartitions + 0] };
			endpoints[1] = Endpoint{ maxes[partitionTable * modePartitions + 0], alphaMaxes[partitionTable * modePartitions + 0] };
			endpoints[2] = Endpoint{ mins[partitionTable * modePartitions + 1], alphaMins[partitionTable * modePartitions + 1] };
			endpoints[3] = Endpoint{ maxes[partitionTable * modePartitions + 1], alphaMaxes[partitionTable * modePartitions + 1] };

			choose_best_diagonals_rgba<modePartitions>(pixels, endpoints, table);
		} else {
			if constexpr (modePartitions == 1) {
				endpoints[0] = min(mins[0], mins[1]);
				endpoints[1] = max(maxes[0], maxes[1]);
			} else if constexpr (modePartitions == 2 || modePartitions == 3) {
				endpoints[0] = mins[partitionTable * modePartitions + 0];
				endpoints[1] = maxes[partitionTable * modePartitions + 0];
				endpoints[2] = mins[partitionTable * modePartitions + 1];
				endpoints[3] = maxes[partitionTable * modePartitions + 1];
			}
			if constexpr (modePartitions == 3) {
				endpoints[4] = mins[partitionTable * modePartitions + 2];
				endpoints[5] = maxes[partitionTable * modePartitions + 2];
			}

			choose_best_diagonals<modePartitions>(pixels, endpoints, table);
		}
		
		least_squares_optimize_endpointsx8<Endpoint, modeIndexResolution, modePartitions>(pixels, endpoints, table);

		// Figure out what our indices are going to be for the optimized endpoints
		uint64x4 indices[2];
		floatx8 error = quantize_bc7_endpointsx8<modePartitions, modeComponentBits, modeNumPBits, modeIndexResolution>(pixels, endpoints, table, indices);

		// Check if the errors are the current best, and write out the best values
		floatx8 errorLessThan = _mm256_cmp_ps(error, bestError, _CMP_LT_OQ);
		uint32x8 partitionTablex8 = _mm256_set1_epi32(partitionTable);
		for (uint32_t i = 0; i < (modePartitions * 2); i++) {
			bestEndpoints[i] = blend(bestEndpoints[i], endpoints[i], errorLessThan);
		}
		// Too much casting going on here
		// bestIndices[0] = blend(bestIndices[0], indices[0], signextend32to64(errorLessThanLowerHalf))
		bestIndices[0] = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(bestIndices[0]), _mm256_castsi256_ps(indices[0]), _mm256_castsi256_ps(_mm256_cvtepi32_epi64(_mm256_castsi256_si128(_mm256_castps_si256(errorLessThan))))));
		// bestIndices[1] = blend(bestIndices[1], indices[1], signextend32to64(errorLessThanUpperHalf))
		bestIndices[1] = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(bestIndices[1]), _mm256_castsi256_ps(indices[1]), _mm256_castsi256_ps(_mm256_cvtepi32_epi64(_mm256_extracti128_si256(_mm256_castps_si256(errorLessThan), 1)))));
		bestPartition = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(bestPartition), _mm256_castsi256_ps(partitionTablex8), errorLessThan));
		bestError = _mm256_blendv_ps(bestError, error, errorLessThan);
	}
	uint32x8 tablesPacked[16];
	transpose_partition_tables(bestPartition, tablesPacked, partitionTables);
	check_flip_indices<Endpoint, modePartitions, modeIndexResolution>(bestEndpoints, bestIndices, bestPartition, tablesPacked);
	write_bc7_blockx8<Endpoint, mode>(blocks, bestPartition, bestEndpoints, bestIndices);
	return bestError;
}

floatx8 compress_bc7_block_mode4x8(vec4fx8 pixels[16], BC7Blockx4 blocks[2], vec3fx8 mins[numPartitionsFor2Subsets], vec3fx8 maxes[numPartitionsFor2Subsets], floatx8 minAlphas[numPartitionsFor2Subsets], floatx8 maxAlphas[numPartitionsFor2Subsets]) {
	if (!mode4Enable) {
		return _mm256_set1_ps(FLT_MAX);
	}
	constexpr uint32_t mode4Partitions = 1;
	constexpr uint32_t mode4ComponentBits = 5;
	constexpr uint32_t mode4ComponentAlphaBits = 6;
	constexpr uint32_t mode4PBitsPerParition = 0;
	constexpr uint32_t mode4IndexResolution1 = 2;
	constexpr uint32_t mode4IndexResolution2 = 3;

	vec3fx8 endpoints[mode4Partitions * 2]{
		min(mins[0], mins[1]),
		max(maxes[0], maxes[1])
	};
	floatx8 endpointAlphas[mode4Partitions * 2]{
		_mm256_min_ps(minAlphas[0], minAlphas[1]),
		_mm256_max_ps(maxAlphas[0], maxAlphas[1])
	};

	vec3fx8 bestEndpoints[mode4Partitions * 2];
	floatx8 bestEndpointAlphas[mode4Partitions * 2];
	uint64x4 bestIndices1[2];
	uint64x4 bestIndices2[2];
	floatx8 bestError = _mm256_set1_ps(FLT_MAX);
	uint32x8 bestRotation;
	uint32x8 bestIndexSelection;

	vec3fx8 optimizedEndpoints[mode4Partitions * 2];
	floatx8 optimizedEndpointAlphas[mode4Partitions * 2];
	uint64x4 indices1[2];
	uint64x4 indices2[2];
	floatx8 error;

	floatx8 cmp;
	floatx8 cmpLow;
	floatx8 cmpHigh;

	uint32x8 blockOfZeros[16]{};

	// Quick and dirty code deduplication so I don't end up with a 300 line monster
#define CHECK_CONFIGURATION(idxRes1, idxRes2, indc1, indc2, idxSelect, rot) optimizedEndpoints[0] = endpoints[0];\
	optimizedEndpoints[1] = endpoints[1];\
	optimizedEndpointAlphas[0] = endpointAlphas[0];\
	optimizedEndpointAlphas[1] = endpointAlphas[1];\
	choose_best_diagonals<mode4Partitions>(pixels, optimizedEndpoints, bc7DummyPartitionTableAllZeros);\
	\
	least_squares_optimize_endpointsx8<vec3fx8, idxRes1, mode4Partitions>(pixels, optimizedEndpoints, bc7DummyPartitionTableAllZeros);\
	least_squares_optimize_endpointsx8<floatx8, idxRes2, mode4Partitions>(pixels, optimizedEndpointAlphas, bc7DummyPartitionTableAllZeros);\
	error = quantize_bc7_endpointsx8<mode4Partitions, mode4ComponentBits, mode4PBitsPerParition, idxRes1>(pixels, optimizedEndpoints, bc7DummyPartitionTableAllZeros, indc1);\
	error = _mm256_add_ps(error, quantize_bc7_endpointsx8<mode4Partitions, mode4ComponentAlphaBits, mode4PBitsPerParition, idxRes2, floatx8>(pixels, optimizedEndpointAlphas, bc7DummyPartitionTableAllZeros, indc2));\
	\
	check_flip_indices<vec3fx8, mode4Partitions, idxRes1>(optimizedEndpoints, indc1, _mm256_setzero_si256(), blockOfZeros);\
	check_flip_indices<floatx8, mode4Partitions, idxRes2>(optimizedEndpointAlphas, indc2, _mm256_setzero_si256(), blockOfZeros);\
	\
	cmp = _mm256_cmp_ps(error, bestError, _CMP_LT_OQ);\
	extract_lo_hi_masks(cmp, &cmpLow, &cmpHigh);\
	bestEndpoints[0] = blend(bestEndpoints[0], optimizedEndpoints[0], cmp);\
	bestEndpoints[1] = blend(bestEndpoints[1], optimizedEndpoints[1], cmp);\
	bestEndpointAlphas[0] = blend(bestEndpointAlphas[0], optimizedEndpointAlphas[0], cmp);\
	bestEndpointAlphas[1] = blend(bestEndpointAlphas[1], optimizedEndpointAlphas[1], cmp);\
	bestIndices1[0] = blend(bestIndices1[0], indices1[0], cmpLow);\
	bestIndices1[1] = blend(bestIndices1[1], indices1[1], cmpHigh);\
	bestIndices2[0] = blend(bestIndices2[0], indices2[0], cmpLow);\
	bestIndices2[1] = blend(bestIndices2[1], indices2[1], cmpHigh);\
	bestRotation = blend(bestRotation, _mm256_set1_epi32(rot), cmp);\
	bestError = _mm256_min_ps(bestError, error);\
	bestIndexSelection = blend(bestIndexSelection, _mm256_set1_epi32(idxSelect), cmp);


	// Unchanged rotation
	CHECK_CONFIGURATION(mode4IndexResolution1, mode4IndexResolution2, indices1, indices2, 0, 0);
	CHECK_CONFIGURATION(mode4IndexResolution2, mode4IndexResolution1, indices2, indices1, 1, 0);

	// Rotate R and A
	std::swap(endpoints[0].x, endpointAlphas[0]);
	std::swap(endpoints[1].x, endpointAlphas[1]);
	for (uint32_t i = 0; i < 16; i++) {
		std::swap(pixels[i].x, pixels[i].w);
	}

	CHECK_CONFIGURATION(mode4IndexResolution1, mode4IndexResolution2, indices1, indices2, 0, 1);
	CHECK_CONFIGURATION(mode4IndexResolution2, mode4IndexResolution1, indices2, indices1, 1, 1);

	// Rotate G and A
	std::swap(endpoints[0].x, endpointAlphas[0]);
	std::swap(endpoints[1].x, endpointAlphas[1]);
	std::swap(endpoints[0].y, endpointAlphas[0]);
	std::swap(endpoints[1].y, endpointAlphas[1]);
	for (uint32_t i = 0; i < 16; i++) {
		std::swap(pixels[i].x, pixels[i].w);
		std::swap(pixels[i].y, pixels[i].w);
	}

	CHECK_CONFIGURATION(mode4IndexResolution1, mode4IndexResolution2, indices1, indices2, 0, 2);
	CHECK_CONFIGURATION(mode4IndexResolution2, mode4IndexResolution1, indices2, indices1, 1, 2);

	// Rotate B and A
	std::swap(endpoints[0].y, endpointAlphas[0]);
	std::swap(endpoints[1].y, endpointAlphas[1]);
	std::swap(endpoints[0].z, endpointAlphas[0]);
	std::swap(endpoints[1].z, endpointAlphas[1]);
	for (uint32_t i = 0; i < 16; i++) {
		std::swap(pixels[i].y, pixels[i].w);
		std::swap(pixels[i].z, pixels[i].w);
	}

	CHECK_CONFIGURATION(mode4IndexResolution1, mode4IndexResolution2, indices1, indices2, 0, 3);
	CHECK_CONFIGURATION(mode4IndexResolution2, mode4IndexResolution1, indices2, indices1, 1, 3);

	// Reset pixels after
	for (uint32_t i = 0; i < 16; i++) {
		std::swap(pixels[i].z, pixels[i].w);
	}

#undef CHECK_CONFIGURATION

	vec4fx8 finalEndpoints[2]{
		{ bestEndpoints[0], bestEndpointAlphas[0] },
		{ bestEndpoints[1], bestEndpointAlphas[1] }
	};
	write_bc7_blockx8<vec4fx8, 4>(blocks, _mm256_setzero_si256(), finalEndpoints, bestIndices1, bestIndices2, bestIndexSelection, bestRotation);

	return bestError;
}

floatx8 compress_bc7_block_mode5x8(vec4fx8 pixels[16], BC7Blockx4 blocks[2], vec3fx8 mins[numPartitionsFor2Subsets], vec3fx8 maxes[numPartitionsFor2Subsets], floatx8 minAlphas[numPartitionsFor2Subsets], floatx8 maxAlphas[numPartitionsFor2Subsets]) {
	if (!mode5Enable) {
		return _mm256_set1_ps(FLT_MAX);
	}
	constexpr uint32_t mode5Partitions = 1;
	constexpr uint32_t mode5ComponentBits = 7;
	constexpr uint32_t mode5ComponentAlphaBits = 8;
	constexpr uint32_t mode5PBitsPerParition = 0;
	constexpr uint32_t mode5IndexResolution = 2;

	vec3fx8 endpoints[mode5Partitions * 2]{
		min(mins[0], mins[1]),
		max(maxes[0], maxes[1])
	};
	floatx8 endpointAlphas[mode5Partitions * 2]{
		_mm256_min_ps(minAlphas[0], minAlphas[1]),
		_mm256_max_ps(maxAlphas[0], maxAlphas[1])
	};

	vec3fx8 bestEndpoints[mode5Partitions * 2];
	floatx8 bestEndpointAlphas[mode5Partitions * 2];
	uint64x4 bestIndices1[2];
	uint64x4 bestIndices2[2];
	floatx8 bestError = _mm256_set1_ps(FLT_MAX);
	uint32x8 bestRotation;

	vec3fx8 optimizedEndpoints[mode5Partitions * 2];
	floatx8 optimizedEndpointAlphas[mode5Partitions * 2];
	uint64x4 indices1[2];
	uint64x4 indices2[2];
	floatx8 error;

	floatx8 cmp;
	floatx8 cmpLow;
	floatx8 cmpHigh;

	uint32x8 blockOfZeros[16]{};

	// Quick and dirty code deduplication so I don't end up with a 300 line monster
#define CHECK_CONFIGURATION(rot) optimizedEndpoints[0] = endpoints[0];\
	optimizedEndpoints[1] = endpoints[1];\
	optimizedEndpointAlphas[0] = endpointAlphas[0];\
	optimizedEndpointAlphas[1] = endpointAlphas[1];\
	choose_best_diagonals<mode5Partitions>(pixels, optimizedEndpoints, bc7DummyPartitionTableAllZeros);\
	\
	least_squares_optimize_endpointsx8<vec3fx8, mode5IndexResolution, mode5Partitions>(pixels, optimizedEndpoints, bc7DummyPartitionTableAllZeros);\
	least_squares_optimize_endpointsx8<floatx8, mode5IndexResolution, mode5Partitions>(pixels, optimizedEndpointAlphas, bc7DummyPartitionTableAllZeros);\
	error = quantize_bc7_endpointsx8<mode5Partitions, mode5ComponentBits, mode5PBitsPerParition, mode5IndexResolution>(pixels, optimizedEndpoints, bc7DummyPartitionTableAllZeros, indices1);\
	error = _mm256_add_ps(error, quantize_bc7_endpointsx8<mode5Partitions, mode5ComponentAlphaBits, mode5PBitsPerParition, mode5IndexResolution, floatx8>(pixels, optimizedEndpointAlphas, bc7DummyPartitionTableAllZeros, indices2));\
	\
	check_flip_indices<vec3fx8, mode5Partitions, mode5IndexResolution>(optimizedEndpoints, indices1, _mm256_setzero_si256(), blockOfZeros);\
	check_flip_indices<floatx8, mode5Partitions, mode5IndexResolution>(optimizedEndpointAlphas, indices2, _mm256_setzero_si256(), blockOfZeros);\
	\
	cmp = _mm256_cmp_ps(error, bestError, _CMP_LT_OQ);\
	extract_lo_hi_masks(cmp, &cmpLow, &cmpHigh);\
	bestEndpoints[0] = blend(bestEndpoints[0], optimizedEndpoints[0], cmp);\
	bestEndpoints[1] = blend(bestEndpoints[1], optimizedEndpoints[1], cmp);\
	bestEndpointAlphas[0] = blend(bestEndpointAlphas[0], optimizedEndpointAlphas[0], cmp);\
	bestEndpointAlphas[1] = blend(bestEndpointAlphas[1], optimizedEndpointAlphas[1], cmp);\
	bestIndices1[0] = blend(bestIndices1[0], indices1[0], cmpLow);\
	bestIndices1[1] = blend(bestIndices1[1], indices1[1], cmpHigh);\
	bestIndices2[0] = blend(bestIndices2[0], indices2[0], cmpLow);\
	bestIndices2[1] = blend(bestIndices2[1], indices2[1], cmpHigh);\
	bestRotation = blend(bestRotation, _mm256_set1_epi32(rot), cmp);\
	bestError = _mm256_min_ps(bestError, error);\


	// Unchanged rotation
	CHECK_CONFIGURATION(0);

	// Rotate R and A
	std::swap(endpoints[0].x, endpointAlphas[0]);
	std::swap(endpoints[1].x, endpointAlphas[1]);
	for (uint32_t i = 0; i < 16; i++) {
		std::swap(pixels[i].x, pixels[i].w);
	}

	CHECK_CONFIGURATION(1);

	// Rotate G and A
	std::swap(endpoints[0].x, endpointAlphas[0]);
	std::swap(endpoints[1].x, endpointAlphas[1]);
	std::swap(endpoints[0].y, endpointAlphas[0]);
	std::swap(endpoints[1].y, endpointAlphas[1]);
	for (uint32_t i = 0; i < 16; i++) {
		std::swap(pixels[i].x, pixels[i].w);
		std::swap(pixels[i].y, pixels[i].w);
	}

	CHECK_CONFIGURATION(2);

	// Rotate B and A
	std::swap(endpoints[0].y, endpointAlphas[0]);
	std::swap(endpoints[1].y, endpointAlphas[1]);
	std::swap(endpoints[0].z, endpointAlphas[0]);
	std::swap(endpoints[1].z, endpointAlphas[1]);
	for (uint32_t i = 0; i < 16; i++) {
		std::swap(pixels[i].y, pixels[i].w);
		std::swap(pixels[i].z, pixels[i].w);
	}

	CHECK_CONFIGURATION(3);

	// Reset pixels after
	for (uint32_t i = 0; i < 16; i++) {
		std::swap(pixels[i].z, pixels[i].w);
	}

#undef CHECK_CONFIGURATION

	vec4fx8 finalEndpoints[2]{
		{ bestEndpoints[0], bestEndpointAlphas[0] },
		{ bestEndpoints[1], bestEndpointAlphas[1] }
	};
	write_bc7_blockx8<vec4fx8, 5>(blocks, _mm256_setzero_si256(), finalEndpoints, bestIndices1, bestIndices2, _mm256_setzero_si256(), bestRotation);
	return bestError;
}

floatx8 compress_bc7_block_mode6x8(vec4fx8 pixels[16], BC7Blockx4 blocks[2], vec3fx8 mins[numPartitionsFor2Subsets], vec3fx8 maxes[numPartitionsFor2Subsets], floatx8 minAlphas[numPartitionsFor2Subsets], floatx8 maxAlphas[numPartitionsFor2Subsets]) {
	if (!mode6Enable) {
		return _mm256_set1_ps(FLT_MAX);
	}
	constexpr uint32_t mode6Partitions = 1;
	constexpr uint32_t mode6ComponentBits = 7;
	constexpr uint32_t mode6ComponentAlphaBits = 7;
	constexpr uint32_t mode6PBitsPerParition = 2;
	constexpr uint32_t mode6IndexResolution = 4;

	vec4fx8 endpoints[mode6Partitions * 2]{
		{ min(mins[0], mins[1]), _mm256_min_ps(minAlphas[0], minAlphas[1]) },
		{ max(maxes[0], maxes[1]), _mm256_max_ps(maxAlphas[0], maxAlphas[1]) }
	};

	uint64x4 indices[2];
	//choose_best_diagonals<mode6Partitions>(pixels, endpoints, bc7DummyPartitionTableAllZeros);
	least_squares_optimize_endpointsx8<vec4fx8, mode6IndexResolution, mode6Partitions>(pixels, endpoints, bc7DummyPartitionTableAllZeros);
	floatx8 error = quantize_bc7_endpointsx8<mode6Partitions, mode6ComponentBits, mode6PBitsPerParition, mode6IndexResolution>(pixels, endpoints, bc7DummyPartitionTableAllZeros, indices);
	uint32x8 blockOfZeros[16]{};
	check_flip_indices<vec4fx8, mode6Partitions, mode6IndexResolution>(endpoints, indices, _mm256_setzero_si256(), blockOfZeros);
	write_bc7_blockx8<vec4fx8, 6>(blocks, _mm256_setzero_si256(), endpoints, indices);
	return error;
}

float compress_bc7_block_mode0(vec4f pixels[16], BC7Block& block, vec3f mins[numPartitionsFor3Subsets], vec3f maxes[numPartitionsFor3Subsets], vec3f means[numPartitionsFor3Subsets]) {
	if (!mode0Enable) {
		return FLT_MAX;
	}
	constexpr uint32_t mode0Partitions = 3;
	constexpr uint32_t mode0ComponentBits = 4;
	constexpr uint32_t mode0NumPBits = 2;
	constexpr uint32_t mode0IndexResolution = 3;

	//Check each possible partition and find the one with lowest error
	float bestError = FLT_MAX;
	uint32_t bestPartition;
	vec4f bestEndpoints[6];
	uint64_t bestIndices;
	//Mode 0 only gets 4 bits of partition table index instead of the full 6, so shift the number right by 2.
	//It might be much better for speed to try to find the best pattern or several patterns first with a basic distance from points to line rather than checking them all.
	for (uint32_t partitionTable = 0; partitionTable < (numPartitionTablesPerSubset >> 2); partitionTable++) {
		const BC7PartitionTable& table = bc7PartitionTable3Subsets[partitionTable];
		//Original endpoints can be min and max
		vec3f endpoints[mode0Partitions * 2]{
			mins[partitionTable * mode0Partitions + 0],
			maxes[partitionTable * mode0Partitions + 0],
			mins[partitionTable * mode0Partitions + 1],
			maxes[partitionTable * mode0Partitions + 1],
			mins[partitionTable * mode0Partitions + 2],
			maxes[partitionTable * mode0Partitions + 2]
		};
		//I didn't think it was at first, but this is actually required for the degenerate case where the points all lie on opposing diagonals, projecting to the same point on the original diagonal (which results in least squares erroring).
		choose_best_diagonals<mode0Partitions>(pixels, endpoints, table);
		//Mode 0 has 3 partitions
		for (uint32_t partition = 0; partition < mode0Partitions; partition++) {
			//Find indices from the original endpoints and optimize with a least squares iteration to get the new endpoints
			least_squares_optimize_endpoints_rgb<mode0IndexResolution>(pixels, &endpoints[partition * 2], table, partition);
		}

		/*quantize_bc7_endpoints_mode0(endpoints);

		//Calc final indices
		uint64_t indices = 0;
		for (uint32_t i = 0; i < 16; i++) {
			uint32_t endpointIndex = table.partitionNumbers[i] * 2;
			//3 bit indices
			indices |= static_cast<uint64_t>(find_index3(&endpoints[endpointIndex], pixels[i].xyz() * 255.0F)) << (i * 3);
		}*/

		uint64_t indices;
		//float error = quantize_bc7_endpoints3_mode0(pixels, endpoints, table, &indices);
		float error = quantize_bc7_endpoints<mode0Partitions, mode0ComponentBits, mode0NumPBits, mode0IndexResolution>(pixels, endpoints, table, &indices);

		//Test this partition table's error, see if it's the best one yet
		//float error = error_mode0(pixels, partitionTable, endpoints, indices);
		if (error < bestError) {
			check_flip_indices<vec3f, 3, 3>(endpoints, &indices, partitionTable, table);
			for (uint32_t i = 0; i < (mode0Partitions * 2); i++) {
				bestEndpoints[i] = vec4f{ endpoints[i].x, endpoints[i].y, endpoints[i].z, 0.0F };
			}
			bestIndices = indices;
			bestPartition = partitionTable;
			bestError = error;
		}
	}
	//write_bc7_block_mode0(block, bestPartition, bestEndpoints, bestIndices);
	write_bc7_block(block, 0, bestPartition, bestEndpoints, bestIndices);
	return bestError;
}

float compress_bc7_block_mode1(vec4f pixels[16], BC7Block& block, vec3f mins[numPartitionsFor2Subsets], vec3f maxes[numPartitionsFor2Subsets], vec3f means[numPartitionsFor2Subsets]) {
	if (!mode1Enable) {
		return FLT_MAX;
	}

	constexpr uint32_t mode1Partitions = 2;
	constexpr uint32_t mode1ComponentBits = 6;
	constexpr uint32_t mode1NumPBits = 1;
	constexpr uint32_t mode1IndexResolution = 3;

	//Check each possible partition and find the one with lowest error
	float bestError = FLT_MAX;
	uint32_t bestPartition;
	vec4f bestEndpoints[4];
	uint64_t bestIndices;
	//It might be much better for speed to try to find the best pattern or several patterns first with a basic distance from points to line rather than checking them all.
	for (uint32_t partitionTable = 0; partitionTable < numPartitionTablesPerSubset; partitionTable++) {
		const BC7PartitionTable& table = bc7PartitionTable2Subsets[partitionTable];
		//Original endpoints can be min and max
		vec3f endpoints[mode1Partitions * 2]{
			mins[partitionTable * mode1Partitions + 0],
			maxes[partitionTable * mode1Partitions + 0],
			mins[partitionTable * mode1Partitions + 1],
			maxes[partitionTable * mode1Partitions + 1]
		};
		//I didn't think it was at first, but this is actually required for the degenerate case where the points all lie on opposing diagonals, projecting to the same point on the original diagonal (which results in least squares erroring).
		choose_best_diagonals<mode1Partitions>(pixels, endpoints, table);
		//Mode 1 has 2 partitions
		for (uint32_t partition = 0; partition < mode1Partitions; partition++) {
			//Find indices from the original endpoints and optimize with a least squares iteration to get the new endpoints
			least_squares_optimize_endpoints_rgb<mode1IndexResolution>(pixels, &endpoints[partition * 2], table, partition);
		}

		uint64_t indices;
		//float error = quantize_bc7_endpoints3_mode1(pixels, endpoints, table, &indices);
		float error = quantize_bc7_endpoints<mode1Partitions, mode1ComponentBits, mode1NumPBits, mode1IndexResolution>(pixels, endpoints, table, &indices);

		//Test this partition table's error, see if it's the best one yet
		//float error = error_mode0(pixels, partitionTable, endpoints, indices);
		if (error < bestError) {
			check_flip_indices<vec3f, 2, 3>(endpoints, &indices, partitionTable, table);
			for (uint32_t i = 0; i < (mode1Partitions * 2); i++) {
				bestEndpoints[i] = vec4f{ endpoints[i].x, endpoints[i].y, endpoints[i].z, 0.0F };
			}
			bestIndices = indices;
			bestPartition = partitionTable;
			bestError = error;
		}
	}
	//write_bc7_block_mode1(block, bestPartition, bestEndpoints, bestIndices);
	write_bc7_block(block, 1, bestPartition, bestEndpoints, bestIndices);
	return bestError;
}

float compress_bc7_block_mode2(vec4f pixels[16], BC7Block& block, vec3f mins[numPartitionsFor2Subsets], vec3f maxes[numPartitionsFor2Subsets], vec3f means[numPartitionsFor2Subsets]) {
	if (!mode2Enable) {
		return FLT_MAX;
	}
	constexpr uint32_t mode2Partitions = 3;
	constexpr uint32_t mode2ComponentBits = 5;
	constexpr uint32_t mode2PBitsPerParition = 0;
	constexpr uint32_t mode2IndexResolution = 2;

	//Check each possible partition and find the one with lowest error
	float bestError = FLT_MAX;
	uint32_t bestPartition;
	vec4f bestEndpoints[mode2Partitions * 2];
	uint64_t bestIndices;
	//It might be much better for speed to try to find the best pattern or several patterns first with a basic distance from points to line rather than checking them all.
	for (uint32_t partitionTable = 0; partitionTable < numPartitionTablesPerSubset; partitionTable++) {
		const BC7PartitionTable& table = bc7PartitionTable3Subsets[partitionTable];
		//Original endpoints can be min and max
		vec3f endpoints[mode2Partitions * 2]{
			mins[partitionTable * mode2Partitions + 0],
			maxes[partitionTable * mode2Partitions + 0],
			mins[partitionTable * mode2Partitions + 1],
			maxes[partitionTable * mode2Partitions + 1],
			mins[partitionTable * mode2Partitions + 2],
			maxes[partitionTable * mode2Partitions + 2]
		};
		//I didn't think it was at first, but this is actually required for the degenerate case where the points all lie on opposing diagonals, projecting to the same point on the original diagonal (which results in least squares erroring).
		choose_best_diagonals<mode2Partitions>(pixels, endpoints, table);
		//Mode 1 has 2 partitions
		for (uint32_t partition = 0; partition < mode2Partitions; partition++) {
			//Find indices from the original endpoints and optimize with a least squares iteration to get the new endpoints
			least_squares_optimize_endpoints_rgb<mode2IndexResolution>(pixels, &endpoints[partition * 2], table, partition);
		}

		uint64_t indices;
		float error = quantize_bc7_endpoints<mode2Partitions, mode2ComponentBits, mode2PBitsPerParition, mode2IndexResolution>(pixels, endpoints, table, &indices);

		//Test this partition table's error, see if it's the best one yet
		if (error < bestError) {
			check_flip_indices<vec3f, mode2Partitions, mode2IndexResolution>(endpoints, &indices, partitionTable, table);
			for (uint32_t i = 0; i < (mode2Partitions * 2); i++) {
				bestEndpoints[i] = vec4f{ endpoints[i].x, endpoints[i].y, endpoints[i].z, 0.0F };
			}
			bestIndices = indices;
			bestPartition = partitionTable;
			bestError = error;
		}
	}
	write_bc7_block(block, 2, bestPartition, bestEndpoints, bestIndices);
	return bestError;
}

float compress_bc7_block_mode3(vec4f pixels[16], BC7Block& block, vec3f mins[numPartitionsFor2Subsets], vec3f maxes[numPartitionsFor2Subsets], vec3f means[numPartitionsFor2Subsets]) {
	if (!mode3Enable) {
		return FLT_MAX;
	}
	constexpr uint32_t mode3Partitions = 2;
	constexpr uint32_t mode3ComponentBits = 7;
	constexpr uint32_t mode3PBitsPerParition = 2;
	constexpr uint32_t mode3IndexResolution = 2;

	//Check each possible partition and find the one with lowest error
	float bestError = FLT_MAX;
	uint32_t bestPartition;
	vec4f bestEndpoints[mode3Partitions * 2];
	uint64_t bestIndices;
	//It might be much better for speed to try to find the best pattern or several patterns first with a basic distance from points to line rather than checking them all.
	for (uint32_t partitionTable = 0; partitionTable < numPartitionTablesPerSubset; partitionTable++) {
		const BC7PartitionTable& table = bc7PartitionTable2Subsets[partitionTable];
		//Original endpoints can be min and max
		vec3f endpoints[mode3Partitions * 2]{
			mins[partitionTable * mode3Partitions + 0],
			maxes[partitionTable * mode3Partitions + 0],
			mins[partitionTable * mode3Partitions + 1],
			maxes[partitionTable * mode3Partitions + 1]
		};
		//I didn't think it was at first, but this is actually required for the degenerate case where the points all lie on opposing diagonals, projecting to the same point on the original diagonal (which results in least squares erroring).
		choose_best_diagonals<mode3Partitions>(pixels, endpoints, table);
		//Mode 1 has 2 partitions
		for (uint32_t partition = 0; partition < mode3Partitions; partition++) {
			//Find indices from the original endpoints and optimize with a least squares iteration to get the new endpoints
			least_squares_optimize_endpoints_rgb<mode3IndexResolution>(pixels, &endpoints[partition * 2], table, partition);
		}

		uint64_t indices;
		float error = quantize_bc7_endpoints<mode3Partitions, mode3ComponentBits, mode3PBitsPerParition, mode3IndexResolution>(pixels, endpoints, table, &indices);

		//Test this partition table's error, see if it's the best one yet
		if (error < bestError) {
			check_flip_indices<vec3f, mode3Partitions, mode3IndexResolution>(endpoints, &indices, partitionTable, table);
			for (uint32_t i = 0; i < (mode3Partitions * 2); i++) {
				bestEndpoints[i] = vec4f{ endpoints[i].x, endpoints[i].y, endpoints[i].z, 0.0F };
			}
			bestIndices = indices;
			bestPartition = partitionTable;
			bestError = error;
		}
	}
	write_bc7_block(block, 3, bestPartition, bestEndpoints, bestIndices);
	return bestError;
}

float compress_bc7_block_mode4(vec4f pixels[16], BC7Block& block, vec3f mins[numPartitionsFor2Subsets], vec3f maxes[numPartitionsFor2Subsets], float minAlphas[numPartitionsFor2Subsets], float maxAlphas[numPartitionsFor2Subsets]) {
	if (!mode4Enable) {
		return FLT_MAX;
	}
	constexpr uint32_t mode4Partitions = 1;
	constexpr uint32_t mode4ComponentBits = 5;
	constexpr uint32_t mode4ComponentAlphaBits = 6;
	constexpr uint32_t mode4PBitsPerParition = 0;
	constexpr uint32_t mode4IndexResolution1 = 2;
	constexpr uint32_t mode4IndexResolution2 = 3;

	vec3f endpointsA[mode4Partitions * 2]{
		std::min(mins[0], mins[1]),
		std::max(maxes[0], maxes[1])
	};
	float endpointAlphasA[mode4Partitions * 2]{
		std::min(minAlphas[0], minAlphas[1]),
		std::max(maxAlphas[0], maxAlphas[1])
	};
	choose_best_diagonals<mode4Partitions>(pixels, endpointsA, bc7DummyPartitionTableAllZeros);
	vec3f endpointsB[mode4Partitions * 2]{ endpointsA[0], endpointsA[1] };
	float endpointAlphasB[mode4Partitions * 2]{ endpointAlphasA[0], endpointAlphasA[1] };

	// Only two variations to try here, I'm not going to turn it into a whole loop.
	uint64_t indicesA;
	uint64_t indicesAlphaA;
	least_squares_optimize_endpoints_rgb<mode4IndexResolution1>(pixels, endpointsA, bc7DummyPartitionTableAllZeros, 0);
	least_squares_optmize_endpoints_alpha<mode4IndexResolution2>(pixels, endpointAlphasA);
	float errorA = quantize_bc7_endpoints<mode4Partitions, mode4ComponentBits, mode4PBitsPerParition, mode4IndexResolution1>(pixels, endpointsA, bc7DummyPartitionTableAllZeros, &indicesA);
	errorA += quantize_bc7_endpoints<mode4Partitions, mode4ComponentAlphaBits, mode4PBitsPerParition, mode4IndexResolution2, float>(pixels, endpointAlphasA, bc7DummyPartitionTableAllZeros, &indicesAlphaA);

	uint64_t indicesB;
	uint64_t indicesAlphaB;
	least_squares_optimize_endpoints_rgb<mode4IndexResolution2>(pixels, endpointsB, bc7DummyPartitionTableAllZeros, 0);
	least_squares_optmize_endpoints_alpha<mode4IndexResolution1>(pixels, endpointAlphasB);
	float errorB = quantize_bc7_endpoints<mode4Partitions, mode4ComponentBits, mode4PBitsPerParition, mode4IndexResolution2>(pixels, endpointsB, bc7DummyPartitionTableAllZeros, &indicesB);
	errorB += quantize_bc7_endpoints<mode4Partitions, mode4ComponentAlphaBits, mode4PBitsPerParition, mode4IndexResolution1, float>(pixels, endpointAlphasB, bc7DummyPartitionTableAllZeros, &indicesAlphaB);

	if (errorA < errorB) {
		check_flip_indices<vec3f, mode4Partitions, mode4IndexResolution1>(endpointsA, &indicesA, 0, bc7DummyPartitionTableAllZeros);
		check_flip_indices<float, mode4Partitions, mode4IndexResolution2>(endpointAlphasA, &indicesAlphaA, 0, bc7DummyPartitionTableAllZeros);
		vec4f bestEndpoints[2]{
			vec4f{ endpointsA[0].x, endpointsA[0].y, endpointsA[0].z, endpointAlphasA[0] },
			vec4f{ endpointsA[1].x, endpointsA[1].y, endpointsA[1].z, endpointAlphasA[1] }
		};
		write_bc7_block(block, 4, 0, bestEndpoints, indicesA, indicesAlphaA, 0, 0);
		return errorA;
	} else {
		check_flip_indices<vec3f, mode4Partitions, mode4IndexResolution2>(endpointsB, &indicesB, 0, bc7DummyPartitionTableAllZeros);
		check_flip_indices<float, mode4Partitions, mode4IndexResolution1>(endpointAlphasB, &indicesAlphaB, 0, bc7DummyPartitionTableAllZeros);
		vec4f bestEndpoints[2]{
			vec4f{ endpointsB[0].x, endpointsB[0].y, endpointsB[0].z, endpointAlphasB[0] },
			vec4f{ endpointsB[1].x, endpointsB[1].y, endpointsB[1].z, endpointAlphasB[1] }
		};
		write_bc7_block(block, 4, 0, bestEndpoints, indicesAlphaB, indicesB, 1, 0);
		return errorB;
	}
}

float compress_bc7_block_mode5(vec4f pixels[16], BC7Block& block, vec3f mins[numPartitionsFor2Subsets], vec3f maxes[numPartitionsFor2Subsets], float minAlphas[numPartitionsFor2Subsets], float maxAlphas[numPartitionsFor2Subsets]) {
	if (!mode5Enable) {
		return FLT_MAX;
	}
	constexpr uint32_t mode5Partitions = 1;
	constexpr uint32_t mode5ComponentBits = 7;
	constexpr uint32_t mode5ComponentAlphaBits = 8;
	constexpr uint32_t mode5PBitsPerParition = 0;
	constexpr uint32_t mode5IndexResolution = 2;

	vec3f bestEndpoints[mode5Partitions * 2]{
		std::min(mins[0], mins[1]),
		std::max(maxes[0], maxes[1])
	};
	float bestEndpointAlphas[mode5Partitions * 2]{
		std::min(minAlphas[0], minAlphas[1]),
		std::max(maxAlphas[0], maxAlphas[1])
	};
	vec3f bestEndpointsRotR[mode5Partitions * 2]{ bestEndpoints[0], bestEndpoints[1] };
	float bestEndpointAlphasRotR[mode5Partitions * 2]{ bestEndpointAlphas[0], bestEndpointAlphas[1] };
	vec3f bestEndpointsRotG[mode5Partitions * 2]{ bestEndpoints[0], bestEndpoints[1] };
	float bestEndpointAlphasRotG[mode5Partitions * 2]{ bestEndpointAlphas[0], bestEndpointAlphas[1] };
	vec3f bestEndpointsRotB[mode5Partitions * 2]{ bestEndpoints[0], bestEndpoints[1] };
	float bestEndpointAlphasRotB[mode5Partitions * 2]{ bestEndpointAlphas[0], bestEndpointAlphas[1] };
	for (uint32_t i = 0; i < (mode5Partitions * 2); i++) {
		std::swap(bestEndpointsRotR[i].x, bestEndpointAlphasRotR[i]);
		std::swap(bestEndpointsRotG[i].y, bestEndpointAlphasRotG[i]);
		std::swap(bestEndpointsRotB[i].z, bestEndpointAlphasRotB[i]);
	}

	choose_best_diagonals<mode5Partitions>(pixels, bestEndpoints, bc7DummyPartitionTableAllZeros);
	
	uint32_t rotationBits = 0;
	uint64_t bestIndices;
	uint64_t bestIndicesAlpha;
	least_squares_optimize_endpoints_rgb<mode5IndexResolution>(pixels, bestEndpoints, bc7DummyPartitionTableAllZeros, 0);
	least_squares_optmize_endpoints_alpha<mode5IndexResolution>(pixels, bestEndpointAlphas);
	float bestError = quantize_bc7_endpoints<mode5Partitions, mode5ComponentBits, mode5PBitsPerParition, mode5IndexResolution>(pixels, bestEndpoints, bc7DummyPartitionTableAllZeros, &bestIndices);
	bestError += quantize_bc7_endpoints<mode5Partitions, mode5ComponentAlphaBits, mode5PBitsPerParition, mode5IndexResolution, float>(pixels, bestEndpointAlphas, bc7DummyPartitionTableAllZeros, &bestIndicesAlpha);

	for (uint32_t i = 0; i < 16; i++) {
		std::swap(pixels[i].x, pixels[i].w);
	}
	choose_best_diagonals<mode5Partitions>(pixels, bestEndpointsRotR, bc7DummyPartitionTableAllZeros);
	uint64_t indices;
	uint64_t indicesAlpha;
	least_squares_optimize_endpoints_rgb<mode5IndexResolution>(pixels, bestEndpointsRotR, bc7DummyPartitionTableAllZeros, 0);
	least_squares_optmize_endpoints_alpha<mode5IndexResolution>(pixels, bestEndpointAlphasRotR);
	float error = quantize_bc7_endpoints<mode5Partitions, mode5ComponentBits, mode5PBitsPerParition, mode5IndexResolution>(pixels, bestEndpointsRotR, bc7DummyPartitionTableAllZeros, &indices);
	error += quantize_bc7_endpoints<mode5Partitions, mode5ComponentAlphaBits, mode5PBitsPerParition, mode5IndexResolution, float>(pixels, bestEndpointAlphasRotR, bc7DummyPartitionTableAllZeros, &indicesAlpha);
	if (error < bestError) {
		bestError = error;
		bestIndices = indices;
		bestIndicesAlpha = indicesAlpha;
		memcpy(bestEndpoints, bestEndpointsRotR, 2 * sizeof(vec3f));
		memcpy(bestEndpointAlphas, bestEndpointAlphasRotR, 2 * sizeof(float));
		rotationBits = 1;
	}

	for (uint32_t i = 0; i < 16; i++) {
		std::swap(pixels[i].x, pixels[i].w);
		std::swap(pixels[i].y, pixels[i].w);
	}
	choose_best_diagonals<mode5Partitions>(pixels, bestEndpointsRotG, bc7DummyPartitionTableAllZeros);
	least_squares_optimize_endpoints_rgb<mode5IndexResolution>(pixels, bestEndpointsRotG, bc7DummyPartitionTableAllZeros, 0);
	least_squares_optmize_endpoints_alpha<mode5IndexResolution>(pixels, bestEndpointAlphasRotG);
	error = quantize_bc7_endpoints<mode5Partitions, mode5ComponentBits, mode5PBitsPerParition, mode5IndexResolution>(pixels, bestEndpointsRotG, bc7DummyPartitionTableAllZeros, &indices);
	error += quantize_bc7_endpoints<mode5Partitions, mode5ComponentAlphaBits, mode5PBitsPerParition, mode5IndexResolution, float>(pixels, bestEndpointAlphasRotG, bc7DummyPartitionTableAllZeros, &indicesAlpha);
	if (error < bestError) {
		bestError = error;
		bestIndices = indices;
		bestIndicesAlpha = indicesAlpha;
		memcpy(bestEndpoints, bestEndpointsRotG, 2 * sizeof(vec3f));
		memcpy(bestEndpointAlphas, bestEndpointAlphasRotG, 2 * sizeof(float));
		rotationBits = 2;
	}

	for (uint32_t i = 0; i < 16; i++) {
		std::swap(pixels[i].y, pixels[i].w);
		std::swap(pixels[i].z, pixels[i].w);
	}
	choose_best_diagonals<mode5Partitions>(pixels, bestEndpointsRotB, bc7DummyPartitionTableAllZeros);
	least_squares_optimize_endpoints_rgb<mode5IndexResolution>(pixels, bestEndpointsRotB, bc7DummyPartitionTableAllZeros, 0);
	least_squares_optmize_endpoints_alpha<mode5IndexResolution>(pixels, bestEndpointAlphasRotB);
	error = quantize_bc7_endpoints<mode5Partitions, mode5ComponentBits, mode5PBitsPerParition, mode5IndexResolution>(pixels, bestEndpointsRotB, bc7DummyPartitionTableAllZeros, &indices);
	error += quantize_bc7_endpoints<mode5Partitions, mode5ComponentAlphaBits, mode5PBitsPerParition, mode5IndexResolution, float>(pixels, bestEndpointAlphasRotB, bc7DummyPartitionTableAllZeros, &indicesAlpha);
	if (error < bestError) {
		bestError = error;
		bestIndices = indices;
		bestIndicesAlpha = indicesAlpha;
		memcpy(bestEndpoints, bestEndpointsRotB, 2 * sizeof(vec3f));
		memcpy(bestEndpointAlphas, bestEndpointAlphasRotB, 2 * sizeof(float));
		rotationBits = 3;
	}

	for (uint32_t i = 0; i < 16; i++) {
		std::swap(pixels[i].z, pixels[i].w);
	}

	check_flip_indices<vec3f, mode5Partitions, mode5IndexResolution>(bestEndpoints, &bestIndices, 0, bc7DummyPartitionTableAllZeros);
	check_flip_indices<float, mode5Partitions, mode5IndexResolution>(bestEndpointAlphas, &bestIndicesAlpha, 0, bc7DummyPartitionTableAllZeros);
	vec4f bestEndpointsRGBA[2]{
		vec4f{ bestEndpoints[0].x, bestEndpoints[0].y, bestEndpoints[0].z, bestEndpointAlphas[0] },
		vec4f{ bestEndpoints[1].x, bestEndpoints[1].y, bestEndpoints[1].z, bestEndpointAlphas[1] }
	};
	write_bc7_block(block, 5, 0, bestEndpointsRGBA, bestIndices, bestIndicesAlpha, 0, rotationBits);
	return bestError;
}

float compress_bc7_block_mode6(vec4f pixels[16], BC7Block& block, vec3f mins[numPartitionsFor2Subsets], vec3f maxes[numPartitionsFor2Subsets], float minAlphas[numPartitionsFor2Subsets], float maxAlphas[numPartitionsFor2Subsets]) {
	if (!mode6Enable) {
		return FLT_MAX;
	}
	constexpr uint32_t mode6Partitions = 1;
	constexpr uint32_t mode6ComponentBits = 7;
	constexpr uint32_t mode6PBitsPerParition = 2;
	constexpr uint32_t mode6IndexResolution = 4;

	vec4f endpoints[mode6Partitions * 2]{
		vec4f{ std::min(mins[0], mins[1]), std::min(minAlphas[0], minAlphas[1]) },
		vec4f{ std::max(maxes[0], maxes[1]), std::max(maxAlphas[0], maxAlphas[1]) }
	};
	choose_best_diagonals_rgba<mode6Partitions>(pixels, endpoints, bc7DummyPartitionTableAllZeros);

	// Only two variations to try here, I'm not going to turn it into a whole loop.
	uint64_t indices;
	least_squares_optimize_endpoints_rgba<mode6IndexResolution>(pixels, endpoints, bc7DummyPartitionTableAllZeros, 0);
	float error = quantize_bc7_endpoints<mode6Partitions, mode6ComponentBits, mode6PBitsPerParition, mode6IndexResolution>(pixels, endpoints, bc7DummyPartitionTableAllZeros, &indices);

	check_flip_indices<vec4f, mode6Partitions, mode6IndexResolution>(endpoints, &indices, 0, bc7DummyPartitionTableAllZeros);
	write_bc7_block(block, 6, 0, endpoints, indices);
	return error;
}

float compress_bc7_block_mode7(vec4f pixels[16], BC7Block& block, vec3f mins[numPartitionsFor2Subsets], vec3f maxes[numPartitionsFor2Subsets], float minAlphas[numPartitionsFor2Subsets], float maxAlphas[numPartitionsFor2Subsets]) {
	if (!mode7Enable) {
		return FLT_MAX;
	}
	constexpr uint32_t mode7Partitions = 2;
	constexpr uint32_t mode7ComponentBits = 5;
	constexpr uint32_t mode7PBitsPerParition = 2;
	constexpr uint32_t mode7IndexResolution = 2;

	//Check each possible partition and find the one with lowest error
	float bestError = FLT_MAX;
	uint32_t bestPartition;
	vec4f bestEndpoints[mode7Partitions * 2];
	uint64_t bestIndices;
	//It might be much better for speed to try to find the best pattern or several patterns first with a basic distance from points to line rather than checking them all.
	for (uint32_t partitionTable = 0; partitionTable < numPartitionTablesPerSubset; partitionTable++) {
		const BC7PartitionTable& table = bc7PartitionTable2Subsets[partitionTable];
		//Original endpoints can be min and max
		vec4f endpoints[mode7Partitions * 2]{
			vec4f{ mins[partitionTable * mode7Partitions + 0], minAlphas[partitionTable * mode7Partitions + 0] },
			vec4f{ maxes[partitionTable * mode7Partitions + 0], maxAlphas[partitionTable * mode7Partitions + 0] },
			vec4f{ mins[partitionTable * mode7Partitions + 1], minAlphas[partitionTable * mode7Partitions + 1] },
			vec4f{ maxes[partitionTable * mode7Partitions + 1], maxAlphas[partitionTable * mode7Partitions + 1] }
		};
		//I didn't think it was at first, but this is actually required for the degenerate case where the points all lie on opposing diagonals, projecting to the same point on the original diagonal (which results in least squares erroring).
		choose_best_diagonals_rgba<mode7Partitions>(pixels, endpoints, table);
		//Mode 1 has 2 partitions
		for (uint32_t partition = 0; partition < mode7Partitions; partition++) {
			//Find indices from the original endpoints and optimize with a least squares iteration to get the new endpoints
			least_squares_optimize_endpoints_rgba<mode7IndexResolution>(pixels, &endpoints[partition * 2], table, partition);
		}

		uint64_t indices;
		float error = quantize_bc7_endpoints<mode7Partitions, mode7ComponentBits, mode7PBitsPerParition, mode7IndexResolution>(pixels, endpoints, table, &indices);

		//Test this partition table's error, see if it's the best one yet
		if (error < bestError) {
			check_flip_indices<vec4f, mode7Partitions, mode7IndexResolution>(endpoints, &indices, partitionTable, table);
			memcpy(bestEndpoints, endpoints, mode7Partitions * 2 * sizeof(vec4f));
			bestIndices = indices;
			bestPartition = partitionTable;
			bestError = error;
		}
	}
	write_bc7_block(block, 7, bestPartition, bestEndpoints, bestIndices);
	return bestError;
}

void calc_min_max_mean(const BC7PartitionTable& partitionTable, vec4f fPixels[16], vec3f min[3], vec3f max[3], vec3f mean[3]) {
	for (uint32_t i = 0; i < 3; i++) {
		min[i] = { FLT_MAX, FLT_MAX, FLT_MAX };
		max[i] = { -FLT_MAX, -FLT_MAX , -FLT_MAX };
		mean[i] = { 0.0F, 0.0F, 0.0F };
	}

	float meanCounts[3]{ 0.0F, 0.0F, 0.0F };
	for (uint32_t i = 0; i < 16; i++) {
		uint32_t partition = partitionTable.partitionNumbers[i];
		min[partition] = std::min(min[partition], fPixels[i].xyz());
		max[partition] = std::max(max[partition], fPixels[i].xyz());
		mean[partition] += fPixels[i].xyz();
		meanCounts[partition] += 1.0F;
	}

	for (uint32_t i = 0; i < 3; i++) {
		mean[i] /= meanCounts[i];
	}
}

void calc_min_max_f(const BC7PartitionTable& partitionTable, vec4f fPixels[16], float mins[2], float maxes[2]) {
	for (uint32_t i = 0; i < 2; i++) {
		mins[i] = FLT_MAX;
		maxes[i] = -FLT_MAX;
	}
	for (uint32_t i = 0; i < 16; i++) {
		uint32_t partition = partitionTable.partitionNumbers[i];
		mins[partition] = std::min(mins[partition], fPixels[i].w);
		maxes[partition] = std::max(maxes[partition], fPixels[i].w);
	}
}

void calc_min_max_fx8(const BC7PartitionTable& partitionTable, vec4fx8 pixels[16], floatx8 mins[2], floatx8 maxes[2]) {
	for (uint32_t i = 0; i < 2; i++) {
		mins[i] = _mm256_set1_ps(FLT_MAX);
		maxes[i] = _mm256_set1_ps(-FLT_MAX);
	}
	for (uint32_t i = 0; i < 16; i++) {
		uint32_t partition = partitionTable.partitionNumbers[i];
		mins[partition] = _mm256_min_ps(mins[partition], pixels[i].w);
		maxes[partition] = _mm256_max_ps(maxes[partition], pixels[i].w);
	}
}

void calc_min_maxx8(const BC7PartitionTable& partitionTable, vec4fx8 fPixels[16], vec3fx8 mins[3], vec3fx8 maxs[3]) {
	for (uint32_t i = 0; i < 3; i++) {
		mins[i].set(FLT_MAX);
		maxs[i].set(-FLT_MAX);
	}
	for (uint32_t i = 0; i < 16; i++) {
		uint32_t partition = partitionTable.partitionNumbers[i];
		vec3fx8 rgb = fPixels[i].xyz();
		mins[partition] = min(mins[partition], rgb);
		maxs[partition] = max(maxs[partition], rgb);
	}
}

void blend_bc7blockx8(BC7Blockx4 dst[2], BC7Blockx4 src[2], floatx8 mask) {
	// Extract the low and high halves of the mask and extend them to 64 bits
	floatx8 maskLow = _mm256_castsi256_ps(_mm256_cvtepi32_epi64(_mm256_castsi256_si128(_mm256_castps_si256(mask))));
	floatx8 maskHigh = _mm256_castsi256_ps(_mm256_cvtepi32_epi64(_mm256_extracti128_si256(_mm256_castps_si256(mask), 1)));
	// dst[0] = blend(dst[0], src[0], maskLow)
	dst[0].data[0] = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(dst[0].data[0]), _mm256_castsi256_ps(src[0].data[0]), maskLow));
	dst[0].data[1] = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(dst[0].data[1]), _mm256_castsi256_ps(src[0].data[1]), maskLow));
	// dst[1] = blend(dst[1], src[1], maskHigh)
	dst[1].data[0] = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(dst[1].data[0]), _mm256_castsi256_ps(src[1].data[0]), maskHigh));
	dst[1].data[1] = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(dst[1].data[1]), _mm256_castsi256_ps(src[1].data[1]), maskHigh));
}

float compress_bc7_blockx8(vec4fx8 pixels[16], BC7Blockx4 blocks[2]) {
	vec3fx8 mins[totalNumPartitions];
	vec3fx8 maxes[totalNumPartitions];
	floatx8 alphaMins[numPartitionsFor2Subsets];
	floatx8 alphaMaxes[numPartitionsFor2Subsets];
	for (uint32_t i = 0; i < numPartitionTablesPerSubset; i++) {
		calc_min_maxx8(bc7PartitionTable2Subsets[i], pixels, &mins[i * 2], &maxes[i * 2]);
	}
	//Calculate min/max for 3 subsets
	for (uint32_t i = 0; i < numPartitionTablesPerSubset; i++) {
		calc_min_maxx8(bc7PartitionTable3Subsets[i], pixels, &mins[numPartitionsFor2Subsets + i * 3], &maxes[numPartitionsFor2Subsets + i * 3]);
	}
	for (uint32_t i = 0; i < numPartitionTablesPerSubset; i++) {
		calc_min_max_fx8(bc7PartitionTable2Subsets[i], pixels, &alphaMins[i * 2], &alphaMaxes[i * 2]);
	}

	BC7Blockx4 testBlocks[2];
	floatx8 bestError = compress_bc7_block_mode01237x8<0, vec3fx8>(pixels, blocks, mins + numPartitionsFor2Subsets, maxes + numPartitionsFor2Subsets, alphaMins, alphaMaxes);

	floatx8 error = compress_bc7_block_mode01237x8<1, vec3fx8>(pixels, testBlocks, mins, maxes, alphaMins, alphaMaxes);
	blend_bc7blockx8(blocks, testBlocks, _mm256_cmp_ps(error, bestError, _CMP_LT_OQ));
	bestError = _mm256_min_ps(error, bestError);

	error = compress_bc7_block_mode01237x8<2, vec3fx8>(pixels, testBlocks, mins + numPartitionsFor2Subsets, maxes + numPartitionsFor2Subsets, alphaMins, alphaMaxes);
	blend_bc7blockx8(blocks, testBlocks, _mm256_cmp_ps(error, bestError, _CMP_LT_OQ));
	bestError = _mm256_min_ps(error, bestError);

	error = compress_bc7_block_mode01237x8<3, vec3fx8>(pixels, testBlocks, mins, maxes, alphaMins, alphaMaxes);
	blend_bc7blockx8(blocks, testBlocks, _mm256_cmp_ps(error, bestError, _CMP_LT_OQ));
	bestError = _mm256_min_ps(error, bestError);

	error = compress_bc7_block_mode4x8(pixels, testBlocks, mins, maxes, alphaMins, alphaMaxes);
	blend_bc7blockx8(blocks, testBlocks, _mm256_cmp_ps(error, bestError, _CMP_LT_OQ));
	bestError = _mm256_min_ps(error, bestError);

	error = compress_bc7_block_mode5x8(pixels, testBlocks, mins, maxes, alphaMins, alphaMaxes);
	blend_bc7blockx8(blocks, testBlocks, _mm256_cmp_ps(error, bestError, _CMP_LT_OQ));
	bestError = _mm256_min_ps(error, bestError);

	error = compress_bc7_block_mode6x8(pixels, testBlocks, mins, maxes, alphaMins, alphaMaxes);
	blend_bc7blockx8(blocks, testBlocks, _mm256_cmp_ps(error, bestError, _CMP_LT_OQ));
	bestError = _mm256_min_ps(error, bestError);

	error = compress_bc7_block_mode01237x8<7, vec4fx8>(pixels, testBlocks, mins, maxes, alphaMins, alphaMaxes);
	blend_bc7blockx8(blocks, testBlocks, _mm256_cmp_ps(error, bestError, _CMP_LT_OQ));
	bestError = _mm256_min_ps(error, bestError);

	blocks[0] = testBlocks[0];
	blocks[1] = testBlocks[1];
	return horizontal_sum(bestError);
}

float compress_bc7_block(RGBA pixels[16], BC7Block& block) {
	vec4f fPixels[16];
	for (uint32_t i = 0; i < 16; i++) {
		fPixels[i] = vec4f{ static_cast<float>(pixels[i].r) / 255.0F, static_cast<float>(pixels[i].g) / 255.0F, static_cast<float>(pixels[i].b) / 255.0F, static_cast<float>(pixels[i].a) / 255.0F };
	}

	//We're going to precalculate all the min/max/mean values for each subset so we don't have to do recomputation for each mode
	vec3f min[totalNumPartitions];
	vec3f max[totalNumPartitions];
	vec3f mean[totalNumPartitions];

	float alphaMins[numPartitionsFor2Subsets];
	float alphaMaxes[numPartitionsFor2Subsets];

	//Calculate min/max/mean for 2 subsets
	for (uint32_t i = 0; i < numPartitionTablesPerSubset; i++) {
		calc_min_max_mean(bc7PartitionTable2Subsets[i], fPixels, &min[i * 2], &max[i * 2], &mean[i * 2]);
	}
	//Calculate min/max/mean for 3 subsets
	for (uint32_t i = 0; i < numPartitionTablesPerSubset; i++) {
		calc_min_max_mean(bc7PartitionTable3Subsets[i], fPixels, &min[numPartitionsFor2Subsets + i * 3], &max[numPartitionsFor2Subsets + i * 3], &mean[numPartitionsFor2Subsets + i * 3]);
	}
	for (uint32_t i = 0; i < numPartitionTablesPerSubset; i++) {
		calc_min_max_f(bc7PartitionTable2Subsets[i], fPixels, &alphaMins[i * 2], &alphaMaxes[i * 2]);
	}

	BC7Block testBlock;
	float bestError;
	float error = compress_bc7_block_mode0(fPixels, testBlock, min + numPartitionsFor2Subsets, max + numPartitionsFor2Subsets, mean + numPartitionsFor2Subsets);
	bestError = error;
	block = testBlock;
	error = compress_bc7_block_mode1(fPixels, testBlock, min, max, mean);
	if (error < bestError) {
		bestError = error;
		block = testBlock;
	}
	error = compress_bc7_block_mode2(fPixels, testBlock, min + numPartitionsFor2Subsets, max + numPartitionsFor2Subsets, mean + numPartitionsFor2Subsets);
	if (error < bestError) {
		bestError = error;
		block = testBlock;
	}
	error = compress_bc7_block_mode3(fPixels, testBlock, min, max, mean);
	if (error < bestError) {
		bestError = error;
		block = testBlock;
	}
	error = compress_bc7_block_mode4(fPixels, testBlock, min, max, alphaMins, alphaMaxes);
	if (error < bestError) {
		bestError = error;
		block = testBlock;
	}
	error = compress_bc7_block_mode5(fPixels, testBlock, min, max, alphaMins, alphaMaxes);
	if (error < bestError) {
		bestError = error;
		block = testBlock;
	}
	error = compress_bc7_block_mode6(fPixels, testBlock, min, max, alphaMins, alphaMaxes);
	if (error < bestError) {
		bestError = error;
		block = testBlock;
	}
	error = compress_bc7_block_mode7(fPixels, testBlock, min, max, alphaMins, alphaMaxes);
	if (error < bestError) {
		bestError = error;
		block = testBlock;
	}
	return bestError;
}

#define BC7_ENABLE_AVX2 1

//#include <stdlib.h>
#include <chrono>

struct BC7AVXBlockRange {
	uint32_t begin;
	uint32_t end;
	float* finalError;
	vec4fx8* pixelBlocks;
	BC7Block* blocks;
	uint32_t* blockIndices;
	uint32_t opaqueIndex;
};

void bc7_avx_compress_block_range(BC7AVXBlockRange* range) {
	//std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
	uint32_t begin = range->begin;
	uint32_t end = range->end;
	BC7Block* blocks = range->blocks;
	uint32_t* blockIndices = range->blockIndices;
	vec4fx8* pixelBlocks = range->pixelBlocks;
	uint32_t opaqueIndex = range->opaqueIndex;
	float finalError = 0.0F;
	//std::cout << "start\n";
	for (uint32_t i = begin; i < end; i++) {
		//std::cout << "Compressing opaque index: " << i << '\n';
		BC7Blockx4 finalBlocks[2];
		alignas(32) uint64_t data0[8];
		alignas(32) uint64_t data1[8];
		finalError += compress_bc7_blockx8(&pixelBlocks[i * 16], finalBlocks);
		_mm256_store_si256(reinterpret_cast<uint64x4*>(data0), finalBlocks[0].data[0]);
		_mm256_store_si256(reinterpret_cast<uint64x4*>(data0 + 4), finalBlocks[1].data[0]);
		_mm256_store_si256(reinterpret_cast<uint64x4*>(data1), finalBlocks[0].data[1]);
		_mm256_store_si256(reinterpret_cast<uint64x4*>(data1 + 4), finalBlocks[1].data[1]);
		for (uint32_t j = 0; j < std::min(opaqueIndex - i * 8, 8ui32); j++) {
			blocks[blockIndices[i * 8 + j]] = BC7Block{ data0[j], data1[j] };
		}
	}
	*range->finalError += finalError;

	//std::chrono::duration d = std::chrono::high_resolution_clock::now() - t1;
	//std::cout << "Time taken thread " << d.count() << '\n';
}

//Compresion//
BC7Block* compress_bc7(RGBA* image, uint32_t width, uint32_t height, job::JobSystem& jobSystem) {
	uint32_t blockWidth = (width + 3) / 4;
	uint32_t blockHeight = (height + 3) / 4;
	uint32_t numBlocks = blockWidth * blockHeight;
	BC7Block* blocks = reinterpret_cast<BC7Block*>(malloc(numBlocks * sizeof(BC7Block)));
	if (!blocks) {
		return nullptr;
	}

	float finalError = 0;

#if BC7_ENABLE_AVX2 == 0
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
	//4x4 block
	RGBA pixels[4 * 4];
	for (uint32_t y = 0; y < blockHeight; y++) {
		for (uint32_t x = 0; x < blockWidth; x++) {
			//std::cout << "Compressing at " << x << ' ' << y << '\n';
			fill_pixel_block(image, pixels, x, y, width, height);
			BC7Block& block = blocks[y * blockWidth + x];
			finalError += compress_bc7_block(pixels, block);
		}
	}
	std::chrono::duration d = std::chrono::high_resolution_clock::now() - t1;
	std::cout << "Time taken " << d.count() << '\n';
#else
	uint32_t simdBlockCount = (numBlocks + 7) / 8 + 1;
	// YMM align to 32
	// TODO Move this to an actual aligned malloc method
	uint64_t alignment = 32;
	void* pixelBlockMemory = malloc(16 * simdBlockCount * sizeof(vec4fx8) + alignment);
	vec4fx8* pixelBlocks = reinterpret_cast<vec4fx8*>((reinterpret_cast<uintptr_t>(pixelBlockMemory) + alignment - 1) & ~(alignment - 1));
	uint32_t* blockIndices = reinterpret_cast<uint32_t*>(malloc(numBlocks * sizeof(uint32_t)));

	//uint64_t time = __rdtsc();
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

	uint32_t opaqueIndex = 0;
	uint32_t transparentIndex = simdBlockCount - 1;
	for (uint32_t y = 0; y < blockHeight; y++) {
		for (uint32_t x = 0; x < blockWidth; x++) {
			//std::cout << "Loading block at " << x << ' ' << y << " opaque index: " << opaqueIndex << '\n';
			// Oh no, big bad strict aliasing. Whatever, I'll deal with it if my compile actually breaks it. For now, this makes it quite a lot more convenient.
			fill_pixel_blockx8(image, reinterpret_cast<float*>(pixelBlocks), opaqueIndex, x, y, width, height);
			blockIndices[opaqueIndex] = y * blockWidth + x;
			opaqueIndex++;
		}
	}
	
	uint32_t numOpaqueBlocksx8 = (opaqueIndex + 7) / 8;
	uint32_t jobCount = jobSystem.thread_count() * 2;
	BC7AVXBlockRange* ranges = reinterpret_cast<BC7AVXBlockRange*>(alloca(jobCount * sizeof(BC7AVXBlockRange)));
	job::JobDecl* jobs = reinterpret_cast<job::JobDecl*>(alloca(jobCount * sizeof(job::JobDecl)));
	uint32_t step = numOpaqueBlocksx8 / jobCount;
	uint32_t startIdx = 0;
	for (uint32_t i = 0; i < jobCount; i++) {
		ranges[i].begin = startIdx;
		ranges[i].end = startIdx + step;
		startIdx += step;
		ranges[i].finalError = &finalError;
		ranges[i].pixelBlocks = pixelBlocks;
		ranges[i].blocks = blocks;
		ranges[i].blockIndices = blockIndices;
		ranges[i].opaqueIndex = opaqueIndex;

		new (&jobs[i]) job::JobDecl{ bc7_avx_compress_block_range, &ranges[i]};
	}
	ranges[jobCount - 1].end = numOpaqueBlocksx8;
	jobSystem.start_jobs_and_wait_for_counter(jobs, jobCount);

	//std::cout << "Time taken " << (__rdtsc() - time) << '\n';
	std::chrono::duration d = std::chrono::high_resolution_clock::now() - t1;
	std::cout << "Time taken " << d.count() << '\n';

	free(pixelBlockMemory);
	free(blockIndices);
#endif
	// Slightly inaccurate error since it doesn't account for the padding pixels for non multiple of 4 textures and I don't feel like implementing that
	float error = finalError / static_cast<float>(numBlocks * 16 * 4);
	std::cout << "Final Error: " << error << '\n';
	std::cout << "PSNR: " << (10.0F * log10(1.0F / error)) << '\n';
	return blocks;
	}

//Decompression//

uint32_t extract_mode(BC7Block& block) {
	//Possibly use this?
	//return _tzcnt_u64(block.data[0]);
	uint64_t data0 = block.data[0];
	uint32_t mode = 0;
	while ((data0 & 1) == 0) {
		data0 >>= 1;
		mode++;
	}
	return mode;
}

//TODO Most of this mode code seems to be the same, refactor
void decompress_bc7_mode0(BC7Block& block, RGBA pixels[16]) {
	//4 bits of partition selection
	uint32_t partitionSelection = (block.data[0] >> 1) & 0b1111;
	//3 partitions
	const BC7PartitionTable& partition = bc7PartitionTable3Subsets[partitionSelection];
	uint8_t anchor2nd = bc7PartitionTable3Anchors2ndSubset[partitionSelection];
	uint8_t anchor3rd = bc7PartitionTable3Anchors3rdSubset[partitionSelection];

	uint32_t endpointReds = block.data[0] >> 5;
	uint32_t endpointGreens = block.data[0] >> 29;
	uint32_t endpointBlues = (block.data[0] >> 53) | (block.data[1] << (64 - 53));
	uint32_t endpointPBits = block.data[1] >> 13;
	//Mode 0 has 3 endpoints
	RGB endpoints[6];
	for (uint32_t i = 0; i < 6; i++) {
		uint32_t pBit = ((endpointPBits >> i) & 1) << 3;
		//4 bit colors
		endpoints[i].r = (endpointReds >> (i * 4)) & 0b1111;
		endpoints[i].r = (endpoints[i].r << 4) | pBit | (endpoints[i].r >> 1);
		endpoints[i].g = (endpointGreens >> (i * 4)) & 0b1111;
		endpoints[i].g = (endpoints[i].g << 4) | pBit | (endpoints[i].g >> 1);
		endpoints[i].b = (endpointBlues >> (i * 4)) & 0b1111;
		endpoints[i].b = (endpoints[i].b << 4) | pBit | (endpoints[i].b >> 1);
	}

	uint64_t indices = block.data[1] >> 19;
	//Add the upper 0 bits to the anchors to make it easier to decode
	/*uint64_t anchor2ndMask = (1 << (anchor2nd * 4)) - 1;
	uint64_t anchor3rdMask = (1 << (anchor3rd * 4)) - 1;
	uint64_t anchorMasks[2];
	anchorMasks[0] = (1 << (anchor2nd * 4)) - 1;
	anchorMasks[1] = (1 << (anchor3rd * 4)) - 1;
	if (anchor2nd < anchor3rd) {
		//Make sure they're in order from high to low. That way I don't have to change any anchor bit indices.
		std::swap(anchorMasks[0], anchorMasks[1]);
	}
	indices = ((indices & ~anchorMasks[0]) << 1) | (indices & anchorMasks[0]);
	indices = ((indices & ~anchorMasks[1]) << 1) | (indices & anchorMasks[1]);
	indices = ((indices & ~1ULL) << 1) | (indices & 1ULL);*/

	//Decode pixels
	uint32_t shift = 0;
	for (uint32_t i = 0; i < 16; i++) {
		//Perhaps slightly better code than the stuff commented out above
		//3 bit indices
		uint32_t index;
		/*
		uint32_t isNotAnchor = !((anchor2nd == i) | (anchor3rd == i) | (0 == i));
		index = (indices >> shift) & ((0b11 << isNotAnchor) | isNotAnchor);
		shift += 2 + isNotAnchor;
		*/
		if ((anchor2nd == i) | (anchor3rd == i) | (0 == i)) {
			index = (indices >> shift) & 0b11;
			shift += 2;
		} else {
			index = (indices >> shift) & 0b111;
			shift += 3;
		}

		uint32_t interpolationFactor = bc7InterpolationFactors3[index];
		uint8_t partitionNumber = partition.partitionNumbers[i];
		RGB e0 = endpoints[partitionNumber * 2];
		RGB e1 = endpoints[partitionNumber * 2 + 1];
		pixels[i].r = bc7_interpolate(e0.r, e1.r, interpolationFactor);
		pixels[i].g = bc7_interpolate(e0.g, e1.g, interpolationFactor);
		pixels[i].b = bc7_interpolate(e0.b, e1.b, interpolationFactor);
		pixels[i].a = 255;
	}
}

void decompress_bc7_mode1(BC7Block& block, RGBA pixels[16]) {
	//6 bits of partition selection
	uint32_t partitionSelection = (block.data[0] >> 2) & 0b111111;
	//2 partitions
	const BC7PartitionTable& partition = bc7PartitionTable2Subsets[partitionSelection];
	uint8_t anchor2nd = bc7PartitionTable2Anchors2ndSubset[partitionSelection];

	uint32_t endpointReds = block.data[0] >> 8;
	uint32_t endpointGreens = block.data[0] >> 32;
	uint32_t endpointBlues = (block.data[0] >> 56) | (block.data[1] << (64 - 56));
	uint32_t endpointPBits = block.data[1] >> 16;
	//Mode 1 has 2 endpoints
	RGB endpoints[4];
	for (uint32_t i = 0; i < 4; i++) {
		uint32_t pBit = ((endpointPBits >> (i >> 1)) & 1) << 1;
		//6 bit colors
		endpoints[i].r = (endpointReds >> (i * 6)) & 0b111111;
		endpoints[i].r = (endpoints[i].r << 2) | pBit | (endpoints[i].r >> 5);
		endpoints[i].g = (endpointGreens >> (i * 6)) & 0b111111;
		endpoints[i].g = (endpoints[i].g << 2) | pBit | (endpoints[i].g >> 5);
		endpoints[i].b = (endpointBlues >> (i * 6)) & 0b111111;
		endpoints[i].b = (endpoints[i].b << 2) | pBit | (endpoints[i].b >> 5);
	}

	uint64_t indices = block.data[1] >> 18;

	//Decode pixels
	uint32_t shift = 0;
	for (uint32_t i = 0; i < 16; i++) {
		//3 bit indices
		uint32_t index;
		if ((anchor2nd == i) | (0 == i)) {
			index = (indices >> shift) & 0b11;
			shift += 2;
		} else {
			index = (indices >> shift) & 0b111;
			shift += 3;
		}

		uint32_t interpolationFactor = bc7InterpolationFactors3[index];
		uint8_t partitionNumber = partition.partitionNumbers[i];
		RGB e0 = endpoints[partitionNumber * 2];
		RGB e1 = endpoints[partitionNumber * 2 + 1];
		pixels[i].r = bc7_interpolate(e0.r, e1.r, interpolationFactor);
		pixels[i].g = bc7_interpolate(e0.g, e1.g, interpolationFactor);
		pixels[i].b = bc7_interpolate(e0.b, e1.b, interpolationFactor);
		pixels[i].a = 255;
	}
}

void decompress_bc7_mode2(BC7Block& block, RGBA pixels[16]) {
	//6 bits of partition selection
	uint32_t partitionSelection = (block.data[0] >> 3) & 0b111111;
	//3 partitions
	const BC7PartitionTable& partition = bc7PartitionTable3Subsets[partitionSelection];
	uint8_t anchor2nd = bc7PartitionTable3Anchors2ndSubset[partitionSelection];
	uint8_t anchor3rd = bc7PartitionTable3Anchors3rdSubset[partitionSelection];

	uint32_t endpointReds = block.data[0] >> 9;
	uint32_t endpointGreens = (block.data[0] >> 39) | (block.data[1] << (64 - 39));
	uint32_t endpointBlues = block.data[1] >> 5;
	//Mode 2 has 3 endpoints
	RGB endpoints[6];
	for (uint32_t i = 0; i < 6; i++) {
		//5 bit colors
		endpoints[i].r = (endpointReds >> (i * 5)) & 0b11111;
		endpoints[i].r = (endpoints[i].r << 3) | (endpoints[i].r >> 2);
		endpoints[i].g = (endpointGreens >> (i * 5)) & 0b11111;
		endpoints[i].g = (endpoints[i].g << 3) | (endpoints[i].g >> 2);
		endpoints[i].b = (endpointBlues >> (i * 5)) & 0b11111;
		endpoints[i].b = (endpoints[i].b << 3) | (endpoints[i].b >> 2);
	}

	uint64_t indices = block.data[1] >> 35;

	//Decode pixels
	uint32_t shift = 0;
	for (uint32_t i = 0; i < 16; i++) {
		//2 bit indices
		uint32_t index;
		if ((anchor2nd == i) | (anchor3rd == i) | (0 == i)) {
			index = (indices >> shift) & 0b1;
			shift += 1;
		} else {
			index = (indices >> shift) & 0b11;
			shift += 2;
		}

		uint32_t interpolationFactor = bc7InterpolationFactors2[index];
		uint8_t partitionNumber = partition.partitionNumbers[i];
		RGB e0 = endpoints[partitionNumber * 2];
		RGB e1 = endpoints[partitionNumber * 2 + 1];
		pixels[i].r = bc7_interpolate(e0.r, e1.r, interpolationFactor);
		pixels[i].g = bc7_interpolate(e0.g, e1.g, interpolationFactor);
		pixels[i].b = bc7_interpolate(e0.b, e1.b, interpolationFactor);
		pixels[i].a = 255;
	}
}

void decompress_bc7_mode3(BC7Block& block, RGBA pixels[16]) {
	//6 bits of partition selection
	uint32_t partitionSelection = (block.data[0] >> 4) & 0b111111;
	//2 partitions
	const BC7PartitionTable& partition = bc7PartitionTable2Subsets[partitionSelection];
	uint8_t anchor2nd = bc7PartitionTable2Anchors2ndSubset[partitionSelection];

	uint32_t endpointReds = block.data[0] >> 10;
	uint32_t endpointGreens = (block.data[0] >> 38) | (block.data[1] << (64 - 38));
	uint32_t endpointBlues = block.data[1] >> 2;
	uint32_t endpointPBits = block.data[1] >> 30;
	//Mode 3 has 2 endpoints
	RGB endpoints[4];
	for (uint32_t i = 0; i < 4; i++) {
		uint32_t pBit = (endpointPBits >> i) & 1;
		//7 bit colors
		endpoints[i].r = (endpointReds >> (i * 7)) & 0b1111111;
		endpoints[i].r = (endpoints[i].r << 1) | pBit;
		endpoints[i].g = (endpointGreens >> (i * 7)) & 0b1111111;
		endpoints[i].g = (endpoints[i].g << 1) | pBit;
		endpoints[i].b = (endpointBlues >> (i * 7)) & 0b1111111;
		endpoints[i].b = (endpoints[i].b << 1) | pBit;
	}

	uint64_t indices = block.data[1] >> 34;

	//Decode pixels
	uint32_t shift = 0;
	for (uint32_t i = 0; i < 16; i++) {
		//2 bit indices
		uint32_t index;
		if ((anchor2nd == i) | (0 == i)) {
			index = (indices >> shift) & 0b1;
			shift += 1;
		} else {
			index = (indices >> shift) & 0b11;
			shift += 2;
		}

		uint32_t interpolationFactor = bc7InterpolationFactors2[index];
		uint8_t partitionNumber = partition.partitionNumbers[i];
		RGB e0 = endpoints[partitionNumber * 2];
		RGB e1 = endpoints[partitionNumber * 2 + 1];
		pixels[i].r = bc7_interpolate(e0.r, e1.r, interpolationFactor);
		pixels[i].g = bc7_interpolate(e0.g, e1.g, interpolationFactor);
		pixels[i].b = bc7_interpolate(e0.b, e1.b, interpolationFactor);
		pixels[i].a = 255;
	}
}

void rotate(RGBA& pixel, uint32_t rotationBits) {
	switch (rotationBits) {
	case 0b00: break;
	case 0b01: std::swap(pixel.r, pixel.a); break;
	case 0b10: std::swap(pixel.g, pixel.a); break;
	case 0b11: std::swap(pixel.b, pixel.a); break;
	}
}

void decompress_bc7_mode4(BC7Block& block, RGBA pixels[16]) {
	uint32_t rotationBits = (block.data[0] >> 5) & 0b11;
	uint32_t indexSelection = (block.data[0] >> 7) & 1;

	RGBA endpoints[2];
	for (uint32_t i = 0; i < 2; i++) {
		//5 bit color
		uint32_t r = (block.data[0] >> (8 + i * 5)) & 0b11111;
		uint32_t g = (block.data[0] >> (18 + i * 5)) & 0b11111;
		uint32_t b = (block.data[0] >> (28 + i * 5)) & 0b11111;
		//6 bit alpha
		uint32_t a = (block.data[0] >> (38 + i * 6)) & 0b111111;

		endpoints[i].r = (r << 3) | (r >> 2);
		endpoints[i].g = (g << 3) | (g >> 2);
		endpoints[i].b = (b << 3) | (b >> 2);
		endpoints[i].a = (a << 2) | (a >> 4);
	}

	uint64_t indicesPrimary = (block.data[0] >> 50) | (block.data[1] << (64 - 50));
	uint64_t indicesSecondary = block.data[1] >> 17;
	uint32_t shiftPrimary = 0;
	uint32_t shiftSecondary = 0;
	for (uint32_t i = 0; i < 16; i++) {
		//2 and 3 bit indices
		uint32_t indexPrimary;
		uint32_t indexSecondary;
		if (0 == i) {
			indexPrimary = (indicesPrimary >> shiftPrimary) & 0b1;
			indexSecondary = (indicesSecondary >> shiftSecondary) & 0b11;
			shiftPrimary += 1;
			shiftSecondary += 2;
		} else {
			indexPrimary = (indicesPrimary >> shiftPrimary) & 0b11;
			indexSecondary = (indicesSecondary >> shiftSecondary) & 0b111;
			shiftPrimary += 2;
			shiftSecondary += 3;
		}

		uint32_t interpolationFactorPrimary = bc7InterpolationFactors2[indexPrimary];
		uint32_t interpolationFactorSecondary = bc7InterpolationFactors3[indexSecondary];
		RGBA e0 = endpoints[0];
		RGBA e1 = endpoints[1];
		if (indexSelection) {
			//Use secondary for color
			pixels[i].r = bc7_interpolate(e0.r, e1.r, interpolationFactorSecondary);
			pixels[i].g = bc7_interpolate(e0.g, e1.g, interpolationFactorSecondary);
			pixels[i].b = bc7_interpolate(e0.b, e1.b, interpolationFactorSecondary);
			pixels[i].a = bc7_interpolate(e0.a, e1.a, interpolationFactorPrimary);
		} else {
			//Use primary for color
			pixels[i].r = bc7_interpolate(e0.r, e1.r, interpolationFactorPrimary);
			pixels[i].g = bc7_interpolate(e0.g, e1.g, interpolationFactorPrimary);
			pixels[i].b = bc7_interpolate(e0.b, e1.b, interpolationFactorPrimary);
			pixels[i].a = bc7_interpolate(e0.a, e1.a, interpolationFactorSecondary);
		}
		rotate(pixels[i], rotationBits);
	}
}

void decompress_bc7_mode5(BC7Block& block, RGBA pixels[16]) {
	uint32_t rotationBits = (block.data[0] >> 6) & 0b11;
	uint32_t indexSelection = (block.data[0] >> 7) & 1;

	RGBA endpoints[2];
	//8 bit alpha, can extract directly
	endpoints[0].a = (block.data[0] >> 50) & 0xFF;
	endpoints[1].a = ((block.data[0] >> 58) | (block.data[1] << (64 - 58))) & 0xFF;
	for (uint32_t i = 0; i < 2; i++) {
		//7 bit color
		uint32_t r = (block.data[0] >> (8 + i * 7)) & 0b1111111;
		uint32_t g = (block.data[0] >> (22 + i * 7)) & 0b1111111;
		//Ah man, spotted an error in the specification here (listed 35 instead of 36). Turns out I was using an outdated spec version and it was fixed later. I hope I didn't embed any other old spec errors in my code.
		uint32_t b = (block.data[0] >> (36 + i * 7)) & 0b1111111;

		endpoints[i].r = (r << 1) | (r >> 6);	
		endpoints[i].g = (g << 1) | (g >> 6);
		endpoints[i].b = (b << 1) | (b >> 6);
	}

	uint64_t indicesPrimary = block.data[1] >> 2;
	uint64_t indicesSecondary = block.data[1] >> 33;
	uint32_t shift = 0;
	for (uint32_t i = 0; i < 16; i++) {
		//2 bit indices
		uint32_t indexPrimary;
		uint32_t indexSecondary;
		if (0 == i) {
			indexPrimary = (indicesPrimary >> shift) & 0b1;
			indexSecondary = (indicesSecondary >> shift) & 0b1;
			shift += 1;
		} else {
			indexPrimary = (indicesPrimary >> shift) & 0b11;
			indexSecondary = (indicesSecondary >> shift) & 0b11;
			shift += 2;
		}

		uint32_t interpolationFactorPrimary = bc7InterpolationFactors2[indexPrimary];
		uint32_t interpolationFactorSecondary = bc7InterpolationFactors2[indexSecondary];
		RGBA e0 = endpoints[0];
		RGBA e1 = endpoints[1];
		pixels[i].r = bc7_interpolate(e0.r, e1.r, interpolationFactorPrimary);
		pixels[i].g = bc7_interpolate(e0.g, e1.g, interpolationFactorPrimary);
		pixels[i].b = bc7_interpolate(e0.b, e1.b, interpolationFactorPrimary);
		pixels[i].a = bc7_interpolate(e0.a, e1.a, interpolationFactorSecondary);
		rotate(pixels[i], rotationBits);
	}
}

void decompress_bc7_mode6(BC7Block& block, RGBA pixels[16]) {
	uint32_t pBits = (block.data[0] >> 63) | (block.data[1] << 1);
	//Mode 6 has 1 endpoint
	RGBA endpoints[2];
	for (uint32_t i = 0; i < 2; i++) {
		uint32_t pBit = (pBits >> i) & 1;
		//7 bit color
		uint32_t r = (block.data[0] >> (7 + i * 7)) & 0b1111111;
		uint32_t g = (block.data[0] >> (21 + i * 7)) & 0b1111111;
		uint32_t b = (block.data[0] >> (35 + i * 7)) & 0b1111111;
		uint32_t a = (block.data[0] >> (49 + i * 7)) & 0b1111111;

		endpoints[i].r = (r << 1) | pBit;
		endpoints[i].g = (g << 1) | pBit;
		endpoints[i].b = (b << 1) | pBit;
		endpoints[i].a = (a << 1) | pBit;
	}

	uint64_t indices = block.data[1] >> 1;
	uint32_t shift = 0;
	for (uint32_t i = 0; i < 16; i++) {
		//4 bit indices
		uint32_t index;
		if (0 == i) {
			index = (indices >> shift) & 0b111;
			shift += 3;
		} else {
			index = (indices >> shift) & 0b1111;
			shift += 4;
		}

		uint32_t interpolationFactor = bc7InterpolationFactors4[index];
		RGBA e0 = endpoints[0];
		RGBA e1 = endpoints[1];
		pixels[i].r = bc7_interpolate(e0.r, e1.r, interpolationFactor);
		pixels[i].g = bc7_interpolate(e0.g, e1.g, interpolationFactor);
		pixels[i].b = bc7_interpolate(e0.b, e1.b, interpolationFactor);
		pixels[i].a = bc7_interpolate(e0.a, e1.a, interpolationFactor);
	}
}

void decompress_bc7_mode7(BC7Block& block, RGBA pixels[16]) {
	//6 bits of partition selection
	uint32_t partitionSelection = (block.data[0] >> 8) & 0b111111;
	//2 partitions
	const BC7PartitionTable& partition = bc7PartitionTable2Subsets[partitionSelection];
	uint8_t anchor2nd = bc7PartitionTable2Anchors2ndSubset[partitionSelection];

	uint32_t endpointReds = block.data[0] >> 14;
	uint32_t endpointGreens = block.data[0] >> 34;
	uint32_t endpointBlues = (block.data[0] >> 54) | (block.data[1] << (64 - 54));
	uint32_t endpointAlphas = block.data[1] >> 10;
	uint32_t endpointPBits = block.data[1] >> 30;
	//Mode 7 has 2 endpoints
	RGBA endpoints[4];
	for (uint32_t i = 0; i < 4; i++) {
		uint32_t pBit = ((endpointPBits >> i) & 1) << 2;
		//5 bit colors
		endpoints[i].r = (endpointReds >> (i * 5)) & 0b11111;
		endpoints[i].r = (endpoints[i].r << 3) | pBit | (endpoints[i].r >> 3);
		endpoints[i].g = (endpointGreens >> (i * 5)) & 0b11111;
		endpoints[i].g = (endpoints[i].g << 3) | pBit | (endpoints[i].g >> 3);
		endpoints[i].b = (endpointBlues >> (i * 5)) & 0b11111;
		endpoints[i].b = (endpoints[i].b << 3) | pBit | (endpoints[i].b >> 3);
		endpoints[i].a = (endpointAlphas >> (i * 5)) & 0b11111;
		endpoints[i].a = (endpoints[i].a << 3) | pBit | (endpoints[i].a >> 3);
	}

	uint64_t indices = block.data[1] >> 34;
	uint32_t shift = 0;
	for (uint32_t i = 0; i < 16; i++) {
		//2 bit indices
		uint32_t index;
		if ((anchor2nd == i) | (0 == i)) {
			index = (indices >> shift) & 0b1;
			shift += 1;
		} else {
			index = (indices >> shift) & 0b11;
			shift += 2;
		}

		uint32_t interpolationFactor = bc7InterpolationFactors2[index];
		uint8_t partitionNumber = partition.partitionNumbers[i];
		RGBA e0 = endpoints[partitionNumber * 2];
		RGBA e1 = endpoints[partitionNumber * 2 + 1];
		pixels[i].r = bc7_interpolate(e0.r, e1.r, interpolationFactor);
		pixels[i].g = bc7_interpolate(e0.g, e1.g, interpolationFactor);
		pixels[i].b = bc7_interpolate(e0.b, e1.b, interpolationFactor);
		pixels[i].a = bc7_interpolate(e0.a, e1.a, interpolationFactor);
	}
}

void decompress_bc7_block(BC7Block& block, RGBA pixels[16]) {
	uint32_t mode = extract_mode(block);
	//Since most of these are nearly the same I could have made one or two more generic methods to decode all of them.
	//I find it easier to think about if I only have to consider one mode at a time
	switch (mode) {
	case 0: decompress_bc7_mode0(block, pixels); break;
	case 1: decompress_bc7_mode1(block, pixels); break;
	case 2: decompress_bc7_mode2(block, pixels); break;
	case 3: decompress_bc7_mode3(block, pixels); break;
	case 4: decompress_bc7_mode4(block, pixels); break;
	case 5: decompress_bc7_mode5(block, pixels); break;
	case 6: decompress_bc7_mode6(block, pixels); break;
	case 7: decompress_bc7_mode7(block, pixels); break;
	default:
		//Error
		memset(pixels, 0, 16 * sizeof(RGBA));
		return;
	}
}

RGBA* decompress_bc7(BC7Block* blocks, uint32_t finalWidth, uint32_t finalHeight) {
	uint32_t blockWidth = (finalWidth + 3) / 4;
	uint32_t blockHeight = (finalHeight + 3) / 4;
	RGBA* finalImage = reinterpret_cast<RGBA*>(malloc(finalWidth * finalHeight * sizeof(RGBA)));
	if (!finalImage) {
		return nullptr;
	}
	//4x4 block
	RGBA pixels[4 * 4];
	for (uint32_t y = 0; y < blockHeight; y++) {
		for (uint32_t x = 0; x < blockWidth; x++) {
			BC7Block& block = blocks[y * blockWidth + x];
			decompress_bc7_block(block, pixels);
			copy_block_pixels_to_image(pixels, finalImage, x, y, finalWidth, finalHeight);
		}
	}
	return finalImage;
}

enum BC7ReadError {
	BC7_READ_SUCCESS = 0,
	BC7_READ_BAD_HEADER_SIZE = 1,
	BC7_READ_NOT_DDS = 2,
	BC7_READ_NOT_BC7 = 3
};

BC7ReadError read_dds_file_bc7(byte* file, BC7Block** blocks, uint32_t* width, uint32_t* height) {
	if (memcmp(file, "DDS ", 4) != 0) {
		return BC7_READ_SUCCESS;
	}
	file += 4;

	DDSHeader& header = *reinterpret_cast<DDSHeader*>(file);
	if (header.size != 124) {
		return BC7_READ_BAD_HEADER_SIZE;
	}
	file += header.size;

	*width = header.width;
	*height = header.height;

	if (header.pixelFormat.flags & DDPF_FOURCC && memcmp(header.pixelFormat.fourCC, "DX10", 4) == 0) {
		DDSHeaderDX10& dx10Header = *reinterpret_cast<DDSHeaderDX10*>(file);
		file += sizeof(DDSHeaderDX10);
		if (dx10Header.dxgiFormat != DXGI_FORMAT_BC7_TYPELESS && dx10Header.dxgiFormat != DXGI_FORMAT_BC7_UNORM && dx10Header.dxgiFormat != DXGI_FORMAT_BC7_UNORM_SRGB) {
			return BC7_READ_NOT_BC7;
		}
		*blocks = reinterpret_cast<BC7Block*>(file);
		return BC7_READ_SUCCESS;
	}
	return BC7_READ_NOT_BC7;
}