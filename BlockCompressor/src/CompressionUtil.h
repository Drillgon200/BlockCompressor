#pragma once
#include <stdint.h>
#include "SIMDUtil.h"

struct RGBA {
	uint8_t r;
	uint8_t g;
	uint8_t b;
	uint8_t a;

	RGBA convert_to_rgba() {
		return *this;
	}

	RGBA operator+(RGBA other) {
		return RGBA{ static_cast<uint8_t>(r + other.r), static_cast<uint8_t>(g + other.g), static_cast<uint8_t>(b + other.b), static_cast<uint8_t>(a + other.a) };
	}
	RGBA operator-(RGBA other) {
		return RGBA{ static_cast<uint8_t>(r - other.r), static_cast<uint8_t>(g - other.g), static_cast<uint8_t>(b - other.b), static_cast<uint8_t>(a - other.a) };
	}
	bool operator==(RGBA other) {
		return r == other.r && g == other.g && b == other.b && a == other.a;
	}
};

struct R {
	uint8_t r;
	RGBA convert_to_rgba() {
		return RGBA{ r, 0, 0, 255 };
	}

	bool operator==(R other) {
		return r == other.r;
	}
};

struct RG {
	uint8_t r;
	uint8_t g;

	RGBA convert_to_rgba() {
		return RGBA{ r, 0, 0, g };
	}

	bool operator==(RG other) {
		return r == other.r && g == other.g;
	}
};

struct RGB {
	uint8_t r;
	uint8_t g;
	uint8_t b;

	RGBA convert_to_rgba() {
		return RGBA{ r, g, b, 255 };
	}

	RGB operator+(RGB other) {
		return RGB{ static_cast<uint8_t>(r + other.r), static_cast<uint8_t>(g + other.g), static_cast<uint8_t>(b + other.b) };
	}
	RGB operator-(RGB other) {
		return RGB{ static_cast<uint8_t>(r - other.r), static_cast<uint8_t>(g - other.g), static_cast<uint8_t>(b - other.b) };
	}

	bool operator==(RGB other) {
		return r == other.r && g == other.g && b == other.b;
	}
};

float clamp01(float f) {
	return std::max(0.0F, std::min(1.0F, f));
}

inline RGB to_rgb(RGBA rgba) {
	return RGB{ rgba.r, rgba.g, rgba.b };
}

inline RGBA decode_5_6_5(uint16_t compressed) {
	//Extract the quantized bits
	uint8_t r = compressed >> 11;
	uint8_t g = (compressed >> 5) & 0b111111;
	uint8_t b = compressed & 0b11111;
	//Expand 5 to 8 bits by putting the 5 bits in the most significant, then putting the most significant of the 5 bits in the remaining 3 bits
	r = (r << 3) | (r >> 2);
	//Same but with 6 bits instead;
	g = (g << 2) | (g >> 4);
	//Same as 1
	b = (b << 3) | (b >> 2);
	return RGBA{ r, g, b, 255 };
}

inline uint16_t encode_5_6_5(RGB data) {
	return ((data.r << 8) & 0b1111100000000000) | ((data.g << 3) & 0b0000011111100000) | ((data.b >> 3) & 0b0000000000011111);
}

inline uint16_t encode_5_6_5_f(float data[3]) {
	RGB rgb;
	rgb.r = static_cast<uint8_t>(clamp01(data[0]) * 255.0F);
	rgb.g = static_cast<uint8_t>(clamp01(data[1]) * 255.0F);
	rgb.b = static_cast<uint8_t>(clamp01(data[2]) * 255.0F);
	return encode_5_6_5(rgb);
}

inline void quantize_565_f(float rgb[3]) {
	//Magic numbers from https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/dxtc/doc/cuda_dxtc.pdf
	//They found that these factors produced the lowest error. They're pretty close to 1/31 and 1/63 anyway.
	rgb[0] = rintf(clamp01(rgb[0]) * 31.0F) * 0.03227752766457F;
	rgb[1] = rintf(clamp01(rgb[1]) * 63.0F) * 0.01583151765563F;
	rgb[2] = rintf(clamp01(rgb[2]) * 31.0F) * 0.03227752766457F;
}

inline uint32_t get_difference(RGB rgb1, RGB rgb2) {
	//Numbers come from inverse of 0.2989 0.5870 0.1140 color perception weights
	uint32_t diff = (abs(static_cast<int32_t>(rgb1.r) - static_cast<int32_t>(rgb2.r)) * 100) / 335;
	diff += (abs(static_cast<int32_t>(rgb1.g) - static_cast<int32_t>(rgb2.g)) * 100) / 170;
	diff += (abs(static_cast<int32_t>(rgb1.b) - static_cast<int32_t>(rgb2.b)) * 100) / 877;
	return diff;
}

inline void get_average3d(float* outVec, float* inputVectors, uint32_t inputCount) {
	uint32_t inputFloats = inputCount * 3;
	for (uint32_t i = 0; i < inputFloats; i += 3) {
		outVec[0] += inputVectors[i + 0];
		outVec[1] += inputVectors[i + 1];
		outVec[2] += inputVectors[i + 2];
	}
	float invInputCount = 1.0F / static_cast<float>(inputCount);
	outVec[0] *= invInputCount;
	outVec[1] *= invInputCount;
	outVec[2] *= invInputCount;
}

void principle_component_analysis3d(float* outMean, float* outVec, float* inputVectors, uint32_t inputCount) {
	float average[3]{};
	get_average3d(average, inputVectors, inputCount);
	//Upper triangle of covariance matrix
	float covXX = 0.0F;
	float covXY = 0.0F;
	float covXZ = 0.0F;
	float covYY = 0.0F;
	float covYZ = 0.0F;
	float covZZ = 0.0F;
	//Compute covariance matrix. No need to normalize since we only need the direction from it.
	uint32_t inputFloats = inputCount * 3;
	for (uint32_t i = 0; i < inputFloats; i += 3) {
		float translatedX = inputVectors[i + 0] - average[0];
		float translatedY = inputVectors[i + 1] - average[1];
		float translatedZ = inputVectors[i + 2] - average[2];
		covXX += translatedX * translatedX;
		covXY += translatedX * translatedY;
		covXZ += translatedX * translatedZ;
		covYY += translatedY * translatedY;
		covYZ += translatedY * translatedZ;
		covZZ += translatedZ * translatedZ;
	}

	//Power iteration to find the eigenvector we need. Thank god I found this before trying to do the whole cubic formula.
	//http://theory.stanford.edu/~tim/s15/l/l8.pdf
	//This paper says 8 is a good iteration count
	//https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/dxtc/doc/cuda_dxtc.pdf
	constexpr uint32_t powerIterations = 8;
	//__m128 row0 = _mm_setr_ps(covXX, covXY, covXZ, 0.0F);
	//__m128 row1 = _mm_setr_ps(covXY, covYY, covYZ, 0.0F);
	//__m128 row2 = _mm_setr_ps(covXZ, covYZ, covZZ, 0.0F);
	//__m128 vector = _mm_setr_ps(1.0F, 1.0F, 1.0F, 1.0F);
	//Hello, I'm Evan. I'm a Computer Science major who loves game engine programming, and I spend most of my time on a computer. I also enjoy creating game art with programs like blender. Naturally, I love videogames as well, particularly shooters and open world games, as well as virtual reality games. I like a variety of music, including metal, electronic, and traditional, and I play the fiddle and the banjo. I don't party much, and am generally not super outgoing. Discord is Drillgon200#0288.
	float vector[3]{ 1.0F, 1.0F, 1.0F };
	for (uint32_t i = 0; i < powerIterations; i++) {
		vector[0] = vector[0] * covXX + vector[1] * covXY + vector[2] * covXZ;
		vector[1] = vector[0] * covXY + vector[1] * covYY + vector[2] * covYZ;
		vector[2] = vector[0] * covXZ + vector[1] * covYZ + vector[2] * covZZ;
		//__m128 vectorX = _mm_shuffle_ps(vector, vector, _MM_SHUFFLE(0, 0, 0, 0));
		//__m128 vectorY = _mm_shuffle_ps(vector, vector, _MM_SHUFFLE(1, 1, 1, 1));
		//__m128 vectorZ = _mm_shuffle_ps(vector, vector, _MM_SHUFFLE(2, 2, 2, 2));
		//vector = _mm_mul_ps(row0, vectorX);
		//vector = _mm_fmadd_ps(row1, vectorY, vector);
		//vector = _mm_fmadd_ps(row2, vectorZ, vector);

		float rcp = 1.0F/std::max(vector[0], std::max(vector[1], vector[2]));
		vector[0] *= rcp;
		vector[1] *= rcp;
		vector[2] *= rcp;
		//vectorX = _mm_shuffle_ps(vector, vector, _MM_SHUFFLE(0, 0, 0, 0));
		//vectorY = _mm_shuffle_ps(vector, vector, _MM_SHUFFLE(1, 1, 1, 1));
		//vectorZ = _mm_shuffle_ps(vector, vector, _MM_SHUFFLE(2, 2, 2, 2));
		//__m128 rcp = _mm_rcp_ps(_mm_max_ps(_mm_max_ps(vectorX, vectorY), vectorZ));
		//vector = _mm_mul_ps(vector, rcp);

		//covXX *= covXX;
		//covXY *= covXY;
		//covXZ *= covXZ;
		//covYY *= covYY;
		//covYZ *= covYZ;
		//covZZ *= covZZ;
	}
	//float store[4];
	//_mm_store_ps(store, vector);
	//outVec[0] = store[0];
	//outVec[1] = store[1];
	//outVec[2] = store[2];
	float normalize = 1.0F / sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]);
	outVec[0] = vector[0] * normalize;
	outVec[1] = vector[1] * normalize;
	outVec[2] = vector[2] * normalize;
	outMean[0] = average[0];
	outMean[1] = average[1];
	outMean[2] = average[2];
}

//Too lazy to implement a real vector library right now, I'll convert it to vec3f when I move it to my engine
float dot(float* vec1, float* vec2) {
	return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2];
}

void scale(float* out, float* vec, float scalar) {
	out[0] = vec[0] * scalar;
	out[1] = vec[1] * scalar;
	out[2] = vec[2] * scalar;
}

void sub(float* out, float* vec1, float* vec2) {
	out[0] = vec1[0] - vec2[0];
	out[1] = vec1[1] - vec2[1];
	out[2] = vec1[2] - vec2[2];
}

float diff_sq_3(float a[3], float b[3]) {
	float d0 = a[0] - b[0];
	float d1 = a[1] - b[1];
	float d2 = a[2] - b[2];
	return d0 * d0 + d1 * d1 + d2 * d2;
}

float diff_sq_4(float a[4], float b[4]) {
	float d0 = a[0] - b[0];
	float d1 = a[1] - b[1];
	float d2 = a[2] - b[2];
	float d3 = a[3] - b[3];
	return d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
}

float squared_error(float originalImage[16*4], float newImage[16*4]) {
	float error = 0.0F;
	for (uint32_t pix = 0; pix < 16; pix++) {
		float weight = originalImage[pix * 4 + 3];
		for (uint32_t component = 0; component < 3; component++) {
			float diff = originalImage[pix * 4 + component] - newImage[pix * 4 + component];
			error += diff * diff * weight;
		}
		float diff = weight - newImage[pix * 4 + 3];
		error += diff * diff * (1.0F + (1.0F - weight) * 3.0F);
	}
	return error;
}

/*float psnr(float original[16 * 4], float error) {
	error /= 16.0F * 4.0F;
	float originalSq = 0;
	for (uint32_t pix = 0; pix < 16; pix++) {
		for (uint32_t component = 0; component < 4; component++) {
			float comp = original[pix * 4 + component];
			originalSq += comp * comp;
		}
	}
	originalSq /= 16.0F * 4.0F;
	return 10.0F * log10(originalSq / error);
}*/

void ordered_permute(uint8_t* order, int32_t index, uint32_t iterations) {
	if (index < 0) {
		std::cout << static_cast<uint32_t>(order[3]) << " " << static_cast<uint32_t>(order[2]) << " " << static_cast<uint32_t>(order[1]) << " " << static_cast<uint32_t>(order[0]) << std::endl;
	} else {
		for (uint32_t i = order[index+1]; i < iterations; i++) {
			order[index] = i;
			ordered_permute(order, index - 1, iterations);
		}
	}
}

template<typename T>
void generate_ordered_permutations(int32_t iterations, T* out, T (*compressorFunction)(uint8_t[16])) {
	uint8_t indices[17];
	for (uint32_t i = 0; i < 16; i++) {
		indices[i] = 0;
	}
	indices[16] = 0;
	int32_t index = 15;
	uint32_t count = 0;
	while (index < 16) {
		if (index < 0) {
			if (out != nullptr) {
				out[count] = compressorFunction(indices);
			}
			count++;
			index++;
			indices[index]++;
		} else {
			if (indices[index] < iterations) {
				index--;
			} else {
				index++;
				indices[index]++;
				for (int32_t i = index - 1; i >= 0; i--) {
					indices[i] = indices[index];
				}
			}
		}
	}
	//return count;
}

constexpr uint32_t bc1PermutationCount = 969;
constexpr uint32_t bc1AlphaPermutationCount = 153;
uint32_t bc1Permutations[bc1PermutationCount];
uint32_t bc1AlphaPermutations[bc1AlphaPermutationCount];

uint32_t bc1_index_compress(uint8_t indices[16]) {
	//Since the weights are 0, 1, 1/3, 2/3, the actual order we want is 0, 2, 3, 1
	constexpr uint32_t remap[4]{ 0, 2, 3, 1 };
	uint32_t result = 0;
	for (uint32_t i = 0; i < 16; i++) {
		result = (result << 2) | remap[indices[i]];
	}
	return result;
}

uint32_t bc1_index_compress_alpha(uint8_t indices[16]) {
	constexpr uint32_t remap[3]{ 0, 2, 1 };
	uint32_t result = 0;
	for (uint32_t i = 0; i < 16; i++) {
		result = (result << 2) | remap[indices[i]];
	}
	return result;
}

void bc1_index_decompress(uint32_t compressed, uint8_t indices[16]) {
	for (int32_t i = 15; i >= 0; i--) {
		indices[i] = compressed & 3;
		compressed >>= 2;
	}
}

void compute_compression_index_permutations() {
	generate_ordered_permutations<uint32_t>(4, bc1Permutations, bc1_index_compress);
	generate_ordered_permutations<uint32_t>(3, bc1AlphaPermutations, bc1_index_compress_alpha);
	//uint32_t test = generate_ordered_permutations<uint32_t>(8, nullptr, nullptr);
}

//outT will contain the min and max T values for this ray intersecting the box. pos must be inside the box. 
__forceinline void ray_cast_unit_box(float* outT, float* pos, float* dir) {
	float invDir[3];
	invDir[0] = 1.0F / dir[0];
	invDir[1] = 1.0F / dir[1];
	invDir[2] = 1.0F / dir[2];

	float xTMin = (-pos[0]) * invDir[0];
	float xTMax = (1.0F - pos[0]) * invDir[0];
	float yTMin = (-pos[1]) * invDir[1];
	float yTMax = (1.0F - pos[1]) * invDir[1];
	float zTMin = (-pos[2]) * invDir[2];
	float zTMax = (1.0F - pos[2]) * invDir[2];

	outT[0] = std::max(std::min(xTMin, xTMax), std::max(std::min(yTMin, yTMax), std::min(zTMin, zTMax)));
	outT[1] = std::min(std::max(xTMax, xTMin), std::min(std::max(yTMax, yTMin), std::max(zTMax, zTMin)));
}

void find_optimal_endpoints_rgb(float pixels[4*16], float colorVectors[3 * 16], RGB* endpoints, float mean[3], float principleComponent[3], uint32_t colorCount) {
	if (colorCount == 0) {
		//All transparent block
		endpoints[0] = RGB{ 0, 0, 0 };
		endpoints[1] = RGB{ 0, 0, 0 };
		return;
	}

	float boxClampT[2];
	ray_cast_unit_box(boxClampT, mean, principleComponent);
	float t1 = FLT_MAX;
	float t2 = -FLT_MAX;
	for (uint32_t i = 0; i < colorCount; i++) {
		float out[3];
		sub(out, colorVectors + i * 3, mean);
		float t = dot(out, principleComponent);
		t1 = std::min(t1, t);
		t2 = std::max(t2, t);
	}
	t1 = std::max(t1, boxClampT[0]);
	t2 = std::min(t2, boxClampT[1]);

	endpoints[0].r = static_cast<uint8_t>(std::min(std::max(mean[0] + principleComponent[0] * t1, 0.0F), 1.0F) * 255.0F);
	endpoints[0].g = static_cast<uint8_t>(std::min(std::max(mean[1] + principleComponent[1] * t1, 0.0F), 1.0F) * 255.0F);
	endpoints[0].b = static_cast<uint8_t>(std::min(std::max(mean[2] + principleComponent[2] * t1, 0.0F), 1.0F) * 255.0F);
	endpoints[1].r = static_cast<uint8_t>(std::min(std::max(mean[0] + principleComponent[0] * t2, 0.0F), 1.0F) * 255.0F);
	endpoints[1].g = static_cast<uint8_t>(std::min(std::max(mean[1] + principleComponent[1] * t2, 0.0F), 1.0F) * 255.0F);
	endpoints[1].b = static_cast<uint8_t>(std::min(std::max(mean[2] + principleComponent[2] * t2, 0.0F), 1.0F) * 255.0F);
}

void fill_pixel_block(RGBA* image, RGBA pixels[16], uint32_t blockX, uint32_t blockY, uint32_t finalWidth, uint32_t finalHeight) {
	for (uint32_t pixY = 0; pixY < 4; pixY++) {
		for (uint32_t pixX = 0; pixX < 4; pixX++) {
			uint32_t imageIndex = std::min(blockY * 4 + pixY, finalHeight - 1) * finalWidth + std::min(blockX * 4 + pixX, finalWidth - 1);
			uint32_t pixelIndex = pixY * 4 + pixX;
			pixels[pixelIndex] = image[imageIndex];
		}
	}
}

void fill_pixel_blockx8(RGBA* image, float* pixels, int32_t index, uint32_t blockX, uint32_t blockY, uint32_t finalWidth, uint32_t finalHeight) {
	// Every 8 indices, move to the next block of 8 4x4 color blocks
	// Add the lower part of index to offset for the in block offset for this pixel block
	uint32_t blockOffset = (index / 8) * 32 * 16 + index % 8;
	for (uint32_t pixY = 0; pixY < 4; pixY++) {
		for (uint32_t pixX = 0; pixX < 4; pixX++) {
			uint32_t imageIndex = std::min(blockY * 4 + pixY, finalHeight - 1) * finalWidth + std::min(blockX * 4 + pixX, finalWidth - 1);
			// 32 float stride between each 8 pixel block
			uint32_t pixelIndex = (pixY * 4 + pixX) * 32 + blockOffset;
			pixels[pixelIndex + 0]  = static_cast<float>(image[imageIndex].r) / 255.0F;
			pixels[pixelIndex + 8]  = static_cast<float>(image[imageIndex].g) / 255.0F;
			pixels[pixelIndex + 16] = static_cast<float>(image[imageIndex].b) / 255.0F;
			pixels[pixelIndex + 24] = static_cast<float>(image[imageIndex].a) / 255.0F;
		}
	}
}

void copy_block_pixels_to_image(RGBA* blockPixels, RGBA* finalImage, uint32_t blockX, uint32_t blockY, uint32_t finalWidth, uint32_t finalHeight) {
	for (uint32_t pixY = 0; pixY < 4; pixY++) {
		uint32_t imgY = blockY * 4 + pixY;
		if (imgY >= finalHeight) {
			continue;
		}
		for (uint32_t pixX = 0; pixX < 4; pixX++) {
			uint32_t imgX = blockX * 4 + pixX;
			if (imgX >= finalWidth) {
				continue;
			}
			uint32_t imageIndex = imgY * finalWidth + imgX;
			uint32_t pixelIndex = pixY * 4 + pixX;
			finalImage[imageIndex] = blockPixels[pixelIndex];
		}
	}
}