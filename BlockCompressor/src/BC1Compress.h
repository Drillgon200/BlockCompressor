#pragma once

#include <stdint.h>
#include <iostream>
#include <intrin.h>
#include <vector>
#include "CompressionUtil.h"
#include "FileUtil.h"

typedef uint8_t byte;

#pragma pack(push, 1)
struct BC1Block {
	uint16_t endPoints[2];
	uint32_t pixels;
};
#pragma pack(pop)

void flip_bc1_indices(BC1Block& block, bool isAlpha) {
	if ((isAlpha && block.endPoints[0] > block.endPoints[1]) || (!isAlpha && block.endPoints[0] <= block.endPoints[1])) {
		uint16_t tmp = block.endPoints[0];
		block.endPoints[0] = block.endPoints[1];
		block.endPoints[1] = tmp;
		//Flip every second bit to turn the indices around
		uint32_t newPixels = block.pixels ^ 0x55555555;
		if (isAlpha) {
			newPixels &= ~((block.pixels >> 1) & 0x55555555);
		}
		block.pixels = newPixels;
	}
}

void minmax_compress_bc1_block(BC1Block& block, float pixels[16*4]) {
	/*//Stores the first and second min
	float max[6]{ -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX };
	//Stores the first and second max
	float min[6]{ FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX };
	for (uint32_t i = 0; i < 16; i++) {
		for (uint32_t component = 0; component < 3; component++) {
			float comp = pixels[i * 4 + component];

			float old = max[component + 3];
			max[component + 3] = std::max(std::min(max[component], comp), old);
			max[component] = std::max(max[component], comp);

			old = min[component + 3];
			min[component + 3] = std::min(std::max(min[component], comp), old);
			min[component] = std::min(min[component], comp);
		}
	}*/
	float min[3]{ pixels[0], pixels[1], pixels[2] };
	float max[3]{ pixels[0], pixels[1], pixels[2] };
	for (uint32_t i = 1; i < 16; i++) {
		for (uint32_t component = 0; component < 3; component++) {
			min[component] = std::min(min[component], pixels[i * 4 + component]);
			max[component] = std::max(max[component], pixels[i * 4 + component]);
		}
	}
	block.endPoints[0] = encode_5_6_5_f(min);
	block.endPoints[1] = encode_5_6_5_f(max);
	quantize_565_f(min);
	quantize_565_f(max);

	//Set up lookup table for alpha mode
	float table[4 * 4]{
		min[0],
		min[1],
		min[2],
		1.0F,
		max[0],
		max[1],
		max[2],
		1.0F,
		(min[0] + max[0]) / 2.0F,
		(min[1] + max[1]) / 2.0F,
		(min[2] + max[2]) / 2.0F,
		1.0F,
		0.0F,
		0.0F,
		0.0F,
		0.0F
	};

	uint32_t alphaIndices = 0;
	float totalAlphaError = 0.0F;
	for (uint32_t i = 0; i < 16; i++) {
		uint32_t bestIndex = 0;
		float bestError = FLT_MAX;
		//Check each index, pick the one with best error. There are only 4 to check, so it should be fast enough.
		for (uint32_t idx = 0; idx < 4; idx++) {
			float error = diff_sq_4(&pixels[i * 4], &table[idx * 4]);
			if (error < bestError) {
				bestError = error;
				bestIndex = idx;
			}
		}
		alphaIndices |= bestIndex << (i * 2);
		totalAlphaError += bestError;
	}

	//Set up table for color mode
	table[8] = (min[0] * 2 + max[0]) / 3.0F;
	table[9] = (min[1] * 2 + max[1]) / 3.0F;
	table[10] = (min[2] * 2 + max[2]) / 3.0F;
	table[11] = 1.0F;
	table[12] = (min[0] + max[0] * 2) / 3.0F;
	table[13] = (min[1] + max[1] * 2) / 3.0F;
	table[14] = (min[2] + max[2] * 2) / 3.0F;
	table[15] = 1.0F;

	uint32_t indices = 0;
	float totalError = 0.0F;
	for (uint32_t i = 0; i < 16; i++) {
		uint32_t bestIndex = 0;
		float bestError = FLT_MAX;
		//Check each index, pick the one with best error. There are only 4 to check, so it should be fast enough.
		for (uint32_t idx = 0; idx < 4; idx++) {
			float error = diff_sq_4(&pixels[i * 4], &table[idx * 4]);
			if (error < bestError) {
				bestError = error;
				bestIndex = idx;
			}
		}
		indices |= bestIndex << (i * 2);
		totalError += bestError;
	}

	if (block.endPoints[0] == block.endPoints[1]) {
		if (totalAlphaError < totalError) {
			block.pixels = 0xFFFFFFFF;
		} else {
			block.pixels = 0;
		}
		return;
	}

	if (totalAlphaError < totalError) {
		block.pixels = alphaIndices;
	} else {
		block.pixels = indices;
	}
	flip_bc1_indices(block, totalAlphaError < totalError);
}

//https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/dxtc/doc/cuda_dxtc.pdf
float least_squares_cluster_fit_3_bc1(uint32_t indices, float pixels[16 * 4], float outEndpoints[3 * 2]) {
	constexpr float coeff[3]{ 1.0F, 0.0F, 1.0F / 2.0F };

	float alphaSq = 0.0F;
	float alphaBeta = 0.0F;
	float betaSq = 0.0F;
	float alphaX[3]{ 0.0F, 0.0F, 0.0F };
	float betaX[3]{ 0.0F, 0.0F, 0.0F };
	//Find least squares best fit for indices
	for (uint32_t i = 0; i < 16; i++) {
		float alpha = coeff[(indices >> (i * 2)) & 3];
		float beta = 1.0F - alpha;

		//Block has alpha, weight each pixel's contribution by its alpha.
		float weight = pixels[i * 4 + 3];

		alphaSq += alpha * alpha * weight;
		alphaBeta += alpha * beta * weight;
		betaSq += beta * beta * weight;

		alpha *= weight;
		beta *= weight;

		alphaX[0] += alpha * pixels[i * 4 + 0];
		alphaX[1] += alpha * pixels[i * 4 + 1];
		alphaX[2] += alpha * pixels[i * 4 + 2];
		betaX[0] += beta * pixels[i * 4 + 0];
		betaX[1] += beta * pixels[i * 4 + 1];
		betaX[2] += beta * pixels[i * 4 + 2];
	}
	//Inverse matrix
	float inverseDeterminant = 1.0F / (alphaSq * betaSq - alphaBeta * alphaBeta);
	alphaBeta = -alphaBeta * inverseDeterminant;
	alphaSq *= inverseDeterminant;
	betaSq *= inverseDeterminant;

	float endpoint1[3];
	float endpoint2[3];
	for (uint32_t i = 0; i < 3; i++) {
		endpoint1[i] = betaSq * alphaX[i] + alphaBeta * betaX[i];
		endpoint2[i] = alphaBeta * alphaX[i] + alphaSq * betaX[i];
		outEndpoints[i] = endpoint1[i];
		outEndpoints[i + 3] = endpoint2[i];
	}

	quantize_565_f(endpoint1);
	quantize_565_f(endpoint2);

	float newPixels[16 * 4];
	for (uint32_t i = 0; i < 16; i++) {
		uint32_t index = (indices >> (i * 2)) & 3;
		newPixels[i * 4 + 0] = endpoint1[0] * coeff[index] + endpoint2[0] * (1.0F - coeff[index]);
		newPixels[i * 4 + 1] = endpoint1[1] * coeff[index] + endpoint2[1] * (1.0F - coeff[index]);
		newPixels[i * 4 + 2] = endpoint1[2] * coeff[index] + endpoint2[2] * (1.0F - coeff[index]);
		newPixels[i * 4 + 3] = (pixels[i * 4 + 3] < 0.5F) ? 0.0F : 1.0F;
	}
	return squared_error(pixels, newPixels);
}
float least_squares_cluster_fit_4_bc1(uint32_t indices, float pixels[16 * 4], float outEndpoints[3*2]) {
	constexpr float coeff[4]{ 1.0F, 0.0F, 2.0F / 3.0F, 1.0F / 3.0F };

	float alphaSq = 0.0F;
	float alphaBeta = 0.0F;
	float betaSq = 0.0F;
	float alphaX[3]{ 0.0F, 0.0F, 0.0F };
	float betaX[3]{ 0.0F, 0.0F, 0.0F };
	//Find least squares best fit for indices
	for (uint32_t i = 0; i < 16; i++) {
		float alpha = coeff[(indices >> (i * 2)) & 3];
		float beta = 1.0F-alpha;

		alphaSq += alpha * alpha;
		alphaBeta += alpha * beta;
		betaSq += beta * beta;

		alphaX[0] += alpha * pixels[i * 4 + 0];
		alphaX[1] += alpha * pixels[i * 4 + 1];
		alphaX[2] += alpha * pixels[i * 4 + 2];
		betaX[0] += beta * pixels[i * 4 + 0];
		betaX[1] += beta * pixels[i * 4 + 1];
		betaX[2] += beta * pixels[i * 4 + 2];
	}
	//std::cout << indices;
	//std::cout << std::endl;

	//Inverse matrix
	float inverseDeterminant = 1.0F / (alphaSq * betaSq - alphaBeta * alphaBeta);
	alphaBeta = -alphaBeta * inverseDeterminant;
	alphaSq *= inverseDeterminant;
	betaSq *= inverseDeterminant;

	float endpoint1[3];
	float endpoint2[3];
	for (uint32_t i = 0; i < 3; i++) {
		endpoint1[i] = betaSq * alphaX[i] + alphaBeta * betaX[i];
		//Four hours debugging to realize I wrote alphaBeta instead of alphaSq here...
		endpoint2[i] = alphaBeta * alphaX[i] + alphaSq * betaX[i];
		outEndpoints[i] = endpoint1[i];
		outEndpoints[i + 3] = endpoint2[i];
	}

	quantize_565_f(endpoint1);
	quantize_565_f(endpoint2);

	float newPixels[16 * 4];
	for (uint32_t i = 0; i < 16; i++) {
		uint32_t index = (indices >> (i * 2)) & 3;
		newPixels[i * 4 + 0] = endpoint1[0] * coeff[index] + endpoint2[0] * (1.0F - coeff[index]);
		newPixels[i * 4 + 1] = endpoint1[1] * coeff[index] + endpoint2[1] * (1.0F - coeff[index]);
		newPixels[i * 4 + 2] = endpoint1[2] * coeff[index] + endpoint2[2] * (1.0F - coeff[index]);
		newPixels[i * 4 + 3] = 1.0F;
	}
	float sqErr = squared_error(pixels, newPixels);
	return sqErr;
}

float cluster_compress_bc1_block(BC1Block& block, float pixels[16*4], float mean[3], float pcaDirection[3]) {
	//Sort along PCA
	float project[16];
	uint32_t sort[16]{};
	for (uint32_t i = 0; i < 16; i++) {
		float subResult[3];
		sub(subResult, &pixels[i * 4], mean);
		project[i] = dot(subResult, pcaDirection);
	}
	//Good enough. Not like it's going to be expensive compared to the rest of the algorithm
	float sortedPixels[16 * 4];
	for (uint32_t i = 0; i < 16; i++) {
		for (uint32_t j = 0; j < 16; j++) {
			sort[j] += static_cast<uint32_t>(project[i] < project[j]);
		}
	}
	for (uint32_t i = 0; i < 16; i++) {
		for (uint32_t j = i+1; j < 16; j++) {
			sort[j] += static_cast<uint32_t>(sort[i] == sort[j]);
		}
	}
	for (uint32_t i = 0; i < 16; i++) {
		for (uint32_t j = 0; j < 4; j++) {
			sortedPixels[sort[i] * 4 + j] = pixels[i * 4 + j];
		}
	}

	float bestError = FLT_MAX;
	float bestEndpoints[6];
	uint32_t bestIndices;
	bool isAlpha = false;
	for (uint32_t i = 0; i < bc1PermutationCount; i++) {
		float endpoints[6];
		float error = least_squares_cluster_fit_4_bc1(bc1Permutations[i], sortedPixels, endpoints);
		if (error < bestError) {
			bestError = error;
			bestIndices = bc1Permutations[i];
			memcpy(bestEndpoints, endpoints, 6 * sizeof(float));
		}
	}
	for (uint32_t i = 0; i < bc1AlphaPermutationCount; i++) {
		float endpoints[6];
		float error = least_squares_cluster_fit_3_bc1(bc1AlphaPermutations[i], sortedPixels, endpoints);
		if (error < bestError) {
			bestError = error;
			bestIndices = bc1AlphaPermutations[i];
			memcpy(bestEndpoints, endpoints, 6 * sizeof(float));
			isAlpha = true;
		}
	}

	/*for (uint32_t i = 0; i < 16; i++) {
		std::cout << ((bestIndices >> (i*2)) & 3) << " ";
	}
	std::cout << std::endl;*/
	
	block.endPoints[0] = encode_5_6_5_f(bestEndpoints);
	block.endPoints[1] = encode_5_6_5_f(bestEndpoints + 3);
	block.pixels = 0;
	if (block.endPoints[0] == block.endPoints[1]) {
		if (isAlpha) {
			for (uint32_t i = 0; i < 16; i++) {
				if (pixels[i * 4 + 3] < 0.5F) {
					block.pixels |= 3 << (i * 2);
				}
			}
		}
		return bestError;
	}

	for (uint32_t i = 0; i < 16; i++) {
		block.pixels |= ((bestIndices >> (sort[i] * 2)) & 3) << (i * 2);
	}
	flip_bc1_indices(block, isAlpha);
	if (isAlpha) {
		for (uint32_t i = 0; i < 16; i++) {
			if (pixels[i * 4 + 3] < 0.5F) {
				block.pixels |= 3 << (i * 2);
			}
		}
	}
	return bestError;
}

void pca_compress_bc1_block(BC1Block& block, float pixels[16*4], float pixelVectors[3*16], float mean[3], float principleComponent[3], uint32_t colorCount) {
	RGB endpoints[2];
	find_optimal_endpoints_rgb(pixels, pixelVectors, endpoints, mean, principleComponent, colorCount);
	if (colorCount == 0) {
		//Completely empty, fast path to a blank block
		block.endPoints[0] = 0;
		block.endPoints[1] = 0;
		block.pixels = 0xFFFFFFFF;
		return;
	}

	bool hasAlpha = colorCount < 16;

	block.endPoints[0] = encode_5_6_5(endpoints[0]);
	block.endPoints[1] = encode_5_6_5(endpoints[1]);
	block.pixels = 0;
	if (block.endPoints[0] == block.endPoints[1]) {
		if (hasAlpha) {
			block.pixels = 0xFFFFFFFF;
		}
		return;
	}
	endpoints[0] = to_rgb(decode_5_6_5(block.endPoints[0]));
	endpoints[1] = to_rgb(decode_5_6_5(block.endPoints[1]));
	if (block.endPoints[0] > block.endPoints[1] && hasAlpha) {
		uint16_t cTmp = block.endPoints[0];
		block.endPoints[0] = block.endPoints[1];
		block.endPoints[1] = cTmp;
		RGB eTmp = endpoints[0];
		endpoints[0] = endpoints[1];
		endpoints[1] = eTmp;
	}
	RGB colors[4]{ endpoints[0], endpoints[1] };
	if (hasAlpha) {
		colors[2] = RGB{ static_cast<uint8_t>((colors[0].r + colors[1].r) / 2), static_cast<uint8_t>((colors[0].g + colors[1].g) / 2) , static_cast<uint8_t>((colors[0].b + colors[1].b) / 2) };
		for (uint32_t i = 0; i < 16; i++) {
			if (pixels[i * 4 + 3] < 0.5F) {
				block.pixels |= 3 << (i * 2);
			} else {
				uint32_t bestIndex = 0;
				uint32_t bestMatch = 0xFFFFFFFF;
				for (uint32_t j = 0; j < 3; j++) {
					uint32_t diff = get_difference(colors[j], RGB{ static_cast<uint8_t>(pixels[i * 4 + 0] * 255.0F), static_cast<uint8_t>(pixels[i * 4 + 1] * 255.0F), static_cast<uint8_t>(pixels[i * 4 + 2] * 255.0F) });
					if (diff < bestMatch) {
						bestMatch = diff;
						bestIndex = j;
					}
				}
				block.pixels |= bestIndex << (i * 2);
			}
		}
	} else {
		colors[2] = RGB{ static_cast<uint8_t>((colors[0].r * 2 + colors[1].r) / 3), static_cast<uint8_t>((colors[0].g * 2 + colors[1].g) / 3), static_cast<uint8_t>((colors[0].b * 2 + colors[1].b) / 3) };
		colors[3] = RGB{ static_cast<uint8_t>((colors[0].r + colors[1].r * 2) / 3), static_cast<uint8_t>((colors[0].g + colors[1].g * 2) / 3), static_cast<uint8_t>((colors[0].b + colors[1].b * 2) / 3) };
		for (uint32_t i = 0; i < 16; i++) {
			uint32_t bestIndex = 0;
			uint32_t bestMatch = UINT32_MAX;
			for (uint32_t j = 0; j < 4; j++) {
				uint32_t diff = get_difference(colors[j], RGB{ static_cast<uint8_t>(pixels[i * 4 + 0] * 255.0F), static_cast<uint8_t>(pixels[i * 4 + 1] * 255.0F), static_cast<uint8_t>(pixels[i * 4 + 2] * 255.0F) });
				if (diff < bestMatch) {
					bestMatch = diff;
					bestIndex = j;
				}
			}
			block.pixels |= bestIndex << (i * 2);
		}
	}
	flip_bc1_indices(block, hasAlpha);
}

enum BC1CompressType {
	BC1_COMPRESS_PCA,
	BC1_COMPRESS_CLUSTER_FIT,
	BC1_COMPRESS_MINMAX
};

BC1Block* compress_bc1_block(RGBA pixels[16], BC1Block& block, BC1CompressType type) {
	float fpixels[16 * 4];
	//Convert pixels to float
	for (uint32_t i = 0; i < 16; i++) {
		fpixels[i * 4 + 0] = static_cast<float>(pixels[i].r) / 255.0F;
		fpixels[i * 4 + 1] = static_cast<float>(pixels[i].g) / 255.0F;
		fpixels[i * 4 + 2] = static_cast<float>(pixels[i].b) / 255.0F;
		fpixels[i * 4 + 3] = static_cast<float>(pixels[i].a) / 255.0F;
	}
	float colorVectors[16 * 3];
	uint32_t colorCount = 0;
	for (uint32_t i = 0; i < 16; i++) {
		if (fpixels[i * 4 + 3] > 0.5F) {
			colorVectors[colorCount * 3 + 0] = fpixels[i * 4 + 0];
			colorVectors[colorCount * 3 + 1] = fpixels[i * 4 + 1];
			colorVectors[colorCount * 3 + 2] = fpixels[i * 4 + 2];
			colorCount++;
		}
	}
	float mean[3];
	float principleComponent[3];
	principle_component_analysis3d(mean, principleComponent, colorVectors, colorCount);
	if (type == BC1_COMPRESS_PCA) {
		pca_compress_bc1_block(block, fpixels, colorVectors, mean, principleComponent, colorCount);
	} else if (type == BC1_COMPRESS_CLUSTER_FIT) {
		//uint64_t time = __rdtsc();
		cluster_compress_bc1_block(block, fpixels, mean, principleComponent);
		//uint64_t taken = __rdtsc() - time;
		//std::cout << "time: " << taken << std::endl;
	} else if (type == BC1_COMPRESS_MINMAX) {
		minmax_compress_bc1_block(block, fpixels);
	}
	return nullptr;
}

BC1Block* compress_bc1(RGBA* image, uint32_t width, uint32_t height, BC1CompressType type) {
	uint32_t blockWidth = (width + 3) / 4;
	uint32_t blockHeight = (height + 3) / 4;
	uint32_t numBlocks = blockWidth * blockHeight;
	BC1Block* blocks = reinterpret_cast<BC1Block*>(malloc(numBlocks * sizeof(BC1Block)));
	if (!blocks) {
		return nullptr;
	}
	//4x4 block
	RGBA pixels[4*4];
	for (uint32_t y = 0; y < blockHeight; y++) {
		for (uint32_t x = 0; x < blockWidth; x++) {
			fill_pixel_block(image, pixels, x, y, width, height);
			compress_bc1_block(pixels, blocks[y * blockWidth + x], type);
		}
	}
	return blocks;
}

void decompress_bc1_block(BC1Block& block, RGBA* pixels) {
	RGBA colors[4];
	colors[0] = decode_5_6_5(block.endPoints[0]);
	colors[1] = decode_5_6_5(block.endPoints[1]);
	if (block.endPoints[0] <= block.endPoints[1]) {
		//Has alpha
		//color_2 is 1/2 * color_0 + 1/2 * color_1
		colors[2] = RGBA{ static_cast<uint8_t>((colors[0].r + colors[1].r) / 2), static_cast<uint8_t>((colors[0].g + colors[1].g) / 2), static_cast<uint8_t>((colors[0].b + colors[1].b) / 2), 255 };
		//color_3 is black
		colors[3] = RGBA{ 0, 0, 0, 0 };
	} else {
		//No alpha
		//color_2 is 2/3 * color_0 + 1/3 * color_1
		colors[2] = RGBA{ static_cast<uint8_t>((colors[0].r * 2 + colors[1].r) / 3), static_cast<uint8_t>((colors[0].g * 2 + colors[1].g) / 3), static_cast<uint8_t>((colors[0].b * 2 + colors[1].b) / 3), 255 };
		//color_3 is 1/3 * color_0 + 2/3 * color_1
		colors[3] = RGBA{ static_cast<uint8_t>((colors[0].r + colors[1].r * 2) / 3), static_cast<uint8_t>((colors[0].g + colors[1].g * 2) / 3), static_cast<uint8_t>((colors[0].b + colors[1].b * 2) / 3), 255 };
	}
	for (uint32_t pix = 0; pix < 16; pix++) {
		uint32_t pixelIdx = (block.pixels >> (pix * 2)) & 0b11;
		pixels[pix] = colors[pixelIdx];
	}
}

RGBA* decompress_bc1(BC1Block* blocks, uint32_t finalWidth, uint32_t finalHeight) {
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
			decompress_bc1_block(blocks[y * blockWidth + x], pixels);
			copy_block_pixels_to_image(pixels, finalImage, x, y, finalWidth, finalHeight);
		}
	}
	return finalImage;
}

enum BC1ReadError {
	BC1_READ_SUCCESS = 0,
	BC1_READ_BAD_HEADER_SIZE = 1,
	BC1_READ_NOT_DDS = 2,
	BC1_READ_NOT_BC1 = 3
};

BC1ReadError read_dds_file_bc1(byte* file, BC1Block** blocks, uint32_t* width, uint32_t* height) {
	if (memcmp(file, "DDS ", 4) != 0) {
		return BC1_READ_NOT_DDS;
	}
	file += 4;

	DDSHeader& header = *reinterpret_cast<DDSHeader*>(file);
	if (header.size != 124) {
		return BC1_READ_BAD_HEADER_SIZE;
	}
	file += header.size;

	*width = header.width;
	*height = header.height;

	if (header.pixelFormat.flags & DDPF_FOURCC && memcmp(header.pixelFormat.fourCC, "DXT1", 4) == 0) {
		*blocks = reinterpret_cast<BC1Block*>(file);
		return BC1_READ_SUCCESS;
	}
	if (header.pixelFormat.flags & DDPF_FOURCC && memcmp(header.pixelFormat.fourCC, "DX10", 4) == 0) {
		DDSHeaderDX10& dx10Header = *reinterpret_cast<DDSHeaderDX10*>(file);
		file += sizeof(DDSHeaderDX10);
		if (dx10Header.dxgiFormat != DXGI_FORMAT_BC1_TYPELESS && dx10Header.dxgiFormat != DXGI_FORMAT_BC1_UNORM && dx10Header.dxgiFormat != DXGI_FORMAT_BC1_UNORM_SRGB) {
			return BC1_READ_NOT_BC1;
		}
		*blocks = reinterpret_cast<BC1Block*>(file);
		return BC1_READ_SUCCESS;
	}
	return BC1_READ_NOT_BC1;
}