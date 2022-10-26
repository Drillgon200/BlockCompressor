#pragma once

#include "BC1Compress.h"

#pragma pack(push, 1)
struct BC3SingleChannel {
	uint8_t alphaEndpoints[2];
	uint8_t alphaIndices[6];
};
struct BC3Block {
	BC3SingleChannel alpha;
	BC1Block color;
};
#pragma pack(pop)

void calc_single_alpha_indices_bc3(uint8_t endpoint0, uint8_t endpoint1, float originalValues[16], uint8_t outIndices[16]) {
	if (endpoint0 == endpoint1) {
		//If both are the same, this is mode 1, make everything end0 unless 0 or 1 is better
		for (uint32_t val = 0; val < 16; val++) {
			int32_t originalValInt = static_cast<int32_t>(originalValues[val] * 255.0F);
			int32_t end0Val = static_cast<int32_t>(endpoint0);
			if (abs(originalValInt - end0Val) > (255 - originalValInt)) {
				outIndices[val] = 7;
			} else if (abs(originalValInt - end0Val) > originalValInt) {
				outIndices[val] = 6;
			} else {
				outIndices[val] = 0;
			}
		}
		return;
	}
	//Endpoints are stored first, then interpolations, then special values. This maps from linear order to ends first.
	const uint8_t lookupMode0[]{ 0, 2, 3, 4, 5, 6, 7, 1 };
	const uint8_t lookupMode1[]{ 0, 2, 3, 4, 5, 1 };

	float end0 = static_cast<float>(endpoint0) / 255.0F;
	float end1 = static_cast<float>(endpoint1) / 255.0F;
	//Value that normalizes things between end0 and end1 to a 0-1 range.
	if (endpoint0 > endpoint1) {
		for (uint32_t val = 0; val < 16; val++) {
			//get 0-1 value representing position between end0 and end1, multiply by 7 to get to 0-7 range, add 0.5 so it rounds instead of truncating, turn it into a 4 bit integer index
			uint32_t position = static_cast<uint8_t>(clamp01((originalValues[val] - end0) / (end1 - end0)) * 7.0F + 0.5F);
			outIndices[val] = lookupMode0[position];
		}
	} else {
		for (uint32_t val = 0; val < 16; val++) {
			int32_t originalValInt = static_cast<int32_t>(originalValues[val] * 255.0F);
			//get 0-1 value representing position between end0 and end1, multiply by 7 to get to 0-7 range, add 0.5 so it rounds instead of truncating, turn it into a 4 bit integer index
			float position = std::floor(clamp01((originalValues[val] - end0) / (end1 - end0)) * 5.0F + 0.5F);
			int32_t decompress = static_cast<int32_t>(((position / 5.0F) * (end1 - end0) + end0) * 255.0F);
			if (abs(originalValInt - decompress) > (255-originalValInt)) {
				//Index 7, 255 has the best score
				outIndices[val] = 7;
			} else if (abs(originalValInt - decompress) > originalValInt) {
				//Index 6, 0 has the best score
				outIndices[val] = 6;
			} else {
				outIndices[val] = lookupMode1[static_cast<uint8_t>(position)];
			}
		}
	}
}

float calc_error_bc3(float newValues[16], float oldValues[16]) {
	float errAccumulator = 0.0F;
	for (uint32_t val = 0; val < 16; val++) {
		float err = std::fabs(newValues[val] - oldValues[val]);
		errAccumulator += err * err;
	}
	return errAccumulator;
}

float eval_single_alpha_bc3(int32_t endpoints[2], float originalValues[16]) {
	float quantizedValuesMode0[16];
	float quantizedValuesMode1[16];
	float end0 = static_cast<float>(endpoints[0]) / 255.0F;
	if (endpoints[0] == endpoints[1]) {
		//If both are the same, this is mode 1, make everything end0 unless 0 or 1 is closer.
		for (uint32_t val = 0; val < 16; val++) {
			quantizedValuesMode0[val] = quantizedValuesMode1[val] = end0;
			if (fabs(quantizedValuesMode1[val] - originalValues[val]) > (1.0F - originalValues[val])) {
				quantizedValuesMode1[val] = 1.0F;
			}
			if (fabs(quantizedValuesMode1[val] - originalValues[val]) > originalValues[val]) {
				quantizedValuesMode1[val] = 0.0F;
			}
		}
	} else {
		float end1 = static_cast<float>(endpoints[1]) / 255.0F;

		for (uint32_t val = 0; val < 16; val++) {
			float normalized = clamp01((originalValues[val] - end0) / (end1 - end0));
			//Quantize with 6 interpolated values
			quantizedValuesMode0[val] = (std::truncf(normalized * 7.0F + 0.5F) / 7.0F) * (end1 - end0) + end0;
			//Quantize with 5 interpolated values, and also check the hard coded 0 and 1 cases.
			quantizedValuesMode1[val] = (std::truncf(normalized * 5.0F + 0.5F) / 5.0F) * (end1 - end0) + end0;

			if (fabs(quantizedValuesMode1[val] - originalValues[val]) > (1.0F - originalValues[val])) {
				quantizedValuesMode1[val] = 1.0F;
			} else if (fabs(quantizedValuesMode1[val] - originalValues[val]) > (originalValues[val])) {
				quantizedValuesMode1[val] = 0.0F;
			}
		}
	}
	
	//Try interpolate 6
	float errMode0 = calc_error_bc3(quantizedValuesMode0, originalValues);
	//Try interpolate 4
	float errMode1 = calc_error_bc3(quantizedValuesMode1, originalValues);

	if (errMode0 < errMode1) {
		std::swap(endpoints[0], endpoints[1]);
		return errMode0;
	} else {
		return errMode1;
	}
}

void find_optimal_single_endpoints_range_search_bc3(float values[16], uint8_t outEndpoints[2]) {

	//Algorithm that should be fairly accurate and fast enough off the top of my head: Do a coarse search comparing every few values then a finer search.
	//I don't think the endpoint->error function should be all that high frequency.

	int32_t bestEndpoints[2];
	float bestError = FLT_MAX;

	const int32_t roughStep = 4;
	//Rough search
	for (int32_t end0 = 0; end0 < 256; end0 += roughStep) {
		for (int32_t end1 = end0; end1 < 256; end1 += roughStep) {
			int32_t endpoints[2]{end0, end1};
			float err = eval_single_alpha_bc3(endpoints, values);
			if (err < bestError) {
				bestEndpoints[0] = endpoints[0];
				bestEndpoints[1] = endpoints[1];
				bestError = err;
			}
		}
	}

	const int32_t fineOffset = roughStep - 1;
	//Fine search
	int32_t start0 = std::max(std::min(bestEndpoints[0], bestEndpoints[1]) - fineOffset, 0);
	int32_t final0 = std::min(std::min(bestEndpoints[0], bestEndpoints[1]) + fineOffset, 255);
	int32_t start1 = std::max(std::max(bestEndpoints[0], bestEndpoints[1]) - fineOffset, 0);
	int32_t final1 = std::min(std::max(bestEndpoints[0], bestEndpoints[1]) + fineOffset, 255);
	for (int32_t end0 = start0; end0 <= final0; end0++) {
		for (int32_t end1 = start1; end1 <= final1; end1++) {
			int32_t endpoints[2]{ end0, end1 };
			float err = eval_single_alpha_bc3(endpoints, values);
			if (err < bestError) {
				bestEndpoints[0] = endpoints[0];
				bestEndpoints[1] = endpoints[1];
				bestError = err;
			}
		}
	}

	outEndpoints[0] = bestEndpoints[0];
	outEndpoints[1] = bestEndpoints[1];
}

void compress_bc3_single_channel_block(float channel[16], BC3SingleChannel& alpha) {
	find_optimal_single_endpoints_range_search_bc3(channel, alpha.alphaEndpoints);
	uint8_t decompressedIndices[16];
	calc_single_alpha_indices_bc3(alpha.alphaEndpoints[0], alpha.alphaEndpoints[1], channel, decompressedIndices);
	uint64_t indices = 0;
	for (uint32_t i = 0; i < 16; i++) {
		indices |= static_cast<uint64_t>(decompressedIndices[i]) << (i * 3);
	}
	for (uint32_t i = 0; i < 6; i++) {
		alpha.alphaIndices[i] = indices >> (i * 8);
	}
}

BC3Block* compress_bc3(RGBA* image, uint32_t width, uint32_t height, BC1CompressType type) {
	uint32_t blockWidth = (width + 3) / 4;
	uint32_t blockHeight = (height + 3) / 4;
	uint32_t numBlocks = blockWidth * blockHeight;
	BC3Block* blocks = reinterpret_cast<BC3Block*>(malloc(numBlocks * sizeof(BC3Block)));
	if (!blocks) {
		return nullptr;
	}
	//4x4 block
	RGBA pixels[4 * 4];
	for (uint32_t y = 0; y < blockHeight; y++) {
		for (uint32_t x = 0; x < blockWidth; x++) {
			fill_pixel_block(image, pixels, x, y, width, height);
			compress_bc1_block(pixels, blocks[y * blockWidth + x].color, type);
			float alphas[16];
			for (uint32_t i = 0; i < 16; i++) {
				alphas[i] = static_cast<float>(pixels[i].a) / 255.0F;
			}
			compress_bc3_single_channel_block(alphas, blocks[y * blockWidth + x].alpha);
		}
	}
	return blocks;
}

void decompress_bc3_single_channel_block(BC3SingleChannel& alpha, byte values[16]) {
	uint64_t indices = 0;
	for (uint32_t i = 0; i < 6; i++) {
		indices |= static_cast<uint64_t>(alpha.alphaIndices[i]) << (i * 8);
	}
	uint8_t alphaLUT[8];
	alphaLUT[0] = alpha.alphaEndpoints[0];
	alphaLUT[1] = alpha.alphaEndpoints[1];
	if (alpha.alphaEndpoints[0] > alpha.alphaEndpoints[1]) {
		alphaLUT[2] = (6 * alpha.alphaEndpoints[0] + 1 * alpha.alphaEndpoints[1]) / 7;
		alphaLUT[3] = (5 * alpha.alphaEndpoints[0] + 2 * alpha.alphaEndpoints[1]) / 7;
		alphaLUT[4] = (4 * alpha.alphaEndpoints[0] + 3 * alpha.alphaEndpoints[1]) / 7;
		alphaLUT[5] = (3 * alpha.alphaEndpoints[0] + 4 * alpha.alphaEndpoints[1]) / 7;
		alphaLUT[6] = (2 * alpha.alphaEndpoints[0] + 5 * alpha.alphaEndpoints[1]) / 7;
		alphaLUT[7] = (1 * alpha.alphaEndpoints[0] + 6 * alpha.alphaEndpoints[1]) / 7;
	} else {
		alphaLUT[2] = (4 * alpha.alphaEndpoints[0] + 1 * alpha.alphaEndpoints[1]) / 5;
		alphaLUT[3] = (3 * alpha.alphaEndpoints[0] + 2 * alpha.alphaEndpoints[1]) / 5;
		alphaLUT[4] = (2 * alpha.alphaEndpoints[0] + 3 * alpha.alphaEndpoints[1]) / 5;
		alphaLUT[5] = (1 * alpha.alphaEndpoints[0] + 4 * alpha.alphaEndpoints[1]) / 5;
		alphaLUT[6] = 0;
		alphaLUT[7] = 255;
	}
	for (uint32_t pix = 0; pix < 16; pix++) {
		uint32_t index = indices & 0b111;
		indices >>= 3;
		values[pix] = alphaLUT[index];
	}
}

RGBA* decompress_bc3(BC3Block* blocks, uint32_t finalWidth, uint32_t finalHeight) {
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
			BC3Block& block = blocks[y * blockWidth + x];
			decompress_bc1_block(block.color, pixels);
			byte alpha[16];
			decompress_bc3_single_channel_block(block.alpha, alpha);
			for (uint32_t i = 0; i < 16; i++) {
				pixels[i].a = alpha[i];
			}
			copy_block_pixels_to_image(pixels, finalImage, x, y, finalWidth, finalHeight);
		}
	}
	return finalImage;
}

enum BC3ReadError {
	BC3_READ_SUCCESS = 0,
	BC3_READ_BAD_HEADER_SIZE = 1,
	BC3_READ_NOT_DDS = 2,
	BC3_READ_NOT_BC3 = 3
};

BC3ReadError read_dds_file_bc3(byte* file, BC3Block** blocks, uint32_t* width, uint32_t* height) {
	if (memcmp(file, "DDS ", 4) != 0) {
		return BC3_READ_NOT_DDS;
	}
	file += 4;

	DDSHeader& header = *reinterpret_cast<DDSHeader*>(file);
	if (header.size != 124) {
		return BC3_READ_BAD_HEADER_SIZE;
	}
	file += header.size;

	*width = header.width;
	*height = header.height;

	if (header.pixelFormat.flags & DDPF_FOURCC && memcmp(header.pixelFormat.fourCC, "DXT5", 4) == 0) {
		*blocks = reinterpret_cast<BC3Block*>(file);
		return BC3_READ_SUCCESS;
	}
	if (header.pixelFormat.flags & DDPF_FOURCC && memcmp(header.pixelFormat.fourCC, "DX10", 4) == 0) {
		DDSHeaderDX10& dx10Header = *reinterpret_cast<DDSHeaderDX10*>(file);
		file += sizeof(DDSHeaderDX10);
		if (dx10Header.dxgiFormat != DXGI_FORMAT_BC3_TYPELESS && dx10Header.dxgiFormat != DXGI_FORMAT_BC3_UNORM && dx10Header.dxgiFormat != DXGI_FORMAT_BC3_UNORM_SRGB) {
			return BC3_READ_NOT_BC3;
		}
		*blocks = reinterpret_cast<BC3Block*>(file);
		return BC3_READ_SUCCESS;
	}
	return BC3_READ_NOT_BC3;
}