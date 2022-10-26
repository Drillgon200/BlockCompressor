#pragma once

#include "BC3Compress.h"

#pragma pack(push, 1)
struct BC4Block {
	BC3SingleChannel rChannel;
};
#pragma pack(pop)

BC4Block* compress_bc4(RGBA* image, uint32_t width, uint32_t height) {
	uint32_t blockWidth = (width + 3) / 4;
	uint32_t blockHeight = (height + 3) / 4;
	uint32_t numBlocks = blockWidth * blockHeight;
	BC4Block* blocks = reinterpret_cast<BC4Block*>(malloc(numBlocks * sizeof(BC4Block)));
	if (!blocks) {
		return nullptr;
	}
	//4x4 block
	RGBA pixels[4 * 4];
	for (uint32_t y = 0; y < blockHeight; y++) {
		for (uint32_t x = 0; x < blockWidth; x++) {
			fill_pixel_block(image, pixels, x, y, width, height);
			float reds[16];
			for (uint32_t i = 0; i < 16; i++) {
				reds[i] = static_cast<float>(pixels[i].r) / 255.0F;
			}
			compress_bc3_single_channel_block(reds, blocks[y * blockWidth + x].rChannel);
		}
	}
	return blocks;
}

RGBA* decompress_bc4(BC4Block* blocks, uint32_t finalWidth, uint32_t finalHeight) {
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
			BC4Block& block = blocks[y * blockWidth + x];
			byte values[16];
			decompress_bc3_single_channel_block(block.rChannel, values);
			for (uint32_t i = 0; i < 16; i++) {
				pixels[i].r = pixels[i].g = pixels[i].b = values[i];
				pixels[i].a = 255;
			}
			copy_block_pixels_to_image(pixels, finalImage, x, y, finalWidth, finalHeight);
		}
	}
	return finalImage;
}

enum BC4ReadError {
	BC4_READ_SUCCESS = 0,
	BC4_READ_BAD_HEADER_SIZE = 1,
	BC4_READ_NOT_DDS = 2,
	BC4_READ_NOT_BC4 = 3
};

BC4ReadError read_dds_file_bc4(byte* file, BC4Block** blocks, uint32_t* width, uint32_t* height) {
	if (memcmp(file, "DDS ", 4) != 0) {
		return BC4_READ_SUCCESS;
	}
	file += 4;

	DDSHeader& header = *reinterpret_cast<DDSHeader*>(file);
	if (header.size != 124) {
		return BC4_READ_BAD_HEADER_SIZE;
	}
	file += header.size;

	*width = header.width;
	*height = header.height;

	if (header.pixelFormat.flags & DDPF_FOURCC && memcmp(header.pixelFormat.fourCC, "DX10", 4) == 0) {
		DDSHeaderDX10& dx10Header = *reinterpret_cast<DDSHeaderDX10*>(file);
		file += sizeof(DDSHeaderDX10);
		if (dx10Header.dxgiFormat != DXGI_FORMAT_BC4_TYPELESS && dx10Header.dxgiFormat != DXGI_FORMAT_BC4_UNORM && dx10Header.dxgiFormat != DXGI_FORMAT_BC4_SNORM) {
			return BC4_READ_NOT_BC4;
		}
		*blocks = reinterpret_cast<BC4Block*>(file);
		return BC4_READ_SUCCESS;
	}
	return BC4_READ_NOT_BC4;
}