#pragma once

#include "BC3Compress.h"

#pragma pack(push, 1)
struct BC5Block {
	BC3SingleChannel rChannel;
	BC3SingleChannel gChannel;
};
#pragma pack(pop)

BC5Block* compress_bc5(RGBA* image, uint32_t width, uint32_t height) {
	uint32_t blockWidth = (width + 3) / 4;
	uint32_t blockHeight = (height + 3) / 4;
	uint32_t numBlocks = blockWidth * blockHeight;
	BC5Block* blocks = reinterpret_cast<BC5Block*>(malloc(numBlocks * sizeof(BC5Block)));
	if (!blocks) {
		return nullptr;
	}
	//4x4 block
	RGBA pixels[4 * 4];
	for (uint32_t y = 0; y < blockHeight; y++) {
		for (uint32_t x = 0; x < blockWidth; x++) {
			fill_pixel_block(image, pixels, x, y, width, height);
			float reds[16];
			float greens[16];
			for (uint32_t i = 0; i < 16; i++) {
				reds[i] = static_cast<float>(pixels[i].r) / 255.0F;
				greens[i] = static_cast<float>(pixels[i].g) / 255.0F;
			}
			compress_bc3_single_channel_block(reds, blocks[y * blockWidth + x].rChannel);
			compress_bc3_single_channel_block(greens, blocks[y * blockWidth + x].gChannel);
		}
	}
	return blocks;
}

RGBA* decompress_bc5(BC5Block* blocks, uint32_t finalWidth, uint32_t finalHeight) {
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
			BC5Block& block = blocks[y * blockWidth + x];
			byte rChannel[16];
			byte gChannel[16];
			decompress_bc3_single_channel_block(block.rChannel, rChannel);
			decompress_bc3_single_channel_block(block.gChannel, gChannel);
			for (uint32_t i = 0; i < 16; i++) {
				pixels[i].r = rChannel[i];
				pixels[i].g = gChannel[i];
				pixels[i].b = 0;
				pixels[i].a = 255;
			}
			copy_block_pixels_to_image(pixels, finalImage, x, y, finalWidth, finalHeight);
		}
	}
	return finalImage;
}

enum BC5ReadError {
	BC5_READ_SUCCESS = 0,
	BC5_READ_BAD_HEADER_SIZE = 1,
	BC5_READ_NOT_DDS = 2,
	BC5_READ_NOT_BC5 = 3
};

BC5ReadError read_dds_file_bc5(byte* file, BC5Block** blocks, uint32_t* width, uint32_t* height) {
	if (memcmp(file, "DDS ", 4) != 0) {
		return BC5_READ_SUCCESS;
	}
	file += 4;

	DDSHeader& header = *reinterpret_cast<DDSHeader*>(file);
	if (header.size != 124) {
		return BC5_READ_BAD_HEADER_SIZE;
	}
	file += header.size;

	*width = header.width;
	*height = header.height;

	if (header.pixelFormat.flags & DDPF_FOURCC && memcmp(header.pixelFormat.fourCC, "DX10", 4) == 0) {
		DDSHeaderDX10& dx10Header = *reinterpret_cast<DDSHeaderDX10*>(file);
		file += sizeof(DDSHeaderDX10);
		if (dx10Header.dxgiFormat != DXGI_FORMAT_BC5_TYPELESS && dx10Header.dxgiFormat != DXGI_FORMAT_BC5_UNORM && dx10Header.dxgiFormat != DXGI_FORMAT_BC5_SNORM) {
			return BC5_READ_NOT_BC5;
		}
		*blocks = reinterpret_cast<BC5Block*>(file);
		return BC5_READ_SUCCESS;
	}
	return BC5_READ_NOT_BC5;
}