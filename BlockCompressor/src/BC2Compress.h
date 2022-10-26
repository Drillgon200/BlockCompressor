#pragma once

#include "BC1Compress.h"

#pragma pack(push, 1)
struct BC2Block {
	//Alpha is 4 bits per color, each uint16 has 4 colors.
	uint16_t alpha[4];
	//Color block same as BC1
	BC1Block color;
};
#pragma pack(pop)

void compress_bc2_alpha_block(RGBA pixels[16], uint16_t outAlpha[4]) {
	memset(outAlpha, 0, 4 * sizeof(uint16_t));
	for (uint32_t y = 0; y < 4; y++) {
		for (uint32_t x = 0; x < 4; x++) {
			//Add 8 to round instead of truncate (we're in units of 16 here)
			uint16_t quantizedAlpha = std::min(static_cast<uint16_t>(pixels[y * 4 + x].a) + 8, 255) >> 4;
			outAlpha[y] |= quantizedAlpha << (x * 4);
		}
	}
}

BC2Block* compress_bc2(RGBA* image, uint32_t width, uint32_t height, BC1CompressType type) {
	uint32_t blockWidth = (width + 3) / 4;
	uint32_t blockHeight = (height + 3) / 4;
	uint32_t numBlocks = blockWidth * blockHeight;
	BC2Block* blocks = reinterpret_cast<BC2Block*>(malloc(numBlocks * sizeof(BC2Block)));
	if (!blocks) {
		return nullptr;
	}
	//4x4 block
	RGBA pixels[4 * 4];
	for (uint32_t y = 0; y < blockHeight; y++) {
		for (uint32_t x = 0; x < blockWidth; x++) {
			fill_pixel_block(image, pixels, x, y, width, height);
			//TODO I implemented this from Microsoft's article instead of the specification, and BC2/BC3 actually can't use the 3 color mode. I have to make sure to not encode that.
			compress_bc1_block(pixels, blocks[y * blockWidth + x].color, type);
			compress_bc2_alpha_block(pixels, blocks[y * blockWidth + x].alpha);
		}
	}
	return blocks;
}

void decompress_bc2_alpha_block(uint16_t bc2AlphaBlock[4], RGBA* pixels) {
	for (uint32_t y = 0; y < 4; y++) {
		for (uint32_t x = 0; x < 4; x++) {
			uint32_t compressedAlpha = (bc2AlphaBlock[y] >> (x * 4)) & 0b1111;
			pixels[y * 4 + x].a = (compressedAlpha << 4) | compressedAlpha;
		}
	}
}

RGBA* decompress_bc2(BC2Block* blocks, uint32_t finalWidth, uint32_t finalHeight) {
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
			decompress_bc1_block(blocks[y * blockWidth + x].color, pixels);
			decompress_bc2_alpha_block(blocks[y * blockWidth + x].alpha, pixels);
			copy_block_pixels_to_image(pixels, finalImage, x, y, finalWidth, finalHeight);
		}
	}
	return finalImage;
}

enum BC2ReadError {
	BC2_READ_SUCCESS = 0,
	BC2_READ_BAD_HEADER_SIZE = 1,
	BC2_READ_NOT_DDS = 2,
	BC2_READ_NOT_BC2 = 3
};

BC2ReadError read_dds_file_bc2(byte* file, BC2Block** blocks, uint32_t* width, uint32_t* height) {
	if (memcmp(file, "DDS ", 4) != 0) {
		return BC2_READ_NOT_DDS;
	}
	file += 4;

	DDSHeader& header = *reinterpret_cast<DDSHeader*>(file);
	if (header.size != 124) {
		return BC2_READ_BAD_HEADER_SIZE;
	}
	file += header.size;

	*width = header.width;
	*height = header.height;

	if (header.pixelFormat.flags & DDPF_FOURCC && memcmp(header.pixelFormat.fourCC, "DXT3", 4) == 0) {
		*blocks = reinterpret_cast<BC2Block*>(file);
		return BC2_READ_SUCCESS;
	}
	if (header.pixelFormat.flags & DDPF_FOURCC && memcmp(header.pixelFormat.fourCC, "DX10", 4) == 0) {
		DDSHeaderDX10& dx10Header = *reinterpret_cast<DDSHeaderDX10*>(file);
		file += sizeof(DDSHeaderDX10);
		if (dx10Header.dxgiFormat != DXGI_FORMAT_BC2_TYPELESS && dx10Header.dxgiFormat != DXGI_FORMAT_BC2_UNORM && dx10Header.dxgiFormat != DXGI_FORMAT_BC2_UNORM_SRGB) {
			return BC2_READ_NOT_BC2;
		}
		*blocks = reinterpret_cast<BC2Block*>(file);
		return BC2_READ_SUCCESS;
	}
	return BC2_READ_NOT_BC2;
}