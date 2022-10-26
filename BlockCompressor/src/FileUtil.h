#pragma once

#include <stdint.h>
#include <fstream>
typedef uint8_t byte;

//All this DDS stuff is from microsoft's DDS format documentation
enum DDSPixelFormatFlags {
	DDPF_ALPHAPIXELS = 0x1,
	DDPF_ALPHA = 0x2,
	DDPF_FOURCC = 0x4,
	DDPF_RGB = 0x40,
	DDPF_YUV = 0x200,
	DDPF_LUMINANCE = 0x20000
};
enum DDSFlags {
	DDSD_CAPS = 0x1,
	DDSD_HEIGHT = 0x2,
	DDSD_WIDTH = 0x4,
	DDSD_PITCH = 0x8,
	DDSD_PIXELFORMAT = 0x1000,
	DDSD_MIPMAPCOUNT = 0x20000,
	DDSD_LINEARSIZE = 0x80000,
	DDSD_DEPTH = 0x800000
};
enum D3D10ResourceDimension {
	D3D10_RESOURCE_DIMENSION_UNKNOWN = 0,
	D3D10_RESOURCE_DIMENSION_BUFFER = 1,
	D3D10_RESOURCE_DIMENSION_TEXTURE1D = 2,
	D3D10_RESOURCE_DIMENSION_TEXTURE2D = 3,
	D3D10_RESOURCE_DIMENSION_TEXTURE3D = 4,
	D3D10_RESOURCE_DIMENSION_FORCE_UINT = 0xffffffff
};
enum DDSAlphaMode {
	DDS_ALPHA_MODE_UNKNOWN = 0x0,
	DDS_ALPHA_MODE_STRAIGHT = 0x1,
	DDS_ALPHA_MODE_PREMULTIPLIED = 0x2,
	DDS_ALPHA_MODE_OPAQUE = 0x3,
	DDS_ALPHA_MODE_CUSTOM = 0x4
};
enum DXGIFormat {
	DXGI_FORMAT_UNKNOWN = 0,
	DXGI_FORMAT_R32G32B32A32_TYPELESS = 1,
	DXGI_FORMAT_R32G32B32A32_FLOAT = 2,
	DXGI_FORMAT_R32G32B32A32_UINT = 3,
	DXGI_FORMAT_R32G32B32A32_SINT = 4,
	DXGI_FORMAT_R32G32B32_TYPELESS = 5,
	DXGI_FORMAT_R32G32B32_FLOAT = 6,
	DXGI_FORMAT_R32G32B32_UINT = 7,
	DXGI_FORMAT_R32G32B32_SINT = 8,
	DXGI_FORMAT_R16G16B16A16_TYPELESS = 9,
	DXGI_FORMAT_R16G16B16A16_FLOAT = 10,
	DXGI_FORMAT_R16G16B16A16_UNORM = 11,
	DXGI_FORMAT_R16G16B16A16_UINT = 12,
	DXGI_FORMAT_R16G16B16A16_SNORM = 13,
	DXGI_FORMAT_R16G16B16A16_SINT = 14,
	DXGI_FORMAT_R32G32_TYPELESS = 15,
	DXGI_FORMAT_R32G32_FLOAT = 16,
	DXGI_FORMAT_R32G32_UINT = 17,
	DXGI_FORMAT_R32G32_SINT = 18,
	DXGI_FORMAT_R32G8X24_TYPELESS = 19,
	DXGI_FORMAT_D32_FLOAT_S8X24_UINT = 20,
	DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS = 21,
	DXGI_FORMAT_X32_TYPELESS_G8X24_UINT = 22,
	DXGI_FORMAT_R10G10B10A2_TYPELESS = 23,
	DXGI_FORMAT_R10G10B10A2_UNORM = 24,
	DXGI_FORMAT_R10G10B10A2_UINT = 25,
	DXGI_FORMAT_R11G11B10_FLOAT = 26,
	DXGI_FORMAT_R8G8B8A8_TYPELESS = 27,
	DXGI_FORMAT_R8G8B8A8_UNORM = 28,
	DXGI_FORMAT_R8G8B8A8_UNORM_SRGB = 29,
	DXGI_FORMAT_R8G8B8A8_UINT = 30,
	DXGI_FORMAT_R8G8B8A8_SNORM = 31,
	DXGI_FORMAT_R8G8B8A8_SINT = 32,
	DXGI_FORMAT_R16G16_TYPELESS = 33,
	DXGI_FORMAT_R16G16_FLOAT = 34,
	DXGI_FORMAT_R16G16_UNORM = 35,
	DXGI_FORMAT_R16G16_UINT = 36,
	DXGI_FORMAT_R16G16_SNORM = 37,
	DXGI_FORMAT_R16G16_SINT = 38,
	DXGI_FORMAT_R32_TYPELESS = 39,
	DXGI_FORMAT_D32_FLOAT = 40,
	DXGI_FORMAT_R32_FLOAT = 41,
	DXGI_FORMAT_R32_UINT = 42,
	DXGI_FORMAT_R32_SINT = 43,
	DXGI_FORMAT_R24G8_TYPELESS = 44,
	DXGI_FORMAT_D24_UNORM_S8_UINT = 45,
	DXGI_FORMAT_R24_UNORM_X8_TYPELESS = 46,
	DXGI_FORMAT_X24_TYPELESS_G8_UINT = 47,
	DXGI_FORMAT_R8G8_TYPELESS = 48,
	DXGI_FORMAT_R8G8_UNORM = 49,
	DXGI_FORMAT_R8G8_UINT = 50,
	DXGI_FORMAT_R8G8_SNORM = 51,
	DXGI_FORMAT_R8G8_SINT = 52,
	DXGI_FORMAT_R16_TYPELESS = 53,
	DXGI_FORMAT_R16_FLOAT = 54,
	DXGI_FORMAT_D16_UNORM = 55,
	DXGI_FORMAT_R16_UNORM = 56,
	DXGI_FORMAT_R16_UINT = 57,
	DXGI_FORMAT_R16_SNORM = 58,
	DXGI_FORMAT_R16_SINT = 59,
	DXGI_FORMAT_R8_TYPELESS = 60,
	DXGI_FORMAT_R8_UNORM = 61,
	DXGI_FORMAT_R8_UINT = 62,
	DXGI_FORMAT_R8_SNORM = 63,
	DXGI_FORMAT_R8_SINT = 64,
	DXGI_FORMAT_A8_UNORM = 65,
	DXGI_FORMAT_R1_UNORM = 66,
	DXGI_FORMAT_R9G9B9E5_SHAREDEXP = 67,
	DXGI_FORMAT_R8G8_B8G8_UNORM = 68,
	DXGI_FORMAT_G8R8_G8B8_UNORM = 69,
	DXGI_FORMAT_BC1_TYPELESS = 70,
	DXGI_FORMAT_BC1_UNORM = 71,
	DXGI_FORMAT_BC1_UNORM_SRGB = 72,
	DXGI_FORMAT_BC2_TYPELESS = 73,
	DXGI_FORMAT_BC2_UNORM = 74,
	DXGI_FORMAT_BC2_UNORM_SRGB = 75,
	DXGI_FORMAT_BC3_TYPELESS = 76,
	DXGI_FORMAT_BC3_UNORM = 77,
	DXGI_FORMAT_BC3_UNORM_SRGB = 78,
	DXGI_FORMAT_BC4_TYPELESS = 79,
	DXGI_FORMAT_BC4_UNORM = 80,
	DXGI_FORMAT_BC4_SNORM = 81,
	DXGI_FORMAT_BC5_TYPELESS = 82,
	DXGI_FORMAT_BC5_UNORM = 83,
	DXGI_FORMAT_BC5_SNORM = 84,
	DXGI_FORMAT_B5G6R5_UNORM = 85,
	DXGI_FORMAT_B5G5R5A1_UNORM = 86,
	DXGI_FORMAT_B8G8R8A8_UNORM = 87,
	DXGI_FORMAT_B8G8R8X8_UNORM = 88,
	DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM = 89,
	DXGI_FORMAT_B8G8R8A8_TYPELESS = 90,
	DXGI_FORMAT_B8G8R8A8_UNORM_SRGB = 91,
	DXGI_FORMAT_B8G8R8X8_TYPELESS = 92,
	DXGI_FORMAT_B8G8R8X8_UNORM_SRGB = 93,
	DXGI_FORMAT_BC6H_TYPELESS = 94,
	DXGI_FORMAT_BC6H_UF16 = 95,
	DXGI_FORMAT_BC6H_SF16 = 96,
	DXGI_FORMAT_BC7_TYPELESS = 97,
	DXGI_FORMAT_BC7_UNORM = 98,
	DXGI_FORMAT_BC7_UNORM_SRGB = 99,
	DXGI_FORMAT_AYUV = 100,
	DXGI_FORMAT_Y410 = 101,
	DXGI_FORMAT_Y416 = 102,
	DXGI_FORMAT_NV12 = 103,
	DXGI_FORMAT_P010 = 104,
	DXGI_FORMAT_P016 = 105,
	DXGI_FORMAT_420_OPAQUE = 106,
	DXGI_FORMAT_YUY2 = 107,
	DXGI_FORMAT_Y210 = 108,
	DXGI_FORMAT_Y216 = 109,
	DXGI_FORMAT_NV11 = 110,
	DXGI_FORMAT_AI44 = 111,
	DXGI_FORMAT_IA44 = 112,
	DXGI_FORMAT_P8 = 113,
	DXGI_FORMAT_A8P8 = 114,
	DXGI_FORMAT_B4G4R4A4_UNORM = 115,
	DXGI_FORMAT_P208 = 130,
	DXGI_FORMAT_V208 = 131,
	DXGI_FORMAT_V408 = 132,
	DXGI_FORMAT_SAMPLER_FEEDBACK_MIN_MIP_OPAQUE,
	DXGI_FORMAT_SAMPLER_FEEDBACK_MIP_REGION_USED_OPAQUE,
	DXGI_FORMAT_FORCE_UINT = 0xffffffff
};
enum DDSCAPS {
	DDSCAPS_COMPLEX = 0x8,
	DDSCAPS_MIPMAP = 0x400000,
	DDSCAPS_TEXTURE = 0x1000
};
enum DDSCAPS2 {
	DDSCAPS2_NONE = 0,
	DDSCAPS2_CUBEMAP = 0x200,
	DDSCAPS2_CUBEMAP_POSITIVEX = 0x400,
	DDSCAPS2_CUBEMAP_NEGATIVEX = 0x800,
	DDSCAPS2_CUBEMAP_POSITIVEY = 0x1000,
	DDSCAPS2_CUBEMAP_NEGATIVEY = 0x2000,
	DDSCAPS2_CUBEMAP_POSITIVEZ = 0x4000,
	DDSCAPS2_CUBEMAP_NEGATIVEZ = 0x8000,
	DDSCAPS2_VOLUME = 0x200000
};

#pragma pack(push, 1)
struct DDSPixelFormat {
	uint32_t size;
	DDSPixelFormatFlags flags;
	char fourCC[4];
	uint32_t rgbBitCount;
	uint32_t rBitMask;
	uint32_t gBitMask;
	uint32_t bBitMask;
	uint32_t aBitMask;
};
struct DDSHeader {
	uint32_t size;
	DDSFlags flags;
	uint32_t height;
	uint32_t width;
	uint32_t pitchOrLinearSize;
	uint32_t depth;
	uint32_t mipCount;
	uint32_t reserved[11];
	DDSPixelFormat pixelFormat;
	DDSCAPS caps;
	DDSCAPS2 caps2;
	uint32_t caps3;
	uint32_t caps4;
	uint32_t reserved2;
};
enum D3D11_RESOURCE_MISC_FLAG {
	D3D11_RESOURCE_MISC_GENERATE_MIPS = 0x1L,
	D3D11_RESOURCE_MISC_SHARED = 0x2L,
	D3D11_RESOURCE_MISC_TEXTURECUBE = 0x4L,
	D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS = 0x10L,
	D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS = 0x20L,
	D3D11_RESOURCE_MISC_BUFFER_STRUCTURED = 0x40L,
	D3D11_RESOURCE_MISC_RESOURCE_CLAMP = 0x80L,
	D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX = 0x100L,
	D3D11_RESOURCE_MISC_GDI_COMPATIBLE = 0x200L,
	D3D11_RESOURCE_MISC_SHARED_NTHANDLE = 0x800L,
	D3D11_RESOURCE_MISC_RESTRICTED_CONTENT = 0x1000L,
	D3D11_RESOURCE_MISC_RESTRICT_SHARED_RESOURCE = 0x2000L,
	D3D11_RESOURCE_MISC_RESTRICT_SHARED_RESOURCE_DRIVER = 0x4000L,
	D3D11_RESOURCE_MISC_GUARDED = 0x8000L,
	D3D11_RESOURCE_MISC_TILE_POOL = 0x20000L,
	D3D11_RESOURCE_MISC_TILED = 0x40000L,
	D3D11_RESOURCE_MISC_HW_PROTECTED = 0x80000L,
	D3D11_RESOURCE_MISC_SHARED_DISPLAYABLE,
	D3D11_RESOURCE_MISC_SHARED_EXCLUSIVE_WRITER
};
struct DDSHeaderDX10 {
	DXGIFormat dxgiFormat;
	D3D10ResourceDimension resourceDimension;
	uint32_t miscFlag;
	uint32_t arraySize;
	//Lower 3 bits are alpha mode
	uint32_t miscFlags2;
};
#pragma pack(pop)

inline byte* read_file_to_bytes(const char* fileName) {
	std::ifstream file{};
	file.open(fileName, std::ios::ate | std::ios::binary);
	if (!file.is_open()) {
		throw std::runtime_error("Failed to open file!");
	}
	size_t fileSize = file.tellg();
	file.seekg(0);
	byte* bytes = new byte[fileSize];
	file.read(reinterpret_cast<char*>(bytes), fileSize);
	file.close();
	return bytes;
}

enum BlockCompressFormat {
	BLOCK_COMPRESS_FORMAT_BC1,
	BLOCK_COMPRESS_FORMAT_BC2,
	BLOCK_COMPRESS_FORMAT_BC3,
	BLOCK_COMPRESS_FORMAT_BC4,
	BLOCK_COMPRESS_FORMAT_BC5,
	BLOCK_COMPRESS_FORMAT_BC6H,
	BLOCK_COMPRESS_FORMAT_BC7
};

void fill_dx10_dds_header_bc(DDSHeaderDX10& header, DXGIFormat format, DDSAlphaMode alphaMode) {
	header.dxgiFormat = format;
	header.resourceDimension = D3D10_RESOURCE_DIMENSION_TEXTURE2D;
	header.miscFlag = 0;
	header.arraySize = 1;
	header.miscFlags2 = alphaMode;
}

#include <string>

void fill_dds_header_bc(DDSHeader& header, DDSHeaderDX10& dx10Header, BlockCompressFormat format, uint32_t width, uint32_t height) {
	uint32_t linearSize = 0;
	const char* fourCC = nullptr;
	uint32_t blockCount = ((width + 3) / 4) * ((height + 3) / 4);
	dx10Header.dxgiFormat = DXGI_FORMAT_UNKNOWN;

	switch (format) {
	case BLOCK_COMPRESS_FORMAT_BC1:
		linearSize = 8 * blockCount;
		fourCC = "DXT1";
		break;
	case BLOCK_COMPRESS_FORMAT_BC2:
		linearSize = 16 * blockCount;
		fourCC = "DXT3";
		break;
	case BLOCK_COMPRESS_FORMAT_BC3:
		linearSize = 16 * blockCount;
		fourCC = "DXT5";
		break;
	case BLOCK_COMPRESS_FORMAT_BC4:
		linearSize = 8 * blockCount;
		fourCC = "DX10";
		fill_dx10_dds_header_bc(dx10Header, DXGI_FORMAT_BC4_TYPELESS, DDS_ALPHA_MODE_OPAQUE);
		break;
	case BLOCK_COMPRESS_FORMAT_BC5:
		linearSize = 16 * blockCount;
		fourCC = "DX10";
		fill_dx10_dds_header_bc(dx10Header, DXGI_FORMAT_BC5_TYPELESS, DDS_ALPHA_MODE_OPAQUE);
		break;
	case BLOCK_COMPRESS_FORMAT_BC7:
		linearSize = 16 * blockCount;
		fourCC = "DX10";
		fill_dx10_dds_header_bc(dx10Header, DXGI_FORMAT_BC7_TYPELESS, DDS_ALPHA_MODE_OPAQUE);
		break;
	default:
		//Not implemented
		return;
	}
	header.size = 124;
	header.flags = static_cast<DDSFlags>(DDSD_CAPS | DDSD_WIDTH | DDSD_HEIGHT | DDSD_PIXELFORMAT | DDSD_LINEARSIZE);
	header.width = width;
	header.height = height;
	header.pitchOrLinearSize = linearSize;
	header.depth = 1;
	header.mipCount = 1;
	memset(header.reserved, 0, 11 * sizeof(uint32_t));

	header.pixelFormat.size = 32;
	header.pixelFormat.flags = DDPF_FOURCC;
	memcpy(header.pixelFormat.fourCC, fourCC, 4);
	//None of these are applicable for compressed formats
	header.pixelFormat.rgbBitCount = 0;
	header.pixelFormat.rBitMask = 0;
	header.pixelFormat.gBitMask = 0;
	header.pixelFormat.bBitMask = 0;
	header.pixelFormat.aBitMask = 0;

	header.caps = DDSCAPS_TEXTURE;
	header.caps2 = DDSCAPS2_NONE;
	header.caps3 = 0;
	header.caps4 = 0;
	header.reserved2 = 0;
}

void write_dds_file_bc(const char* fileName, void* blocks, uint32_t width, uint32_t height, BlockCompressFormat format) {
	std::ofstream file{};
	file.open(fileName, std::ios::binary);
	if (!file.is_open()) {
		throw std::runtime_error("Failed to open file!");
	}
	file.write("DDS ", 4);
	DDSHeader header;
	DDSHeaderDX10 dx10Header;
	fill_dds_header_bc(header, dx10Header, format, width, height);
	file.write(reinterpret_cast<char*>(&header), sizeof(DDSHeader));
	if (dx10Header.dxgiFormat != DXGI_FORMAT_UNKNOWN) {
		file.write(reinterpret_cast<char*>(&dx10Header), sizeof(DDSHeaderDX10));
	}
	file.write(reinterpret_cast<char*>(blocks), header.pitchOrLinearSize);
}