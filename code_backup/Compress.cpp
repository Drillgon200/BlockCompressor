#include <iostream>
#include <fstream>
#include <stdint.h>
#ifdef _WIN32
#pragma comment(lib, "Ws2_32.lib")
#define NOMINMAX
#include <winsock.h>
#else
#include <arpa/inet.h>
#endif
#include <assert.h>
extern "C" {
#include "puff.h"
}
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

struct BigEndianDataWrapper {
	byte* data;

	uint32_t read_uint32() {
		uint32_t num = *reinterpret_cast<uint32_t*>(data);
		data += sizeof(uint32_t);
		return ntohl(num);
	}
	uint16_t read_uint16() {
		uint16_t num = *reinterpret_cast<uint16_t*>(data);
		data += sizeof(uint16_t);
		return ntohs(num);
	}
	uint8_t read_uint8() {
		uint8_t num = *data;
		data += sizeof(uint8_t);
		return num;
	}
	float read_float() {
		uint32_t num = *reinterpret_cast<uint32_t*>(data);
		data += sizeof(uint32_t);
		num = ntohl(num);
		return *reinterpret_cast<float*>(&num);
	}
};

//Based on what I remember from some guy's game networking article
struct BitReader {
	uint64_t scratchValue;
	uint32_t* data;
	uint32_t numBits;

	BitReader(void* input) {
		byte* bytes = reinterpret_cast<byte*>(input);
		uint32_t byteReadCount = ((4 - (reinterpret_cast<uintptr_t>(bytes) & 3)) + 4);
		numBits = 0;
		scratchValue = 0;
		for (uint32_t i = 0; i < byteReadCount; i++) {
			scratchValue = scratchValue | (static_cast<uint64_t>(*bytes) << numBits);
			numBits += 8;
			bytes++;
		}
		data = reinterpret_cast<uint32_t*>(bytes);
		//read the first two dwords into the scratch value
		//data = reinterpret_cast<uint32_t*>(input);
		//scratchValue = (static_cast<uint64_t>(data[1]) << 32) | data[0];
		//data += 2;
		//numBits = 64;
	}

	void try_read_next() {
		if (numBits <= 32) {
			scratchValue = (static_cast<uint64_t>(*data++) << numBits) | scratchValue;
			numBits += 32;
		}
		/*uint64_t test = numBits <= 32;
		uint64_t mask = ~test + 1;
		scratchValue = ((static_cast<uint64_t>(*data) << numBits) & mask) | scratchValue;
		data += test;
		numBits += test << 5;*/
	}

	//Assumes byte alignment. Call align_to_byte if unsure.
	inline void increase_byte_pos(uint32_t amount) {
		assert((numBits & 0b111) == 0 && "Bits not aligned! Data could be lost");
		//Clear the scratch value, restore those bytes to the data block, increase the data block pointer by specified bytes.
		uint32_t numBytesInBuffer = (numBits + 7) / 8;
		byte* newData = reinterpret_cast<byte*>(data);
		newData -= numBytesInBuffer;
		newData += amount;
		data = reinterpret_cast<uint32_t*>(newData);
		//Put data back into the scratch block
		scratchValue = (static_cast<uint64_t>(data[1]) << 32) | data[0];
		data += 2;
		numBits = 64;
	}

	inline void align_to_byte() {
		//Take off the lowest 3 bits, 0-7, to align to 8 bits
		uint32_t oldNumBits = numBits;
		numBits &= ~0B111;
		scratchValue = scratchValue >> (oldNumBits - numBits);
		try_read_next();
	}

	inline byte* get_current_pointer() {
		uint32_t numBytesInBuffer = (numBits + 7) / 8;
		return reinterpret_cast<byte*>(data) - numBytesInBuffer;
	}

	inline uint32_t read_bits(uint32_t bitCount) {
		numBits -= bitCount;
		uint32_t val = static_cast<uint32_t>(scratchValue) & ((1 << bitCount) - 1);
		scratchValue = scratchValue >> bitCount;
		try_read_next();
		return val;
	}

	inline uint32_t read_bits_no_check(uint32_t bitCount) {
		numBits -= bitCount;
		uint32_t val = static_cast<uint32_t>(scratchValue & ((1 << bitCount) - 1));
		scratchValue = scratchValue >> bitCount;
		return val;
	}

	inline void put_back_bits(uint32_t bits, uint32_t bitCount) {
		numBits += bitCount;
		scratchValue = (scratchValue << bitCount) | (bits & ((1 << bitCount) - 1));
	}

	inline uint8_t read_uint8() {
		numBits -= 8;
		uint8_t val = static_cast<uint8_t>(scratchValue & ((~0ULL) >> 56));
		scratchValue = scratchValue >> 8;
		try_read_next();
		return val;
	}

	inline uint16_t read_uint16() {
		numBits -= 16;
		uint16_t val = static_cast<uint16_t>(scratchValue & ((~0ULL) >> 48));
		scratchValue = scratchValue >> 16;
		try_read_next();
		return val;
	}

	inline uint32_t read_uint32() {
		return read_bits(32);
	}
};

struct HuffmanTree {
	constexpr static uint32_t MAX_BIT_LENGTH = 15;
	//uint16_t lookupTable[32768];
	uint16_t lengthCounts[MAX_BIT_LENGTH+1];
	uint16_t* symbols;

	inline uint16_t read_next(BitReader& reader) {
		/*uint16_t bits = reader.read_bits_no_check(MAX_BIT_LENGTH);
		uint16_t value = lookupTable[bits];
		uint16_t length = value >> 12;
		value &= 0xFFF;
		reader.put_back_bits(bits >> length, MAX_BIT_LENGTH-length);
		reader.try_read_next();
		return value;*/
		int32_t symbolIndex = 0;
		int32_t firstCodeForLength = 0;
		int32_t huffmanCode = 0;
		
		for (int32_t treeLevel = 1; treeLevel <= MAX_BIT_LENGTH; treeLevel++) {
			int32_t bit = reader.read_bits(1);
			huffmanCode = (huffmanCode << 1) | bit;
			int32_t count = lengthCounts[treeLevel];
			if ((huffmanCode - count) < firstCodeForLength) {
				int32_t index = symbolIndex + (huffmanCode - firstCodeForLength);
				return symbols[index];
			}
			symbolIndex += count;
			firstCodeForLength = (firstCodeForLength + count) << 1;
		}
		constexpr uint16_t error = ~0;
		assert(false && "Huffman decode failed");
		return error;
	}

	uint16_t reverse_bits(uint16_t bits) {
		bits = ((bits & 0x00FF) << 8) | ((bits & 0xFF00) >> 8);
		bits = ((bits & 0x0F0F) << 4) | ((bits & 0xF0F0) >> 4);
		bits = ((bits & 0x3333) << 2) | ((bits & 0xCCCC) >> 2);
		bits = ((bits & 0x5555) << 1) | ((bits & 0xAAAA) >> 1);
		return bits;
	}

	void build(uint8_t* valueLengths, uint32_t numValues) {
		memset(lengthCounts, 0, (MAX_BIT_LENGTH + 1) * sizeof(uint16_t));
		for (uint32_t i = 0; i < numValues; i++) {
			lengthCounts[valueLengths[i]]++;
		}
		lengthCounts[0] = 0;
		uint32_t offsets[MAX_BIT_LENGTH + 2];
		offsets[0] = 0;
		offsets[1] = 0;
		//uint16_t code = 0;
		//uint32_t nextCode[MAX_BIT_LENGTH+1];
		for (uint32_t i = 1; i <= MAX_BIT_LENGTH; i++) {
			offsets[i+1] = offsets[i] + lengthCounts[i];
			//code = (code + lengthCounts[i - 1]) << 1;
			//nextCode[i] = code;
		}
		symbols = reinterpret_cast<uint16_t*>(malloc(offsets[MAX_BIT_LENGTH+1] * sizeof(uint16_t)));
		for (uint16_t value = 0; value < numValues; value++) {
			uint8_t valLength = valueLengths[value];
			if (valLength > 0) {
				/*uint16_t huffman = nextCode[valLength] << (MAX_BIT_LENGTH - valLength);
				nextCode[valLength]++;
				for (uint32_t fill = 0; fill < (1 << (MAX_BIT_LENGTH - valLength)); fill++) {
					uint32_t lookupIndex = reverse_bits(huffman + fill) >> 1;
					lookupTable[lookupIndex] = (valLength << 12) | value;
				}*/

				uint32_t offset = offsets[valLength];
				symbols[offset] = value;
				offsets[valLength]++;
			}
		}

	}

	void destroy() {
		free(symbols);
	}
};

enum CompressionMethod {
	COMPRESSION_METHOD_NONE = 0,
	COMPRESSION_METHOD_DEFLATE = 8,
	COMPRESSION_METHOD_RESERVED = 15
};

enum CompressionLevel {
	COMPRESSION_LEVEL_FASTEST = 0,
	COMPRESSION_LEVEL_FAST = 1,
	COMPRESSION_LEVEL_DEFAULT = 2,
	COMPRESSION_LEVEL_MAXIMUM = 3
};

enum DeflateCompressionType {
	//No compression, stored as is
	DEFLATE_COMPRESSION_NONE = 0,
	//Compressed with a fixed huffman tree defined by spec
	DEFLATE_COMPRESSION_FIXED = 1,
	//Huffman tree is stored as well
	DEFLATE_COMPRESSION_DYNAMIC = 2,
	//Reserved for future use, error
	DEFLATE_COMPRESSION_RESERVED = 3
};

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

//Adler32 as described by the zlib format spec
inline uint32_t adler32(byte* data, uint32_t length) {
	uint32_t s1 = 1;
	uint32_t s2 = 0;
	for (uint32_t i = 0; i < length; i++) {
		//The mod could be optimized and run only every 5552 bytes according to spec. I don't feel like doing that right now.
		s1 = (s1 + data[i]) % 65521;
		s2 = (s2 + s1) % 65521;
	}
	return (s2 << 16) | s1;
}

//CRC code based on the lib png sample
uint32_t crcTable[256];

inline void compute_crc_table() {
	for (uint32_t n = 0; n < 256; n++) {
		uint32_t val = n;
		for (uint32_t k = 0; k < 8; k++) {
			if (val & 1) {
				val = 0xEDB88320UL ^ (val >> 1);
			} else {
				val = val >> 1;
			}
		}
		crcTable[n] = val;
	}
}

inline uint32_t crc32(byte* data, uint32_t length) {
	uint64_t crc = 0xFFFFFFFFUL;
	for (uint32_t i = 0; i < length; i++) {
		crc = crcTable[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
	}
	return crc ^ 0xFFFFFFFFUL;
}

inline void resize_buffer(byte** buffer, uint32_t* size, uint32_t usedBytes, uint32_t accomodateSize) {
	if (accomodateSize < *size) {
		return;
	}
	while (accomodateSize >= *size) {
		*size = *size * 1.5;
	}
	*buffer = reinterpret_cast<byte*>(realloc(*buffer, *size));
}

void generate_fixed_tree(HuffmanTree& litlenFixedTree, HuffmanTree& distFixedTree) {
	uint8_t litLenValueLengths[288];
	memset(litLenValueLengths, 8, 144);
	memset(litLenValueLengths + 144, 9, 112);
	memset(litLenValueLengths + 256, 7, 24);
	memset(litLenValueLengths + 280, 8, 8);
	litlenFixedTree.build(litLenValueLengths, 288);

	uint8_t distVals[30];
	memset(distVals, 5, 30);
	distFixedTree.build(distVals, 30);
}

void read_tree_lengths(BitReader& reader, HuffmanTree& decompressTree, uint32_t numCodes, uint8_t* codeLengths) {
	constexpr uint32_t COPY3_6 = 16;
	constexpr uint32_t ZERO3_10 = 17;
	constexpr uint32_t ZERO11_138 = 18;

	for (uint32_t codeIdx = 0; codeIdx < numCodes;) {
		uint8_t length = static_cast<uint8_t>(decompressTree.read_next(reader));
		if (length <= 15) {
			codeLengths[codeIdx++] = length;
		} else if (length == COPY3_6) {
			uint8_t copy = codeLengths[codeIdx - 1];
			uint32_t copyCount = 3 + reader.read_bits(2);
			for (uint32_t j = 0; j < copyCount; j++) {
				codeLengths[codeIdx++] = copy;
			}
		} else if (length == ZERO3_10) {
			uint32_t zeroCount = 3 + reader.read_bits(3);
			for (uint32_t j = 0; j < zeroCount; j++) {
				codeLengths[codeIdx++] = 0;
			}
		} else if (length == ZERO11_138) {
			uint32_t zeroCount = 11 + reader.read_bits(7);
			for (uint32_t j = 0; j < zeroCount; j++) {
				codeLengths[codeIdx++] = 0;
			}
		} else {
			assert(false && "Bad length!");
		}
	}
}

void decompress_trees(BitReader& reader, HuffmanTree* outLitLen, HuffmanTree* outDist) {
	constexpr uint32_t maxHLit = 286;
	constexpr uint32_t minHLit = 257;
	constexpr uint32_t maxHDist = 32;
	constexpr uint32_t minHDistn = 1;
	constexpr uint32_t maxHCLen = 19;
	constexpr uint32_t minHCLen = 4;

	uint32_t hlit = reader.read_bits(5) + 257;
	uint32_t hdist = reader.read_bits(5) + 1;
	uint32_t hclen = reader.read_bits(4) + 4;
	assert(hlit <= maxHLit && "HLIT out of range!");
	assert(hdist <= maxHDist && "HDIST out of range!");
	assert(hclen <= maxHCLen && "HCLEN out of range!");

	constexpr uint32_t codeLengthOrder[maxHCLen]{16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
	uint8_t lengths[maxHCLen]{};
	for (uint32_t i = 0; i < hclen; i++) {
		lengths[codeLengthOrder[i]] = reader.read_bits(3);
	}
	HuffmanTree decompressTree;
	decompressTree.build(lengths, maxHCLen);

	uint8_t litlenCodeLengths[maxHLit];
	read_tree_lengths(reader, decompressTree, hlit, litlenCodeLengths);

	uint8_t distCodeLengths[maxHDist];
	read_tree_lengths(reader, decompressTree, hdist, distCodeLengths);

	outLitLen->build(litlenCodeLengths, hlit);
	outDist->build(distCodeLengths, hdist);

	decompressTree.destroy();
}

byte* inflate(byte* data, byte** result, uint32_t* resultLength, uint32_t defaultBufferSize, HuffmanTree& litlenFixedTree, HuffmanTree& distFixedTree) {
	BitReader reader{ data };
	bool finalBlock = false;
	uint32_t bufferSize = defaultBufferSize;
	uint32_t decompressedSize = 0;
	byte* decompressedOutput = reinterpret_cast<byte*>(malloc(bufferSize));

	constexpr uint32_t extraBitLengthTable[29]{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0 };
	constexpr uint32_t startingLengthTable[29]{ 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258 };
	constexpr uint32_t extraBitDistTable[30]{ 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13 };
	constexpr uint32_t startingDistTable[30]{ 1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577 };
	constexpr uint32_t endOfBlock = 256;

	//uint64_t huffmanReadTime = 0;
	//uint64_t decompressTime = 0;
	//uint64_t totalReadTime = 0;
	while (!finalBlock) {
		finalBlock = reader.read_bits(1);

		DeflateCompressionType compressionType = static_cast<DeflateCompressionType>(reader.read_bits(2));
		assert(compressionType != DEFLATE_COMPRESSION_RESERVED && "Bad compression type! This value is reserved");
		if (compressionType == DEFLATE_COMPRESSION_NONE) {
			//Skip remaining bits in current processing byte
			reader.align_to_byte();
			//Read LEN and NLEN
			uint16_t len = reader.read_uint16();
			uint16_t nlen = reader.read_uint16();
			assert(len == (~nlen & 0xFFFF) && "Data corruption? Length and the one's complement of length are not inverse of each other");
			byte* storedData = reader.get_current_pointer();
			reader.increase_byte_pos(len);
			resize_buffer(&decompressedOutput, &bufferSize, decompressedSize, decompressedSize + len);
			//Copy LEN bytes of data to output
			memcpy(decompressedOutput + decompressedSize, storedData, len);
			decompressedSize += len;
		} else if (compressionType == DEFLATE_COMPRESSION_FIXED || compressionType == DEFLATE_COMPRESSION_DYNAMIC) {
			HuffmanTree litLenTree;
			HuffmanTree distTree;
			HuffmanTree litLenTreeRef = litlenFixedTree;
			HuffmanTree distTreeRef = distFixedTree;
			//uint64_t time1 = __rdtsc();
			if (compressionType == DEFLATE_COMPRESSION_DYNAMIC) {
				decompress_trees(reader, &litLenTree, &distTree);
				litLenTreeRef = litLenTree;
				distTreeRef = distTree;
			}
			//decompressTime += __rdtsc() - time1;

			
			//uint64_t time2 = __rdtsc();
			while (true) {
				//uint64_t time = __rdtsc();
				uint16_t value = litLenTreeRef.read_next(reader);
				//huffmanReadTime += __rdtsc() - time;
				if (value == endOfBlock) {
					break;
				} else if (value < 256) {
					resize_buffer(&decompressedOutput, &bufferSize, decompressedSize, decompressedSize + 1);
					decompressedOutput[decompressedSize++] = static_cast<uint8_t>(value);
				} else if (value < 286) {
					value -= 257;
					uint32_t extraBits = extraBitLengthTable[value];
					uint32_t length = startingLengthTable[value] + reader.read_bits(extraBits);
					//time = __rdtsc();
					uint32_t distance = distTreeRef.read_next(reader);
					//huffmanReadTime += __rdtsc() - time;
					assert(distance < 30 && "Distance read out of range");
					extraBits = extraBitDistTable[distance];
					distance = startingDistTable[distance] + reader.read_bits(extraBits);
					resize_buffer(&decompressedOutput, &bufferSize, decompressedSize, decompressedSize + length);
					uint32_t start = decompressedSize - distance;
					for (uint32_t i = 0; i < length; i++) {
						decompressedOutput[decompressedSize + i] = decompressedOutput[start + i];
					}
					decompressedSize += length;
				} else {
					assert(false && "Read wrong value!");
				}
			}
			//totalReadTime += __rdtsc() - time2;

			if (compressionType == DEFLATE_COMPRESSION_DYNAMIC) {
				litLenTree.destroy();
				distTree.destroy();
			}
		}
	}

	//std::cout << "Huffman read time: " << huffmanReadTime << "\nDecompress time: " << decompressTime << "\nTotal read time: " << totalReadTime << "\n" << std::endl;

	*result = decompressedOutput;
	*resultLength = decompressedSize;

	reader.align_to_byte();
	return reader.get_current_pointer();
}

enum InterlaceMethod {
	NONE = 0,
	ADAM7 = 1
};

enum FilterMode {
	FILTER_MODE_NONE = 0,
	FILTER_MODE_SUB = 1,
	FILTER_MODE_UP = 2,
	FILTER_MODE_AVERAGE = 3,
	FILTER_MODE_PAETH = 4
};

struct RGB {
	uint8_t r;
	uint8_t g;
	uint8_t b;

	RGB operator+(RGB other) {
		return RGB{ static_cast<uint8_t>(r + other.r), static_cast<uint8_t>(g + other.g), static_cast<uint8_t>(b + other.b) };
	}
	RGB operator-(RGB other) {
		return RGB{ static_cast<uint8_t>(r - other.r), static_cast<uint8_t>(g - other.g), static_cast<uint8_t>(b - other.b) };
	}
};

struct RGBA {
	uint8_t r;
	uint8_t g;
	uint8_t b;
	uint8_t a;

	RGBA operator+(RGBA other) {
		return RGBA{ static_cast<uint8_t>(r + other.r), static_cast<uint8_t>(g + other.g), static_cast<uint8_t>(b + other.b), static_cast<uint8_t>(a + other.a) };
	}
	RGBA operator-(RGBA other) {
		return RGBA{ static_cast<uint8_t>(r - other.r), static_cast<uint8_t>(g - other.g), static_cast<uint8_t>(b - other.b), static_cast<uint8_t>(a - other.a) };
	}
};

struct ImageHeader {
	uint32_t width;
	uint32_t height;
	uint8_t bitDepth;
	bool hasPalette;
	bool hasColor;
	bool hasAlpha;
	InterlaceMethod interlace;
};

void read_ihdr(BigEndianDataWrapper& ihdr, ImageHeader& header) {
	header.width = ihdr.read_uint32();
	header.height = ihdr.read_uint32();
	header.bitDepth = ihdr.read_uint8();

	uint8_t colorType = ihdr.read_uint8();
	constexpr uint8_t allowedBitDepths[5]{ 1, 2, 4, 8, 16 };
	uint8_t bitCheckRange[2]{ 0, 5 };
	switch (colorType) {
	case 0: break;
	case 2: bitCheckRange[0] = 3; break;
	case 3: bitCheckRange[1] = 4; break;
	case 4: bitCheckRange[0] = 3; break;
	case 6: bitCheckRange[0] = 3; break;
	default: assert(false && "Color type is wrong!"); break;
	}
	bool bitDepthVerified = false;
	for (uint32_t i = bitCheckRange[0]; i < bitCheckRange[1]; i++) {
		if (header.bitDepth == allowedBitDepths[i]) {
			bitDepthVerified = true;
			break;
		}
	}
	assert(bitDepthVerified && "Bit depth not acceptable for color type!");

	header.hasPalette = colorType & 1;
	header.hasColor = colorType & 2;
	header.hasAlpha = colorType & 4;

	uint8_t compressionMethod = ihdr.read_uint8();
	assert(compressionMethod == 0 && "Compression method isn't deflate!");
	uint8_t filterMethod = ihdr.read_uint8();
	assert(filterMethod == 0 && "Filter method isn't recognized!");
	header.interlace = static_cast<InterlaceMethod>(ihdr.read_uint8());
}

inline uint8_t paeth_predictor(uint8_t left , uint8_t up, uint8_t upLeft) {
	int32_t initialEstimate = left + up - upLeft;
	int32_t distLeft = abs(initialEstimate - left);
	int32_t distUp = abs(initialEstimate - up);
	int32_t distUpLeft = abs(initialEstimate - upLeft);
	if (distLeft <= distUp && distLeft <= distUpLeft) {
		return left;
	} else if (distUp <= distUpLeft) {
		return up;
	} else {
		return upLeft;
	}
}

//Index 0 is the regular image format, transmitted line by line. Indices 1-8 represent the 7 interlaced passes
constexpr uint32_t interlaceXOffset[8]{ 0, 0, 4, 0, 2, 0, 1, 0 };
constexpr uint32_t interlaceYOffset[8]{ 0, 0, 0, 4, 0, 2, 0, 1 };
constexpr uint32_t interlaceXStride[8]{ 1, 8, 8, 4, 4, 2, 2, 1 };
constexpr uint32_t interlaceYStride[8]{ 1, 8, 8, 8, 4, 4, 2, 2 };

inline byte sample_line(byte* line, uint32_t lineWidth, uint32_t x) {
	//Greater than or equals handles less than zero as well due to unsigned math
	if (line == nullptr || x >= lineWidth) {
		return 0;
	}
	return line[x];
}

byte* translate_pass(ImageHeader& header, byte* data, uint32_t dataSize, byte* finalData, uint32_t pass, uint32_t numComponents, uint32_t bytesPerPixel, uint32_t pixelsPerByte) {
	uint32_t passWidth = (header.width - interlaceXOffset[pass]  + (interlaceXStride[pass]-1)) / interlaceXStride[pass];
	uint32_t passHeight = (header.height - interlaceYOffset[pass] + (interlaceYStride[pass]-1)) / interlaceYStride[pass];
	uint32_t bytesPerLine = (passWidth * numComponents * header.bitDepth + 7) / 8;

	byte* previousLine = nullptr;
	for (uint32_t y = 0; y < passHeight; y++) {
		FilterMode filterMode = static_cast<FilterMode>(*data++);
		assert(filterMode < 5 && "Filter mode out of range");
		for (uint32_t xByte = 0; xByte < bytesPerLine; xByte++) {
			byte currentByte = data[xByte];

			switch (filterMode) {
			case FILTER_MODE_NONE:
				break;
			case FILTER_MODE_SUB:
				currentByte = currentByte + sample_line(finalData, bytesPerLine, xByte - bytesPerPixel);
				break;
			case FILTER_MODE_UP:
				currentByte = currentByte + sample_line(previousLine, bytesPerLine, xByte);
				break;
			case FILTER_MODE_AVERAGE:
			{
				byte left = sample_line(finalData, bytesPerLine, xByte - bytesPerPixel);
				byte up = sample_line(previousLine, bytesPerLine, xByte);
				currentByte = currentByte + static_cast<uint8_t>((left + up) / 2);
			}
			break;
			case FILTER_MODE_PAETH:
			{
				byte left = sample_line(finalData, bytesPerLine, xByte - bytesPerPixel);
				byte up = sample_line(previousLine, bytesPerLine, xByte);
				byte upLeft = sample_line(previousLine, bytesPerLine, xByte - bytesPerPixel);
				currentByte = currentByte + paeth_predictor(left, up, upLeft);
			}
			break;
			}
			finalData[xByte] = currentByte;
		}
		data += bytesPerLine;
		previousLine = finalData;
		finalData += bytesPerLine;
	}
	return data;
}

void rescale_bit_depth(ImageHeader& header, byte* finalData, RGBA* finalImage, uint32_t pass, uint32_t numComponents, uint32_t bytesPerPixel, uint32_t pixelsPerByte, RGB* palette, uint32_t paletteEntries, byte* transparency, uint32_t transparencyEntries) {
	uint32_t passWidth = (header.width - interlaceXOffset[pass] + (interlaceXStride[pass]-1)) / interlaceXStride[pass];
	uint32_t passHeight = (header.height - interlaceYOffset[pass] + (interlaceYStride[pass]-1)) / interlaceYStride[pass];
	uint32_t paletteIndexMask = header.hasPalette ? 0x00000000 : 0xFFFFFFFF;
	uint32_t bytesPerLine = (passWidth * numComponents * header.bitDepth + 7) / 8;

	//Transparency but not null;
	byte transparencyDummyData[8];
	byte* checkTransparency = transparency ? transparency : transparencyDummyData;
	for (uint32_t y = 0; y < passHeight; y++) {
		uint32_t finalY = y * interlaceYStride[pass] + interlaceYOffset[pass];
		//uint32_t lineOffset = (y * bytesPerPixel * passWidth + pixelsPerByte-1) / pixelsPerByte;
		byte* line = finalData + bytesPerLine * y;
		for (uint32_t x = 0; x < passWidth; x++) {
			uint32_t finalX = x * interlaceXStride[pass] + interlaceXOffset[pass];

			uint8_t rescaledComponents[4];
			bool transparencyCheck = true;
			uint32_t test1;
			uint32_t test2;
			//Read up to 4 components into the array
			for (uint32_t i = 0; i < numComponents; i++) {
				uint32_t element = (x * numComponents + i);
				uint32_t xByte = (element / pixelsPerByte) * bytesPerPixel;
				
				switch (header.bitDepth) {
				case 16:
					//Only use most significant byte
					rescaledComponents[i] = line[element * 2];
					transparencyCheck &= line[element * 2] == checkTransparency[i * 2] && line[element * 2 + 1] == checkTransparency[i * 2 + 1];
					break;
				case 8:
					//pass through
					rescaledComponents[i] = line[element];
					transparencyCheck &= line[element] == checkTransparency[i];
					break;
				case 4:
					rescaledComponents[i] = (line[element / 2] >> ((1-(element & 1)) * 4)) & 15;
					test1 = (line[element / 2] >> 4) & 15;
					test2 = line[element / 2] & 15;
					transparencyCheck &= rescaledComponents[i] == checkTransparency[i];
					//Put the bits in both bottom and top half of byte, that way 0 maps to 0 and 15 maps to 255.
					rescaledComponents[i] |= (rescaledComponents[i] << 4) & paletteIndexMask;
					break;
				case 2:
					rescaledComponents[i] = (line[element/4] >> ((3-(element & 3)) * 2)) & 3;
					transparencyCheck &= rescaledComponents[i] == checkTransparency[i];
					//Repeat the 2 bits 4 times in the byte, same reason as above
					rescaledComponents[i] |= (rescaledComponents[i] << 2) & paletteIndexMask;
					rescaledComponents[i] |= (rescaledComponents[i] << 4) & paletteIndexMask;
					break;
				case 1:
					//Extract the bit and choose 255 or 0.
				{
					uint32_t ele = line[element / 8];
					uint32_t shift = (7 - (element & 7));
					rescaledComponents[i] = ((line[element / 8] >> (7 - (element & 7))) & 1) * (1 + 254 * (paletteIndexMask > 0));
					transparencyCheck &= (rescaledComponents[i] > 0) == checkTransparency[i];
				}
					break;
				}
			}
			//Map those components to the RGBA final output.
			uint32_t finalidx = finalY * header.width + finalX;
			if (header.hasPalette) {
				uint8_t index = rescaledComponents[0];
				assert(index < paletteEntries);
				RGB color = palette[index];
				uint8_t alpha;
				if (index < transparencyEntries) {
					alpha = transparency[index];
				} else {
					alpha = 255;
				}
				finalImage[finalidx] = RGBA{ color.r, color.g, color.b, alpha };
			} else {
				uint32_t numComponentsWithExtraAlpha = numComponents + (transparency != nullptr);
				if (transparency) {
					rescaledComponents[numComponentsWithExtraAlpha - 1] = (!!transparencyCheck) * 255;
				}
				switch (numComponentsWithExtraAlpha) {
				case 1:
					//One greyscale component
					finalImage[finalidx] = RGBA{ rescaledComponents[0], rescaledComponents[0], rescaledComponents[0], 255 };
					break;
				case 2:
					//Greyscale with alpha
					finalImage[finalidx] = RGBA{ rescaledComponents[0], rescaledComponents[0], rescaledComponents[0], rescaledComponents[1] };
					break;
				case 3:
					//RGB color only
					finalImage[finalidx] = RGBA{ rescaledComponents[0], rescaledComponents[1], rescaledComponents[2], 255 };
					break;
				case 4:
					//RGBA
					finalImage[finalidx] = RGBA{ rescaledComponents[0], rescaledComponents[1], rescaledComponents[2], rescaledComponents[3] };
					break;
				}
			}

		}
	}
}

void translate_png_data(ImageHeader& header, byte* data, uint32_t dataSize, RGBA* finalImage, RGB* palette, uint32_t paletteEntries, byte* transparency, uint32_t transparencyEntries) {
	uint32_t numComponents = header.hasColor ? 3 : 1;
	if (header.hasAlpha) {
		numComponents += 1;
	}
	if (header.hasPalette) {
		//If it has a palette, each pixel is an index
		numComponents = 1;
	}
	uint32_t bitsPerPixel = header.bitDepth * numComponents;
	uint32_t bytesPerPixel = (bitsPerPixel + 7) / 8;
	uint32_t pixelsPerByte = std::max(8/(header.bitDepth * numComponents), static_cast<uint32_t>(1));

	byte* finalData;
	if (header.bitDepth != 8 || header.interlace || numComponents != 4) {
		finalData = reinterpret_cast<byte*>(malloc(((header.width * bitsPerPixel + 7) / 8) * header.height));
	} else {
		finalData = reinterpret_cast<byte*>(finalImage);
	}

	if (header.interlace) {
		for (uint32_t pass = 1; pass <= 7; pass++) {
			if (header.width <= interlaceXOffset[pass] || header.height <= interlaceYOffset[pass]) {
				continue;
			}
			data = translate_pass(header, data, dataSize, finalData, pass, numComponents, bytesPerPixel, pixelsPerByte);
			rescale_bit_depth(header, finalData, finalImage, pass, numComponents, bytesPerPixel, pixelsPerByte, palette, paletteEntries, transparency, transparencyEntries);
		}
	} else {
		data = translate_pass(header, data, dataSize, finalData, 0, numComponents, bytesPerPixel, pixelsPerByte);
		if (header.bitDepth != 8 || numComponents != 4) {
			rescale_bit_depth(header, finalData, finalImage, 0, numComponents, bytesPerPixel, pixelsPerByte, palette, paletteEntries, transparency, transparencyEntries);
		}
	}

	if (header.bitDepth != 8 || header.interlace) {
		free(finalData);
	}
}

void load_image(const char* fileName, RGBA** outImage, uint32_t* outWidth, uint32_t* outHeight) {
	*outImage = nullptr;
	*outWidth = 0;
	*outHeight = 0;
	uint8_t* file = read_file_to_bytes(fileName);
	BigEndianDataWrapper data{ file };

	byte pngSignature[]{ 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A };
	if (memcmp(data.data, pngSignature, 8) == 0) {
		data.data += 8;

		compute_crc_table();
		HuffmanTree litlenFixedTree;
		HuffmanTree distFixedTree;
		generate_fixed_tree(litlenFixedTree, distFixedTree);

		//Blocks we care about
		BigEndianDataWrapper ihdr{};
		uint32_t idatSize = 0;
		void* idatAllocation = nullptr;
		BigEndianDataWrapper idat{};
		uint32_t paletteSize = 0;
		BigEndianDataWrapper plte{};
		uint32_t transparencySize = 0;
		BigEndianDataWrapper trns{};
		//Load chunks
		while (true) {
			uint32_t length = data.read_uint32();
			if (memcmp(data.data, "IHDR", 4) == 0) {
				assert(idat.data == nullptr && "IDAT appeared before IHDR!");
				assert(plte.data == nullptr && "PLTE appeared before IHDR!");
				assert(trns.data == nullptr && "tRNS appeared before IHDR!");
				ihdr.data = data.data + 4;
			} else if (memcmp(data.data, "IDAT", 4) == 0) {
				if (idat.data) {
					//More than one IDAT chunk, put them all in the same memory. I could make some system that uses less memory and jumps between idat chunks, but this simplifies things a bit.
					if (idatAllocation) {
						idatAllocation = idat.data = reinterpret_cast<byte*>(realloc(idatAllocation, idatSize + length));
						memcpy(idat.data + idatSize, data.data + 4, length);
					} else {
						byte* newData = reinterpret_cast<byte*>(malloc(idatSize + length));
						memcpy(newData, idat.data, idatSize);
						memcpy(newData + idatSize, data.data + 4, length);

						free(idatAllocation);

						idatAllocation = idat.data = newData;
					}
				} else {
					idat.data = data.data + 4;
				}
				idatSize += length;
			} else if (memcmp(data.data, "tRNS", 4) == 0) {
				assert(idat.data == nullptr && "IDAT appeared before tRNS!");
				trns.data = data.data + 4;
				transparencySize = length;
			} else if (memcmp(data.data, "PLTE", 4) == 0) {
				assert(trns.data == nullptr && "tRNS appeared before PLTE!");
				assert((length % 3) == 0 && "Pallet length not given in RGB triplets!");
				paletteSize = length / 3;
				assert(paletteSize <= 256 && "Palette length is greater than max allowed size of 256!");
				plte.data = data.data + 4;
			} else if (memcmp(data.data, "IEND", 4) == 0) {
				break;
			} else {
				//Skip this block, we don't care about it
				//block name
				data.data += 4;
				//main block
				data.data += length;
				//crc32
				data.data += 4;
				continue;
			}
			uint32_t blockCrc = crc32(data.data, length + 4);
			//chunk name
			data.data += 4;
			//chunk data
			data.data += length;
			uint32_t checkCrc = data.read_uint32();
			assert(blockCrc == checkCrc && "Block CRC32 doesn't match! Data corruption?");
		}
		assert(ihdr.data && "No header!");
		assert(idat.data && "No data!");

		ImageHeader header;
		read_ihdr(ihdr, header);

		if (header.hasPalette) {
			assert(plte.data && "Palette required in header, but none provided!");
			assert(transparencySize <= paletteSize && "tRNS has more entries than PLTE!");
		} else if(trns.data) {
			if (header.hasColor) {
				assert(transparencySize == 6 && "Transparency not a 6 byte RGB triplet!");
			} else {
				assert(transparencySize == 2 && "Transparency not a 2 byte single value!");
			}
		}
		

		uint8_t cmf = idat.read_uint8();
		CompressionMethod compression = static_cast<CompressionMethod>(cmf & 0b1111u);
		assert(compression == COMPRESSION_METHOD_DEFLATE && "Compression wasn't deflate, this is not supported");
		uint32_t windowBits = ((cmf >> 4) & 0b1111u) + 8;
		assert(windowBits <= 15 && "Window bits was greater than 15! This is illegal");

		uint8_t flg = idat.read_uint8();
		bool dictPresent = (flg >> 5) & 1;
		assert(dictPresent == false && "Unknown dictionary!");
		CompressionLevel compressionLevel = static_cast<CompressionLevel>((flg >> 6) & 0b11u);

		
		byte dst[1024*32];
		unsigned long dstSize = 1024 * 32;
		unsigned long srcSize = idatSize-6;
		uint64_t puffTime = __rdtsc();
		int32_t error = puff(dst, &dstSize, idat.data, &srcSize);
		puffTime = __rdtsc() - puffTime;
		byte* oldData = idat.data;
		uint32_t puffAdler32 = adler32(dst, dstSize);
		uint32_t puffStoredAdler32 = ntohl(*reinterpret_cast<uint32_t*>(oldData + srcSize));
		

		byte* decompressedData;
		uint32_t decompressedSize;
		uint32_t numComponents = 1 + (!!header.hasColor) * 2 + (!!header.hasAlpha);
		uint32_t sizeGuess = ((header.width * numComponents * header.bitDepth + 7) / 8 + 1) * header.height;
		uint64_t inflateTime = __rdtsc();
		idat.data = inflate(idat.data, &decompressedData, &decompressedSize, sizeGuess, litlenFixedTree, distFixedTree);
		inflateTime = __rdtsc() - inflateTime;
		uint32_t storedAdler32 = idat.read_uint32();
		uint32_t currentAdler32 = adler32(decompressedData, decompressedSize);
		assert(storedAdler32 == currentAdler32 && "Data checksums don't match!");
		if (idatAllocation) {
			free(idatAllocation);
		}

		RGBA* finalImage = reinterpret_cast<RGBA*>(malloc(header.width * header.height * sizeof(RGBA)));
		uint64_t translationTime = __rdtsc();
		translate_png_data(header, decompressedData, decompressedSize, finalImage, reinterpret_cast<RGB*>(plte.data), paletteSize, trns.data, transparencySize);
		translationTime = __rdtsc() - translationTime;

		std::cout << "Inflate time: " << inflateTime << "\nPuff time:" << puffTime << "\nTranslate time: " << translationTime << "\n" << std::endl;
		free(decompressedData);

		litlenFixedTree.destroy();
		distFixedTree.destroy();

		*outImage = finalImage;
		*outWidth = header.width;
		*outHeight = header.height;
	}
	delete[] file;
}

void free_image(RGBA* image) {
	free(image);
}

void test_image_file(const char* name) {
	//My image load
	RGBA* dImage = nullptr;
	uint32_t dWidth = 0;
	uint32_t dHeight = 0;
	uint64_t time = __rdtsc();
	load_image(name, &dImage, &dWidth, &dHeight);
	time = __rdtsc() - time;
	assert(dImage && "Failed image load!");


	//Validate against a real image library
	int32_t xDim;
	int32_t yDim;
	int32_t comp;
	
	uint64_t time2 = __rdtsc();
	byte* stbImgData = stbi_load(name, &xDim, &yDim, &comp, 4);
	time2 = __rdtsc() - time2;
	std::cout << "Image load: " << time << "\nSTB Image load: " << time2 << "\n" << std::endl;
	exit(0);
	assert(stbImgData && "Failed stb image load!");

	assert(xDim == dWidth && "width doesn't match!");
	assert(yDim == dHeight && "height doesn't match!");
	for (uint32_t y = 0; y < yDim; y++) {
		for (uint32_t x = 0; x < xDim; x++) {
			uint32_t idx = (y * xDim + x) * 4 + 0;
			if (stbImgData[idx] != reinterpret_cast<byte*>(dImage)[idx]) {
				std::cout << "Byte mismatch!" << std::endl;
				exit(-1);
			}
		}
	}

	free_image(dImage);
}

void test_range(const char** names, uint32_t count) {
	const char* dir = "resources/PngSuite-2017jul19/";
	const char* ext = ".png";
	for (uint32_t i = 0; i < count; i++) {
		std::cout << "Testing: " << names[i] << std::endl;
		uint32_t size = strlen(dir) + strlen(names[i]) + strlen(ext) + 1;
		char* file = new char[size];
		file[0] = '\0';
		strcat_s(file, size, dir);
		strcat_s(file, size, names[i]);
		strcat_s(file, size, ext);
		uint64_t time = __rdtsc();
		test_image_file(file);
		time = __rdtsc() - time;
		std::cout << "Success!" << std::endl;
	}
}

int main() {
	const char* basic[30] = { 
		"basn0g01", "basn0g02", "basn0g04", "basn0g08", "basn0g16", "basn2c08", "basn2c16", "basn3p01", "basn3p02", "basn3p04", "basn3p08", "basn4a08", "basn4a16", "basn6a08", "basn6a16", 
		"basi0g01", "basi0g02", "basi0g04", "basi0g08", "basi0g16", "basi2c08", "basi2c16", "basi3p01", "basi3p02", "basi3p04", "basi3p08", "basi4a08", "basi4a16", "basi6a08", "basi6a16" 
	};
	const char* oddSizes[36] = {
		"s01n3p01", "s02n3p01", "s03n3p01", "s04n3p01", "s05n3p02", "s06n3p02", "s07n3p02", "s08n3p02", "s09n3p02", "s32n3p04", "s33n3p04", "s34n3p04", "s35n3p04", "s36n3p04", "s37n3p04", "s38n3p04", "s39n3p04", "s40n3p04", 
		"s01i3p01", "s02i3p01", "s03i3p01", "s04i3p01", "s05i3p02", "s06i3p02", "s07i3p02", "s08i3p02", "s09i3p02", "s32i3p04", "s33i3p04", "s34i3p04", "s35i3p04", "s36i3p04", "s37i3p04", "s38i3p04", "s39i3p04", "s40i3p04"
	};
	const char* backgroundColors[8] = {
		"bgbn4a08", "bggn4a16", "bgwn6a08", "bgyn6a16",
		"bgai4a08", "bgai4a16", "bgan6a08", "bgan6a16"
	};
	const char* transparency[14] = {
		"tbbn0g04", "tbbn2c16", "tbbn3p08", "tbgn2c16", "tbgn3p08", "tbrn2c08", "tbwn0g16", "tbwn3p08", "tbyn3p08", "tp0n0g08", "tp0n2c08", "tp0n3p08", "tp1n3p08", "tm3n3p02"
	};
	const char* gamma[18] = {
		"g03n0g16", "g03n2c08", "g03n3p04", "g04n0g16", "g04n2c08", "g04n3p04", "g05n0g16", "g05n2c08", "g05n3p04", "g07n0g16", "g07n2c08", "g07n3p04", "g10n0g16", "g10n2c08", "g10n3p04", "g25n0g16", "g25n2c08", "g25n3p04"
	};
	const char* filtering[11] = {
		"f00n0g08", "f00n2c08",
		"f01n0g08", "f01n2c08",
		"f02n0g08", "f02n2c08",
		"f03n0g08", "f03n2c08",
		"f04n0g08", "f04n2c08",
		"f99n0g04"
	};
	const char* additionalPalettes[6] = {
		"pp0n2c16", "pp0n6a08", "ps1n0g08", "ps1n2c16", "ps2n0g08", "ps2n2c16"
	};
	const char* ancillaryChunks[26] = {
		"ccwn2c08", "ccwn3p08",
		"cdfn2c08", "cdhn2c08", "cdsn2c08", "cdun2c08",
		"ch1n3p04", "ch2n3p08",
		"cm0n0g04", "cm7n0g04", "cm9n0g04",
		"cs3n2c16", "cs3n3p08", "cs5n2c08", "cs5n3p08", "cs8n2c08", "cs8n3p08",
		"ct0n0g04", "ct1n0g04", "ctzn0g04", "cten0g04", "ctfn0g04", "ctgn0g04", "cthn0g04", "ctjn0g04",
		"exif2c08"
	};
	const char* chunkOrdering[8] = {
		"oi1n0g16", "oi1n2c16", "oi2n0g16", "oi2n2c16", "oi4n0g16", "oi4n2c16", "oi9n0g16", "oi9n2c16"
	};
	const char* compressionLevel[4] = {
		"z00n2c08", "z03n2c08", "z06n2c08", "z09n2c08"
	};
	const char* corrupted[14] = {
		"xs1n0g01", "xs2n0g01", "xs4n0g01", "xs7n0g01", "xcrn0g04", "xlfn0g04", "xhdn0g08", "xc1n0g08", "xc9n2c08", "xd0n2c08", "xd3n2c08", "xd9n2c08", "xdtn0g01", "xcsn0g01"
	};

	const char* imageFile = "resources/PngSuite-2017jul19/basi0g01.png";

	test_image_file("resources/gravelc.png");

	std::cout << "Doing basic tests...\n" << std::endl;
	test_range(basic, 30);
	std::cout << "\nBasic tests complete!\n\n" << std::endl;

	std::cout << "Doing odd size tests...\n" << std::endl;
	test_range(oddSizes, 36);
	std::cout << "\nOdd size tests complete!\n\n" << std::endl;

	std::cout << "Doing background color tests...\n" << std::endl;
	test_range(backgroundColors, 8);
	std::cout << "\nBackground color tests complete!\n\n" << std::endl;

	std::cout << "Doing transparency tests...\n" << std::endl;
	test_range(transparency, 14);
	std::cout << "\nTransparency tests complete!\n\n" << std::endl;

	std::cout << "Doing gama tests...\n" << std::endl;
	test_range(gamma, 18);
	std::cout << "\nGama tests complete!\n\n" << std::endl;

	std::cout << "Doing filtering tests...\n" << std::endl;
	test_range(filtering, 11);
	std::cout << "\nFiltering tests complete!\n\n" << std::endl;

	std::cout << "Doing additional palette tests..\n" << std::endl;
	test_range(additionalPalettes, 6);
	std::cout << "\nAdditional palette tests complete!\n\n" << std::endl;

	std::cout << "Doing ancillary chunk tests...\n" << std::endl;
	test_range(ancillaryChunks, 26);
	std::cout << "\nAncillary chunk tests complete!\n\n" << std::endl;

	std::cout << "Doing chunk ordering tests...\n" << std::endl;
	test_range(chunkOrdering, 8);
	std::cout << "\nChunk ordering tests complete!\n\n" << std::endl;

	std::cout << "Doing compression level tests...\n" << std::endl;
	test_range(compressionLevel, 4);
	std::cout << "\nCompression level tests complete!\n\n" << std::endl;

	return 0;
}