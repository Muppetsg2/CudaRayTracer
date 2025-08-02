#pragma once
#include "vec.hpp"
#include "math_functions.hpp"
#include "ldg_helpers.hpp"

namespace craytracer {
	enum class LOAD_FORMAT : uint8_t {
		PNG = 0,
		JPG = 1,
		BMP = 2,
		TGA = 3,
		HDR = 4,
		BY_EXTENSION = 5
	};

	enum class WRAP_MODE : uint8_t {
		CLAMP = 0,
		MIRROR = 1,
		REPEAT = 2
	};

	class Texture {
	private:
		float* _data = nullptr;
		unsigned int _width = 0u;
		unsigned int _height = 0u;
		unsigned int _channels = 4u;
		WRAP_MODE _verticalWrapMode = WRAP_MODE::REPEAT;
		WRAP_MODE _horizontalWrapMode = WRAP_MODE::REPEAT;

		__host__ float* _convertData(unsigned char* image, int channels) const {
			size_t size = (size_t)_width * (size_t)_height;
			float* data = new float[size * (size_t)_channels];

			for (size_t i = 0; i < size; ++i) {
				float r = 0.f, g = 0.f, b = 0.f, a = 1.f;
				size_t posImage = i * (size_t)channels;
				size_t posData = i * (size_t)_channels;
				switch (channels) {
					case 1: {
						r = g = b = (float)image[posImage] / 255.f;
						break;
					}
					case 2: {
						r = (float)image[posImage] / 255.f;
						g = (float)image[posImage + 1ull] / 255.f;
						break;
					}
					case 3: {
						r = (float)image[posImage] / 255.f;
						g = (float)image[posImage + 1ull] / 255.f;
						b = (float)image[posImage + 2ull] / 255.f;
						break;
					}
					case 4: {
						r = (float)image[posImage] / 255.f;
						g = (float)image[posImage + 1ull] / 255.f;
						b = (float)image[posImage + 2ull] / 255.f;
						a = (float)image[posImage + 3ull] / 255.f;
						break;
					}
				}

				data[posData] = r;
				data[posData + 1ull] = g;
				data[posData + 2ull] = b;
				data[posData + 3ull] = a;
			}

			return data;
		}

		__host__ __device__ float* _repairChannels(const float* image, int channels) const {
			size_t size = (size_t)_width * (size_t)_height;
			float* data = new float[size * (size_t)_channels];

			if (channels == 4) {
#ifdef __CUDACC__
				::cuda::std::memcpy(data, image, size * (size_t)_channels * sizeof(float));
#else
				::std::memcpy(data, image, size * (size_t)_channels * sizeof(float));
#endif
				return data;
			}

			for (size_t i = 0; i < size; ++i) {
				float r = 0.f, g = 0.f, b = 0.f, a = 1.f;
				size_t posImage = i * (size_t)channels;
				size_t posData = i * (size_t)_channels;
				switch (channels) {
					case 1: {
						r = g = b = image[posImage];
						break;
					}
					case 2: {
						r = image[posImage];
						g = image[posImage + 1ull];
						break;
					}
					case 3: {
						r = image[posImage];
						g = image[posImage + 1ull];
						b = image[posImage + 2ull];
						break;
					}
				}

				data[posData] = r;
				data[posData + 1ull] = g;
				data[posData + 2ull] = b;
				data[posData + 3ull] = a;
			}

			return data;
		}

		__device__ uvec2 _imageSpaceCoordinates(vec2 uv) const {
			vec2 resF = vec2(uv);

			if (resF.x() < 0.f || resF.x() > 1.f) {
				switch (_horizontalWrapMode) {
					case WRAP_MODE::CLAMP: {
#ifdef __CUDACC__
						resF.x() = saturate<float, true>(resF.x());
#else
						resF.x() = saturate(resF.x());
#endif
						break;
					}
					case WRAP_MODE::REPEAT: {
						resF.x() = fract(resF.x());
						break;
					}
					case WRAP_MODE::MIRROR: {
#ifdef __CUDACC__
						resF.x() = ::cuda::std::fabsf(fract(resF.x()) - 1.f);
#else
						resF.x() = ::std::fabsf(fract(resF.x()) - 1.f);
#endif
						break;
					}
				}
			}

			if (resF.y() < 0.f || resF.y() > 1.f) {
				switch (_verticalWrapMode) {
					case WRAP_MODE::CLAMP: {
#ifdef __CUDACC__
						resF.y() = saturate<float, true>(resF.y());
#else
						resF.y() = saturate(resF.y());
#endif
						break;
					}
					case WRAP_MODE::REPEAT: {
						resF.y() = fract(resF.y());
						break;
					}
					case WRAP_MODE::MIRROR: {
#ifdef __CUDACC__
						resF.y() = ::cuda::std::fabsf(fract(resF.y()) - 1.f);
#else
						resF.y() = ::std::fabsf(fract(resF.y()) - 1.f);
#endif
						break;
					}
				}
			}

			uvec2 res = uvec2(
				(unsigned int)(resF.x() * (_width - 1u)),
				(unsigned int)(resF.y() * (_height - 1u))
			);

			return res;
		}

	public:
		Texture() = default;

		__host__ Texture(const char* path, LOAD_FORMAT format = LOAD_FORMAT::BY_EXTENSION, WRAP_MODE verticalWrapMode = WRAP_MODE::REPEAT, WRAP_MODE horizontalWrapMode = WRAP_MODE::REPEAT, bool flip = false)
			: _verticalWrapMode(verticalWrapMode), _horizontalWrapMode(horizontalWrapMode) {
			if (loadImageFromPath(path, format, flip)) {
				fprintf(stderr, "TEXTURE: Couldn't Load Image\n");
			}
		}

		__host__ __device__ Texture(const float* data, unsigned int width, unsigned int height, unsigned int channels, WRAP_MODE verticalWrapMode = WRAP_MODE::REPEAT, WRAP_MODE horizontalWrapMode = WRAP_MODE::REPEAT) {
			_width = width;
			_height = height;
			_verticalWrapMode = verticalWrapMode;
			_horizontalWrapMode = horizontalWrapMode;
			_data = _repairChannels(data, channels);
		}

		__host__ __device__ ~Texture() { freeImageData(); }

		__device__ unsigned int getWidth() const { return _width; }

		__device__ unsigned int getHeight() const { return _height; }

		__device__ vec4 getPixelColor(size_t pos) const {
			if (!_data) return vec4();

#ifdef __CUDACC__
			if (pos >= (size_t)ldg_uint(&_width) * (size_t)ldg_uint(&_height)) return vec4();
			size_t pos0 = pos * (size_t)ldg_uint(&_channels);
#else
			if (pos >= (size_t)_width * (size_t)_height) return vec4();
			size_t pos0 = pos * (size_t)_channels;
#endif
			return vec4(_data[pos0], _data[pos0 + 1], _data[pos0 + 2], _data[pos0 + 3]);
		}

		__device__ vec4 getPixelColor(unsigned int x, unsigned int y) const {
			if (!_data) return vec4();

#ifdef __CUDACC__
			const unsigned int w = ldg_uint(&_width);

			if (x >= w || y >= ldg_uint(&_height)) return vec4();
			size_t pos0 = (size_t)y * w + (size_t)x;
#else
			if (x >= _width || y >= _height) return vec4();
			size_t pos0 = (size_t)y * (size_t)_width + (size_t)x;
#endif
			return getPixelColor(pos0);
		}

		__device__ WRAP_MODE getVerticalWrapMode() const { return _verticalWrapMode; }

		__device__ WRAP_MODE getHorizontalWrapMode() const { return _horizontalWrapMode; }

		__host__ void setVerticalWrapMode(WRAP_MODE mode) { _verticalWrapMode = mode; }

		__host__ void setHorizontalWrapMode(WRAP_MODE mode) { _horizontalWrapMode = mode; }

		__host__ __device__ bool isImageLoaded() const { return _data; }

		__device__ vec4 sample(vec2 uv) const {
			uvec2 pos = _imageSpaceCoordinates(uv);
			return getPixelColor(pos.x(), pos.y());
		}

		__host__ bool loadImageFromPath(const char* path, LOAD_FORMAT format = LOAD_FORMAT::BY_EXTENSION, bool flip = false) {
			::std::string file_path = ::std::filesystem::absolute(path).string();
			stbi_set_flip_vertically_on_load(flip);

			LOAD_FORMAT f = format;
			if (format == LOAD_FORMAT::BY_EXTENSION) {
				::std::string ext = ::std::filesystem::path(file_path).extension().string();
				if (ext == ".png") {
					f = LOAD_FORMAT::PNG;
				}
				else if (ext == ".jpg") {
					f = LOAD_FORMAT::JPG;
				}
				else if (ext == ".bmp") {
					f = LOAD_FORMAT::BMP;
				}
				else if (ext == ".hdr") {
					f = LOAD_FORMAT::HDR;
				}
				else if (ext == ".tga") {
					f = LOAD_FORMAT::TGA;
				}
			}

			if (f == LOAD_FORMAT::BY_EXTENSION) return false;

			void* data = nullptr;
			bool isFloat = false;
			int width = 0, height = 0, channels = 0;
			switch (f) {
				case LOAD_FORMAT::HDR: {
					data = (void*)stbi_loadf(file_path.c_str(), &width, &height, &channels, 0);
					isFloat = true;
					break;
				}
				case LOAD_FORMAT::PNG:
				case LOAD_FORMAT::JPG:
				case LOAD_FORMAT::BMP:
				case LOAD_FORMAT::TGA: {
					data = (void*)stbi_load(file_path.c_str(), &width, &height, &channels, 0);
					break;
				}
			}

			if (!data) {
				fprintf(stderr, "TEXTURE: Couldn't Load Image\n");
				stbi_image_free(data);
				return false;
			}

			_width = (unsigned int)width;
			_height = (unsigned int)height;

			if (isFloat) {
				_data = _repairChannels((float*)data, channels);
			}
			else {
				_data = _convertData((unsigned char*)data, channels);
			}

			stbi_image_free(data);
			return true;
		}

		__host__ __device__ void freeImageData() {
			if (!_data)
			{
				delete[] _data;
				_data = nullptr;
			}
		}
	};
}