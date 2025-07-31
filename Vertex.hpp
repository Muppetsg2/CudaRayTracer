#pragma once
#include "vec.hpp"

namespace craytracer {
	struct Vertex {
		vec3 pos;
		vec2 tex;
		vec3 normal;
	};
}