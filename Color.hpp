#pragma once
#include "vec.hpp"

using namespace MSTD_NAMESPACE;

namespace craytracer {
    class Color {
    public:
#pragma region PREDEFINED COLORS
        // Rainbow
        __device__ static vec4 red()       { return vec4(1.000f, 0.000f, 0.000f, 1.000f); }
        __device__ static vec4 orange()    { return vec4(1.000f, 0.647f, 0.000f, 1.000f); }
        __device__ static vec4 yellow()    { return vec4(1.000f, 1.000f, 0.000f, 1.000f); }
        __device__ static vec4 lime()      { return vec4(0.196f, 0.803f, 0.196f, 1.000f); }
        __device__ static vec4 green()     { return vec4(0.000f, 1.000f, 0.000f, 1.000f); }
        __device__ static vec4 teal()      { return vec4(0.000f, 0.502f, 0.502f, 1.000f); }
        __device__ static vec4 cyan()      { return vec4(0.000f, 1.000f, 1.000f, 1.000f); }
        __device__ static vec4 turquoise() { return vec4(0.251f, 0.878f, 0.816f, 1.000f); }
        __device__ static vec4 lapis()     { return vec4(0.149f, 0.380f, 0.612f, 1.000f); }
        __device__ static vec4 blue()      { return vec4(0.000f, 0.000f, 1.000f, 1.000f); }
        __device__ static vec4 indigo()    { return vec4(0.294f, 0.000f, 0.510f, 1.000f); }
        __device__ static vec4 violet()    { return vec4(0.933f, 0.510f, 0.933f, 1.000f); }
        __device__ static vec4 purple()    { return vec4(0.502f, 0.000f, 0.502f, 1.000f); }
        __device__ static vec4 magenta()   { return vec4(1.000f, 0.000f, 1.000f, 1.000f); }
        __device__ static vec4 pink()      { return vec4(1.000f, 0.753f, 0.796f, 1.000f); }

        // Neutral & Other Colors
        __device__ static vec4 brown()     { return vec4(0.647f, 0.165f, 0.165f, 1.000f); }
        __device__ static vec4 maroon()    { return vec4(0.502f, 0.000f, 0.000f, 1.000f); }
        __device__ static vec4 olive()     { return vec4(0.502f, 0.502f, 0.000f, 1.000f); }
        __device__ static vec4 gold()      { return vec4(1.000f, 0.843f, 0.000f, 1.000f); }
        __device__ static vec4 silver()    { return vec4(0.753f, 0.753f, 0.753f, 1.000f); }
        __device__ static vec4 navy()      { return vec4(0.000f, 0.000f, 0.502f, 1.000f); }
        __device__ static vec4 mint()      { return vec4(0.741f, 0.988f, 0.788f, 1.000f); }
        __device__ static vec4 beige()     { return vec4(0.961f, 0.961f, 0.863f, 1.000f); }
        __device__ static vec4 salmon()    { return vec4(0.980f, 0.502f, 0.447f, 1.000f); }
        __device__ static vec4 coral()     { return vec4(1.000f, 0.498f, 0.314f, 1.000f); }

        // Black & White
        __device__ static vec4 white()     { return vec4(1.000f, 1.000f, 1.000f, 1.000f); }
        __device__ static vec4 gray()      { return vec4(0.500f, 0.500f, 0.500f, 1.000f); }
        __device__ static vec4 black()     { return vec4(0.000f, 0.000f, 0.000f, 1.000f); }
#pragma endregion
    };
}