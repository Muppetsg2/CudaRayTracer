/**************************************************************
 *                                                            *
 *  Project:   CudaRayTracer                                  *
 *  Authors:   Muppetsg2 & MAIPA01                            *
 *  License:   MIT License                                    *
 *  Last Update: 22.09.2025                                   *
 *                                                            *
 **************************************************************/

#pragma once

#include "stb_image.h"
#include "stb_image_write.h"
#include <SFML/Graphics.hpp>
#include <yaml-cpp/yaml.h>
#include <stdio.h>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <thrust/functional.h>
#include <thrust/swap.h>
#include <thrust/pair.h>
#include <thrust/tuple.h>
#include <corecrt_math_defines.h>
#include <time.h>
#include <type_traits>
#include <vector>


#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#elif defined(__linux__)
#include <unistd.h>
#endif