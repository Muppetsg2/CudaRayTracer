/**************************************************************
 *                                                            *
 *  Project:   CudaRayTracer                                  *
 *  Authors:   Muppetsg2 & MAIPA01                            *
 *  License:   MIT License                                    *
 *  Last Update: 29.09.2025                                   *
 *                                                            *
 **************************************************************/

#pragma once
#include <algorithm>
#include <chrono>
#include <corecrt_math_defines.h>
#include <cstdio>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <stack>
#include <stdexcept>
#include <stdio.h>
#include <string>
#include <time.h>
#include <type_traits>
#include <vector>

#include <SFML/Graphics.hpp>
#include <yaml-cpp/yaml.h>
#include <thrust/functional.h>
#include <thrust/pair.h>
#include <thrust/swap.h>
#include <thrust/tuple.h>

#include "stb_image.h"
#include "stb_image_write.h"


#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#elif defined(__linux__)
#include <unistd.h>
#endif