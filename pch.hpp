/**************************************************************
 *                                                            *
 *  Project:   CudaRayTracer                                  *
 *  Authors:   Muppetsg2 & MAIPA01                            *
 *  License:   MIT License                                    *
 *  Last Update: 10.08.2025                                   *
 *                                                            *
 **************************************************************/

#pragma once

#include "stb_image.h"
#include "stb_image_write.h"
#include <SFML/Graphics.hpp>
#include <stdio.h>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>
#include <thrust/functional.h>
#include <thrust/swap.h>
#include <thrust/pair.h>
#include <thrust/tuple.h>
#include <corecrt_math_defines.h>
#include <time.h>