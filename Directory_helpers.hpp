/**************************************************************
 *                                                            *
 *  Project:   CudaRayTracer                                  *
 *  Authors:   Muppetsg2 & MAIPA01                            *
 *  License:   MIT License                                    *
 *  Last Update: 22.09.2025                                   *
 *                                                            *
 **************************************************************/

#pragma once

namespace craytracer {
    inline std::filesystem::path getExecutableDir() {
        char buffer[4096];
        uint32_t size = sizeof(buffer);

#if defined(_WIN32)
        DWORD len = GetModuleFileNameA(NULL, buffer, size);
        if (len == 0 || len == size) {
            fprintf(stderr, "Couldn't fetch exe file directory");
            exit(98);
        }
#elif defined(__APPLE__)
        if (_NSGetExecutablePath(buffer, &size) != 0) {
            fprintf(stderr, "Couldn't fetch exe file directory");
            exit(98);
        }
#elif defined(__linux__)
        ssize_t len = readlink("/proc/self/exe", buffer, size - 1);
        if (len == -1) {
            fprintf(stderr, "Couldn't fetch exe file directory");
            exit(98);
        }
        buffer[len] = '\0';
#else
        fprintf(stderr, "Unknown OS");
#endif

        return std::filesystem::absolute(std::filesystem::path(buffer)).parent_path();
    }
}