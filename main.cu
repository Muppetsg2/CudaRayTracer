#ifdef __INTELLISENSE__
#define __CUDACC__
#endif // __INTELLISENSE__

#pragma region PCH
#include "pch.hpp"
#pragma endregion

#pragma region CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#pragma endregion

#pragma region MSTD
#include "vec.hpp"
#include "math_functions.hpp"
#pragma endregion

#pragma region MyClasses
#include "Color.hpp"
#include "Material.hpp"
#include "Geometry.hpp"
#include "GeometryList.hpp"
#include "Sphere.hpp"
#include "Quad.hpp"
#include "Camera.hpp"
#include "Light.hpp"
#include "LightList.hpp"
#include "AreaLight.hpp"
#include "ldg_helpers.hpp"
#pragma endregion

using namespace craytracer;

#define checkCudaErrors(val) checkCuda((val), #val, __FILE__, __LINE__)
void checkCuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA error = %d at %s: %d '%s'\n", static_cast<unsigned int>(result), file, line, func);
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__device__ __forceinline__ void get_surface_coordinate_system(const vec3& hitNormal, vec3& Nx, vec3& Nz) {
    if (::cuda::std::fabsf(hitNormal.x()) > ::cuda::std::fabsf(hitNormal.y())) {
        Nx = vec3(hitNormal.z(), 0.0f, -hitNormal.x()) * rsqrtf(hitNormal.x() * hitNormal.x() + hitNormal.z() * hitNormal.z());
    }
    else {
        Nx = vec3(0.0f, -hitNormal.z(), hitNormal.y()) * rsqrtf(hitNormal.y() * hitNormal.y() + hitNormal.z() * hitNormal.z());
    }
    Nz = hitNormal.cross(Nx);
}

__device__ __forceinline__ ::cuda::std::pair<vec3, vec3> get_random_ray_values_in_hemisphere(float r1, float r2, const vec3& hitPos, const vec3& hitNormal, const vec3& Nx, const vec3& Nz) {
    float sinTheta = 1.f - r1 * r1;
    float phi = r2 * 2.f * (float)M_PI;

    float x = sinTheta * ::cuda::std::cosf(phi);
    float z = sinTheta * ::cuda::std::sinf(phi);

    vec3 rayDirWorld = vec3(
        x * Nz.x() + r1 * hitNormal.x() + z * Nx.x(),
        x * Nz.y() + r1 * hitNormal.y() + z * Nx.y(),
        x * Nz.z() + r1 * hitNormal.z() + z * Nx.z()
    );

    return ::cuda::std::make_pair(hitPos + 0.01f * rayDirWorld, rayDirWorld);
}

__device__ __forceinline__ ::cuda::std::pair<vec3, vec3> get_reflect_ray(Ray rayIn, vec3 hitPos, vec3 hitNormal)
{
    vec3 I = rayIn.getDirection().normalized();
    vec3 rayDir = reflect(I, hitNormal.normalized());
    return ::cuda::std::make_pair(hitPos + 0.01f * rayDir, rayDir);
}

__device__ __forceinline__ ::cuda::std::pair<vec3, vec3> get_refraction_ray(Ray rayIn, vec3 hitPos, vec3 hitNormal, float refIndex)
{
    bool front_face = dot(rayIn.getDirection(), hitNormal) < 0.f;

    vec3 norm = front_face ? hitNormal : -hitNormal;

    float ratio = refIndex / AIR_INDEX;
    if (front_face) ratio = 1.0f / ratio;
    vec3 rayDirection = rayIn.getDirection().normalized();

    float cos_theta = ::cuda::std::fminf(dot(-rayDirection, norm), 1.0f);
    float sin_theta = 1.0f / rsqrtf(1.0f - cos_theta * cos_theta);

    //bool cannot_refract = ratio * sin_theta > 1.0f || reflectance(cos_theta, ratio) > random_double();
    bool cannot_refract = ratio * sin_theta > 1.0f || reflectance(cos_theta, ratio) > 1.0f;

    vec3 dir;
    if (cannot_refract) {
        dir = reflect(rayDirection, norm);
    }
    else {
        dir = refract(rayDirection, norm, ratio);
    }

    return ::cuda::std::make_pair(hitPos + 0.01f * dir, dir);
}

__device__ vec4 color(const Ray& r, Geometry** world, Light** lights, curandState* local_rand_state, unsigned int ref_iter, unsigned int gl_iter, unsigned int ind_rays) {
    constexpr size_t MAX_STACK = 128;
    const vec4 skyColor = Color::black();
    
    struct RayState {
        vec3 r_o;
        vec3 r_dir;
        vec4 attenuation;
        unsigned int ref_iter_remaining;
        unsigned int global_depth;
    };

    RayState stack[MAX_STACK] = {};
    size_t stackPtr = 0;

    stack[stackPtr++] = { r.getOrigin(), r.getDirection(), Color::white(), ref_iter, gl_iter};

    vec4 finalColor = vec4::zero();
    Ray ray;
    ::cuda::std::pair<vec3, vec3> p;

    while (stackPtr > 0) {
        RayState state = stack[--stackPtr];

        if (state.ref_iter_remaining == 0) continue;

        ray.setOrigin(state.r_o);
        ray.setDirection(state.r_dir);

        RayHit hit;
        if (!(*world)->hit(ray, hit)) {
            /*vec3 unit_direction = currentRay.getDirection().normalized();
            float t = 0.5f * (unit_direction.y() + 1.0f);
            vec4 skyColor = (1.0f - t) * vec4::one() + t * vec4(0.5f, 0.7f, 1.0f, 1.0f);*/
            //vec4 skyColor = Color::black();
            finalColor += state.attenuation * skyColor;
            continue;
        }

        const vec4 mat_diff = ldg_vec4(&hit.hitMat->diffuse);
        const float mat_ref_idx = ldg_float(&hit.hitMat->refractIndex);
        const MaterialType mat_type = ldg_enum_uint8_type(&hit.hitMat->type);
        switch (mat_type) {
            case MaterialType::Diffuse: {
                LightInput input {
#if _HAS_CXX20
                    .fragPos = hit.hitPoint,
                    .uv = hit.hitUV,
                    .norm = hit.hitNormal,
                    .viewDir = -state.r_dir,
                    .mat = hit.hitMat
#else
                    hit.hitPoint,
                    hit.hitUV,
                    hit.hitNormal,
                    -state.r_dir,
                    hit.hitMat
#endif
                };

                vec4 directDiffuse = (*lights)->calculateColor(input, *world, local_rand_state);

                if (state.global_depth > 0u) {
                    vec3 Nx, Nz;
                    get_surface_coordinate_system(hit.hitNormal, Nx, Nz);

                    for (unsigned int n = 0; n < ind_rays; ++n) {
                        float r1 = curand_uniform(local_rand_state);
                        float r2 = curand_uniform(local_rand_state);

                        if (stackPtr < MAX_STACK) {
                            p = get_random_ray_values_in_hemisphere(r1, r2, hit.hitPoint, hit.hitNormal, Nx, Nz);
                            stack[stackPtr++] = {
                                p.first,
                                p.second,
                                r1 * mat_diff * 2.f / ind_rays * state.attenuation,
                                ref_iter,
                                state.global_depth - 1u
                            };
                        }
                    }
                }

                finalColor += state.attenuation * directDiffuse;
                break;
            }
            case MaterialType::Reflect: {
                if (stackPtr < MAX_STACK) {
                    p = get_reflect_ray(ray, hit.hitPoint, hit.hitNormal);
                    stack[stackPtr++] = {
                        p.first,
                        p.second,
                        state.attenuation * mat_diff,
                        state.ref_iter_remaining - 1u,
                        state.global_depth
                    };
                }
                break;
            }
            case MaterialType::Refractive: {
                if (stackPtr < MAX_STACK) {
                    p = get_refraction_ray(ray, hit.hitPoint, hit.hitNormal, mat_ref_idx);
                    stack[stackPtr++] = {
                        p.first,
                        p.second,
                        state.attenuation * mat_diff,
                        state.ref_iter_remaining - 1u,
                        state.global_depth
                    };
                }
                break;
            }
        }
    }

    return finalColor;
}

// Max 4 iterations
__device__ vec4 aa_color(float centerX, float centerY, vec2 size, float width, float height, unsigned int aa_iter, unsigned int ref_iter, unsigned int gl_iter, unsigned int ind_rays, Camera* cam, Geometry** world, Light** lights, curandState* local_rand_state) {
    constexpr size_t MAX_STACK = 256;
    const vec2 aa_offsets[4] = { { -1.f, -1.f }, { 1.f, -1.f }, { -1.f, 1.f }, { 1.f, 1.f } };
    aa_iter = aa_iter > 4 ? 4 : aa_iter;

    struct AA_Task {
        float centerX;
        float centerY;
        vec2 size;
        unsigned int sample;
    };

    AA_Task stack[MAX_STACK] = {};
    size_t stack_top = 0ull;

    stack[stack_top++] = { centerX, centerY, size, aa_iter };
    vec4 accumulatedColor = vec4::zero();
    Ray r;
    vec2 halfSize;
    while (stack_top > 0ull) {
        AA_Task t = stack[--stack_top];
        halfSize = t.size * 0.5f;
        float mult = 1.f / ::cuda::std::powf(4.f, aa_iter - t.sample);

        if (t.sample == 0u) {
            r = cam->getRay(t.centerX, t.centerY, width, height);
            accumulatedColor += color(r, world, lights, local_rand_state, ref_iter, gl_iter, ind_rays) * mult;
            continue;
        }

        if (t.sample == 1u) {
            for (int i = 0; i < 4; ++i) {
                r = cam->getRay(t.centerX + aa_offsets[i].x() * halfSize.x() * 0.5f,
                    t.centerY + aa_offsets[i].y() * halfSize.y() * 0.5f,
                    width, height);
                accumulatedColor += color(r, world, lights, local_rand_state, ref_iter, gl_iter, ind_rays) * mult * 0.25f;
            }
            continue;
        }

        vec4 colors[4];
        bool allEqual = true;

        for (int i = 0; i < 4; ++i) {
            r = cam->getRay(t.centerX + aa_offsets[i].x() * halfSize.x(),
                t.centerY + aa_offsets[i].y() * halfSize.y(),
                width, height);
            colors[i] = color(r, world, lights, local_rand_state, ref_iter, gl_iter, ind_rays);
        }

        for (int i = 1; i < 4; ++i) {
            if (colors[i] != colors[0]) {
                allEqual = false;
                break;
            }
        }

        if (allEqual) {
            accumulatedColor += colors[0] * mult;
        }
        else {
            if (stack_top + 4 > MAX_STACK) {
                accumulatedColor += colors[0] * mult * 0.25f;
                accumulatedColor += colors[1] * mult * 0.25f;
                accumulatedColor += colors[2] * mult * 0.25f;
                accumulatedColor += colors[3] * mult * 0.25f;
                continue;
            }

            for (int i = 0; i < 4; ++i) {
                stack[stack_top++] = {
                    t.centerX + aa_offsets[i].x() * halfSize.x() * 0.5f,
                    t.centerY + aa_offsets[i].y() * halfSize.y() * 0.5f,
                    halfSize, t.sample - 1u
                };
            }
        }
    }

    return accumulatedColor;
}

__device__ __forceinline__ vec4 get_world_coordinates(unsigned int x, unsigned int y, unsigned int w, unsigned int h)
{
    float height_world = 2.f;
    float width_world = height_world * ((float)w / (float)h);
    float x_world = remap((float)x, 0.f, (float)w, -width_world * 0.5f, width_world * 0.5f);
    float y_world = remap((float)y, 0.f, (float)h, 1.f, -1.f);

    return vec4(x_world, y_world, width_world, height_world);
}

void randomInit(unsigned int nx, unsigned int ny, unsigned int tx, unsigned int ty, curandState* rand_state);
__global__ void random_init(unsigned int nx, unsigned int ny, long long seed, curandState* rand_state) {
    size_t x = threadIdx.x + blockIdx.x * blockDim.x;
    size_t y = threadIdx.y + blockIdx.y * blockDim.y;

    if ((x >= nx) || (y >= ny)) return;

    size_t idx = y * nx + x;
    curand_init(seed + idx, 0, 0, &rand_state[idx]);
}

void renderWithCuda(float* fb, unsigned int nx, unsigned int ny, unsigned int tx, unsigned int ty, size_t fb_size, unsigned int aa_iter, unsigned int ref_iter, unsigned int gl_iter, unsigned int ind_rays, Camera* d_cam, Geometry** d_world, Light** d_lights, curandState* rand_state);
__global__ void render(float* fb, unsigned int max_x, unsigned int max_y, unsigned int aa_iter, unsigned int ref_iter, unsigned int gl_iter, unsigned int ind_rays, Camera* cam, Geometry** world, Light** lights, curandState* rand_state) {
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    size_t j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    size_t pixel_index = (j * max_x + i);
    size_t idx = pixel_index;
    curandState local_rand_state = rand_state[idx];
    pixel_index *= 4;
    vec4 pos_in_world = get_world_coordinates(i, j, max_x, max_y);
    vec2 pixel_size = vec2(pos_in_world.z() / (float)max_x, pos_in_world.w() / (float)max_y);
    vec4 c = aa_color(pos_in_world.x(), pos_in_world.y(), pixel_size, pos_in_world.z(), pos_in_world.w(), aa_iter, ref_iter, gl_iter, ind_rays, cam, world, lights, &local_rand_state);
    c.saturate();
    //Ray r = cam->getRay(pos_in_world.x(), pos_in_world.y(), pos_in_world.z(), pos_in_world.w());
    //vec4 c = color(r, world, lights, &local_rand_state);
    fb[pixel_index + 0] = c.r();
    fb[pixel_index + 1] = c.g();
    fb[pixel_index + 2] = c.b();
    fb[pixel_index + 3] = c.a();
    rand_state[idx] = local_rand_state;
}

__global__ void create_world(Camera** d_cam, Geometry** d_glist, Geometry** d_gworld, Light** d_llist, Light** d_lworld, unsigned int shadowSamples) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Camera
        *d_cam = new Camera(vec3::zero(), vec3(0.f, 0.f, -1.f), CameraType::PERSPECTIVE, deg_to_rad(45.f), 2.f);

        // AreaLight Position
        vec3 quadPoints[4] = {
            vec3(-.25f, .98f, -1.25f),
            vec3(.25f, .98f, -1.25f),
            vec3(.25f, .98f, -.75f),
            vec3(-.25f, .98f, -.75f)
        };

        // Materials
        Material reflect =
        {
#if _HAS_CXX20
            .type = MaterialType::Reflect,
            .ambient = Color::white() * 0.1f,
            .diffuse = Color::white(),
            .specular = Color::white(),
            .shininess = 0.f
#else
            MaterialType::Reflect,
            Color::white() * 0.1f,
            Color::white(),
            Color::white(),
            0.f
#endif
        };

        Material refractive =
        {
#if _HAS_CXX20
            .type = MaterialType::Refractive,
            .ambient = Color::white() * 0.1f,
            .diffuse = Color::white(),
            .specular = Color::white(),
            .shininess = 0.f,
            .refractIndex = 1.5f
#else
            MaterialType::Refractive,
            Color::white() * 0.1f,
            Color::white(),
            Color::white(),
            0.f,
            1.5f
#endif
        };

        Material whiteEmissive =
        {
#if _HAS_CXX20
            .type = MaterialType::Diffuse,
            .ambient = Color::white(),
            .diffuse = Color::white(),
            .specular = Color::white(),
            .shininess = 0.f
#else
            MaterialType::Diffuse,
            Color::white(),
            Color::white(),
            Color::white(),
            0.f
#endif
        };

        Material white = {
#if _HAS_CXX20
            .type = MaterialType::Diffuse,
            .ambient = Color::white() * 0.1f,
            .diffuse = Color::white(),
            .specular = Color::white(),
            .shininess = 5.f
#else
            MaterialType::Diffuse,
            Color::white() * 0.1f,
            Color::white(),
            Color::white(),
            5.f
#endif
        };

        Material red =
        {
#if _HAS_CXX20
            .type = MaterialType::Diffuse,
            .ambient = Color::red() * 0.1f,
            .diffuse = Color::red(),
            .specular = Color::red(),
            .shininess = 5.f
#else
            MaterialType::Diffuse,
            Color::red() * 0.1f,
            Color::red(),
            Color::red(),
            5.f
#endif
        };

        Material blue =
        {
#if _HAS_CXX20
            .type = MaterialType::Diffuse,
            .ambient = Color::blue() * 0.1f,
            .diffuse = Color::blue(),
            .specular = Color::blue(),
            .shininess = 5.f
#else
            MaterialType::Diffuse,
            Color::blue() * 0.1f,
            Color::blue(),
            Color::blue(),
            5.f
#endif
        };

        d_glist[0] = new Sphere(vec3(-0.25f, -.72f, -1.1f), .275f); // REFLECT
        d_glist[0]->setMaterial(reflect);

        d_glist[1] = new Sphere(vec3(0.3f, -.72f, -.6f), .275f); // REFRACT
        d_glist[1]->setMaterial(refractive);

        d_glist[2] = new Quad // BACK
        (
#if _HAS_CXX20
            { .pos = vec3(-1.f, -1.f, -2.f), .tex = vec2(0.f, 0.f), .normal = vec3(0.f, 0.f, 1.f) },
            { .pos = vec3( 1.f, -1.f, -2.f), .tex = vec2(1.f, 0.f), .normal = vec3(0.f, 0.f, 1.f) },
            { .pos = vec3( 1.f,  1.f, -2.f), .tex = vec2(1.f, 1.f), .normal = vec3(0.f, 0.f, 1.f) },
            { .pos = vec3(-1.f,  1.f, -2.f), .tex = vec2(0.f, 1.f), .normal = vec3(0.f, 0.f, 1.f) }
#else
            { vec3(-1.f, -1.f, -2.f), vec2(0.f, 0.f), vec3(0.f, 0.f, 1.f) },
            { vec3( 1.f, -1.f, -2.f), vec2(1.f, 0.f), vec3(0.f, 0.f, 1.f) },
            { vec3( 1.f,  1.f, -2.f), vec2(1.f, 1.f), vec3(0.f, 0.f, 1.f) },
            { vec3(-1.f,  1.f, -2.f), vec2(0.f, 1.f), vec3(0.f, 0.f, 1.f) }
#endif
        );
        d_glist[2]->setMaterial(white);

        d_glist[3] = new Quad // TOP
        (
#if _HAS_CXX20
            { .pos = vec3(-1.f, 1.f, -2.f), .tex = vec2(0.f, 0.f), .normal = vec3(0.f, -1.f, 0.f) },
            { .pos = vec3( 1.f, 1.f, -2.f), .tex = vec2(1.f, 0.f), .normal = vec3(0.f, -1.f, 0.f) },
            { .pos = vec3( 1.f, 1.f,  0.f), .tex = vec2(1.f, 1.f), .normal = vec3(0.f, -1.f, 0.f) },
            { .pos = vec3(-1.f, 1.f,  0.f), .tex = vec2(0.f, 1.f), .normal = vec3(0.f, -1.f, 0.f) }
#else
            { vec3(-1.f, 1.f, -2.f), vec2(0.f, 0.f), vec3(0.f, -1.f, 0.f) },
            { vec3( 1.f, 1.f, -2.f), vec2(1.f, 0.f), vec3(0.f, -1.f, 0.f) },
            { vec3( 1.f, 1.f,  0.f), vec2(1.f, 1.f), vec3(0.f, -1.f, 0.f) },
            { vec3(-1.f, 1.f,  0.f), vec2(0.f, 1.f), vec3(0.f, -1.f, 0.f) }
#endif
        );
        d_glist[3]->setMaterial(white);

        d_glist[4] = new Quad // BOTTOM
        (
#if _HAS_CXX20
            { .pos = vec3(-1.f, -1.f, -2.f), .tex = vec2(0.f, 0.f), .normal = vec3(0.f, 1.f, 0.f) },
            { .pos = vec3( 1.f, -1.f, -2.f), .tex = vec2(1.f, 0.f), .normal = vec3(0.f, 1.f, 0.f) },
            { .pos = vec3( 1.f, -1.f,  0.f), .tex = vec2(1.f, 1.f), .normal = vec3(0.f, 1.f, 0.f) },
            { .pos = vec3(-1.f, -1.f,  0.f), .tex = vec2(0.f, 1.f), .normal = vec3(0.f, 1.f, 0.f) }
#else
            { vec3(-1.f, -1.f, -2.f), vec2(0.f, 0.f), vec3(0.f, 1.f, 0.f) },
            { vec3( 1.f, -1.f, -2.f), vec2(1.f, 0.f), vec3(0.f, 1.f, 0.f) },
            { vec3( 1.f, -1.f,  0.f), vec2(1.f, 1.f), vec3(0.f, 1.f, 0.f) },
            { vec3(-1.f, -1.f,  0.f), vec2(0.f, 1.f), vec3(0.f, 1.f, 0.f) }
#endif
        );
        d_glist[4]->setMaterial(white);

        d_glist[5] = new Quad // RIGHT
        (
#if _HAS_CXX20
            { .pos = vec3(1.f, -1.f, -2.f), .tex = vec2(0.f, 0.f), .normal = vec3(-1.f, 0.f, 0.f) },
            { .pos = vec3(1.f, -1.f,  0.f), .tex = vec2(1.f, 0.f), .normal = vec3(-1.f, 0.f, 0.f) },
            { .pos = vec3(1.f,  1.f,  0.f), .tex = vec2(1.f, 1.f), .normal = vec3(-1.f, 0.f, 0.f) },
            { .pos = vec3(1.f,  1.f, -2.f), .tex = vec2(0.f, 1.f), .normal = vec3(-1.f, 0.f, 0.f) }
#else
            { vec3(1.f, -1.f, -2.f), vec2(0.f, 0.f), vec3(-1.f, 0.f, 0.f) },
            { vec3(1.f, -1.f,  0.f), vec2(1.f, 0.f), vec3(-1.f, 0.f, 0.f) },
            { vec3(1.f,  1.f,  0.f), vec2(1.f, 1.f), vec3(-1.f, 0.f, 0.f) },
            { vec3(1.f,  1.f, -2.f), vec2(0.f, 1.f), vec3(-1.f, 0.f, 0.f) }
#endif
        );
        d_glist[5]->setMaterial(blue);

        d_glist[6] = new Quad // LEFT
        (
#if _HAS_CXX20
            { .pos = vec3(-1.f, -1.f, -2.f), .tex = vec2(0.f, 0.f), .normal = vec3(1.f, 0.f, 0.f) },
            { .pos = vec3(-1.f,  1.f, -2.f), .tex = vec2(1.f, 0.f), .normal = vec3(1.f, 0.f, 0.f) },
            { .pos = vec3(-1.f,  1.f,  0.f), .tex = vec2(1.f, 1.f), .normal = vec3(1.f, 0.f, 0.f) },
            { .pos = vec3(-1.f, -1.f,  0.f), .tex = vec2(0.f, 1.f), .normal = vec3(1.f, 0.f, 0.f) }
#else
            { vec3(-1.f, -1.f, -2.f), vec2(0.f, 0.f), vec3(1.f, 0.f, 0.f) },
            { vec3(-1.f,  1.f, -2.f), vec2(1.f, 0.f), vec3(1.f, 0.f, 0.f) },
            { vec3(-1.f,  1.f,  0.f), vec2(1.f, 1.f), vec3(1.f, 0.f, 0.f) },
            { vec3(-1.f, -1.f,  0.f), vec2(0.f, 1.f), vec3(1.f, 0.f, 0.f) }
#endif
        );
        d_glist[6]->setMaterial(red);

        d_glist[7] = new Quad // LIGHT
        (
#if _HAS_CXX20
            { .pos = vec3(quadPoints[0].x(), quadPoints[0].y() + 0.01f, quadPoints[0].z()), .tex = vec2(0.f, 0.f), .normal = vec3(0.f, -1.f, 0.f)},
            { .pos = vec3(quadPoints[1].x(), quadPoints[1].y() + 0.01f, quadPoints[1].z()), .tex = vec2(1.f, 0.f), .normal = vec3(0.f, -1.f, 0.f) },
            { .pos = vec3(quadPoints[2].x(), quadPoints[2].y() + 0.01f, quadPoints[2].z()), .tex = vec2(1.f, 1.f), .normal = vec3(0.f, -1.f, 0.f) },
            { .pos = vec3(quadPoints[3].x(), quadPoints[3].y() + 0.01f, quadPoints[3].z()), .tex = vec2(0.f, 1.f), .normal = vec3(0.f, -1.f, 0.f) }
#else
            { vec3(quadPoints[0].x(), quadPoints[0].y() + 0.01f, quadPoints[0].z()), vec2(0.f, 0.f), vec3(0.f, -1.f, 0.f) },
            { vec3(quadPoints[1].x(), quadPoints[1].y() + 0.01f, quadPoints[1].z()), vec2(1.f, 0.f), vec3(0.f, -1.f, 0.f) },
            { vec3(quadPoints[2].x(), quadPoints[2].y() + 0.01f, quadPoints[2].z()), vec2(1.f, 1.f), vec3(0.f, -1.f, 0.f) },
            { vec3(quadPoints[3].x(), quadPoints[3].y() + 0.01f, quadPoints[3].z()), vec2(0.f, 1.f), vec3(0.f, -1.f, 0.f) }
#endif
        );
        d_glist[7]->setMaterial(whiteEmissive);

        *d_gworld = new GeometryList(d_glist, 8);

        d_llist[0] = new AreaLight(quadPoints[0], quadPoints[1], quadPoints[2], quadPoints[3], Color::white(), shadowSamples, 10.f);
        ((AreaLight*)d_llist[0])->rotate(vec3(1.f, 0.f, 0.f), deg_to_rad(180.f));
        *d_lworld = new LightList(d_llist, 1);
    }
}

__global__ void free_world(Camera** d_cam, Geometry** d_glist, Geometry** d_gworld, Light** d_llist, Light** d_lworld) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {        
        delete* d_cam;

        for (int i = 0; i < 8; ++i) {
            delete d_glist[i];
        }

        delete* d_gworld;

        delete d_llist[0];
        delete* d_lworld;
    }
}

void printTime(const char* prefix, float miliseconds);

int main()
{
#pragma region Start

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    fprintf(stdout, "Supports malloc on device: %d\n", props.managedMemory);

    cudaDeviceSetLimit(cudaLimitStackSize, 20480);

    size_t stack_size;
    cudaDeviceGetLimit(&stack_size, cudaLimitStackSize);
    fprintf(stdout, "Default stack size: %zu bytes\n", stack_size);

    int maxThreadsPerBlock;
    cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);
    fprintf(stdout, "Max threads per block: %d\n", props.maxThreadsPerBlock);

    // Choose which GPU to run on, change this on a multi-GPU system.
    checkCudaErrors(cudaSetDevice(0));

#pragma endregion

#pragma region Parameters

    unsigned int nx = 720;
    unsigned int ny = 720;
    unsigned int tx = 19; // Optimized
    unsigned int ty = 19; // Optimized
    unsigned int aa_iter = 1; // Optimized
    unsigned int ref_iter = 4; // Optimized
    unsigned int gl_iter = 5; // Can also be 4 but its darker
    unsigned int ind_rays = 75; // I think its good enough
    unsigned int shadowSamples = 50; // Optimized
    unsigned int num_pixels = nx * ny;
    size_t fb_size = 4 * num_pixels * sizeof(float);

#pragma endregion

#pragma region Time

    cudaEvent_t start, stop;
    float elapsed_time = 0.0f;

    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

#pragma endregion

#pragma region Allocations

    checkCudaErrors(cudaEventRecord(start, 0));

    // Random Init
    curandState* d_rand_state;
    checkCudaErrors(cudaMallocManaged((void**)&d_rand_state, num_pixels * sizeof(curandState)));
    randomInit(nx, ny, tx, ty, d_rand_state);

    // Create World
    Camera** d_cam;
    checkCudaErrors(cudaMallocManaged((void**)&d_cam, sizeof(Camera*)));

    Geometry** d_glist;
    checkCudaErrors(cudaMallocManaged((void**)&d_glist, 8 * sizeof(Geometry*)));

    Geometry** d_gworld;
    checkCudaErrors(cudaMallocManaged((void**)&d_gworld, sizeof(Geometry*)));

    Light** d_llist;
    checkCudaErrors(cudaMallocManaged((void**)&d_llist, 1 * sizeof(Light*)));

    Light** d_lworld;
    checkCudaErrors(cudaMallocManaged((void**)&d_lworld, sizeof(Light*)));

    // Create our world
    create_world<<<1, 1>>>(d_cam, d_glist, d_gworld, d_llist, d_lworld, shadowSamples);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Allocate FB
    float* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsed_time, start, stop));

    printTime("Create World", elapsed_time);

#pragma endregion

#pragma region Render

    checkCudaErrors(cudaEventRecord(start, 0));

    renderWithCuda(fb, nx, ny, tx, ty, fb_size, aa_iter, ref_iter, gl_iter, ind_rays, *d_cam, d_gworld, d_lworld, d_rand_state);

    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsed_time, start, stop));

    printTime("Render", elapsed_time);

#pragma endregion

#pragma region Save

    // Save Image
    fprintf(stdout, "Saving image...");
    stbi_write_hdr("file.hdr", nx, ny, 4, fb);

#pragma endregion

#pragma region Cleanup

    // Free our world
    free_world<<<1, 1>>>(d_cam, d_glist, d_gworld, d_llist, d_lworld);

    // Check for any errors launching the kernel
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    // Free all cuda resources
    checkCudaErrors(cudaFree(d_cam));
    checkCudaErrors(cudaFree(d_glist));
    checkCudaErrors(cudaFree(d_gworld));
    checkCudaErrors(cudaFree(d_llist));
    checkCudaErrors(cudaFree(d_lworld));
    checkCudaErrors(cudaFree(fb));

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    checkCudaErrors(cudaDeviceReset());

#pragma endregion

#pragma region End

    fprintf(stdout, "\nImage saved as 'file.hdr'. Press Enter to exit...");
    getchar();

    return 0;

#pragma endregion
}

// Helper function for setting up CUDA random number generator.
void randomInit(unsigned int nx, unsigned int ny, unsigned int tx, unsigned int ty, curandState* rand_state)
{
    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);

    // Choose which GPU to run on, change this on a multi-GPU system.
    checkCudaErrors(cudaSetDevice(0));

    // Render our buffer
    random_init<<<blocks, threads>>>(nx, ny, time(NULL), rand_state);

    // Check for any errors launching the kernel
    checkCudaErrors(cudaGetLastError());

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    checkCudaErrors(cudaDeviceSynchronize());
}

// Helper function for using CUDA to render image in parallel.
void renderWithCuda(float* fb, unsigned int nx, unsigned int ny, unsigned int tx, unsigned int ty, size_t fb_size, unsigned int aa_iter, unsigned int ref_iter, unsigned int gl_iter, unsigned int ind_rays, Camera* d_cam, Geometry** d_world, Light** d_lights, curandState* rand_state)
{
    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);

    // Choose which GPU to run on, change this on a multi-GPU system.
    checkCudaErrors(cudaSetDevice(0));

    // Render our buffer
    render<<<blocks, threads>>>(fb, nx, ny, aa_iter, ref_iter, gl_iter, ind_rays, d_cam, d_world, d_lights, rand_state);

    // Check for any errors launching the kernel
    checkCudaErrors(cudaGetLastError());

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    checkCudaErrors(cudaDeviceSynchronize());
}

void printTime(const char* prefix, float miliseconds) {
    // Konwersja na godziny, minuty, sekundy, milisekundy
    int total_ms = static_cast<int>(miliseconds);
    int hours = total_ms / (1000 * 60 * 60);
    int minutes = (total_ms / (1000 * 60)) % 60;
    int seconds = (total_ms / 1000) % 60;
    int milliseconds = total_ms % 1000;

    fprintf(stdout, "%s time: %02d:%02d:%02d.%03d (hh:mm:ss.mmm)\n",
        prefix, hours, minutes, seconds, milliseconds);
}