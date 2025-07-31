#pragma once
#include "Geometry.hpp"

namespace craytracer {
    class GeometryList : public Geometry {
    private:
        Geometry** list;
        int list_size;
    public:
        GeometryList() = default;
        __device__ GeometryList(Geometry** l, int n) { list = l; list_size = n; }

        __device__ bool hit(const Ray& ray, RayHit& hit) const {
            RayHit temp_hit;
            bool hit_anything = false;
            float closest_so_far = FLT_MAX;
            for (int i = 0; i < list_size; ++i) {
                if (list[i]->hit(ray, temp_hit)) {
                    if (temp_hit.hitDist > closest_so_far) continue;
                    hit_anything = true;
                    closest_so_far = temp_hit.hitDist;
                    hit = temp_hit;
                }
            }
            return hit_anything;
        }
    };
}