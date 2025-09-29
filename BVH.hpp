/**************************************************************
 *                                                            *
 *  Project:   CudaRayTracer                                  *
 *  Authors:   Muppetsg2 & MAIPA01                            *
 *  License:   MIT License                                    *
 *  Last Update: 29.09.2025                                   *
 *                                                            *
 **************************************************************/

#pragma once

#include "Geometry.hpp"
#include "SceneDescription.hpp"
#include <stack>

using namespace MSTD_NAMESPACE;

namespace craytracer {
    struct BVHNode {
        AABB bounds;
        int left;       // index child (if internal)
        int right;      // index child (if internal)
        int start;      // start index in prim_indices (if leaf)
        int count;      // number of primitives (if leaf)

        bool isLeaf() const { return left == -1; }
    };

    class BVHBuilder {
    private:
        const int num_bins = 16;
        SceneDescription* scene;
        int max_leaf_size;
        int max_depth;

        void buildRecursive(std::vector<int>& prims, const std::vector<vec3>& centroids) {
            struct BuildEntry {
                int start, end;
                int nodeIndex;
                int depth;
                AABB bounds;
                vec3 centroidMin;
                vec3 centroidMax;
            };

            std::vector<int> work_prims = prims;
            std::vector<vec3> work_centroids = centroids;

            AABB root_bounds;
            AABB centroid_bounds;
            for (int i = 0; i < (int)work_prims.size(); ++i) {
                AABB pb = scene->objects[work_prims[i]].getBounds();
                root_bounds.expand(pb);
                centroid_bounds.expand(work_centroids[i]);
            }

            std::stack<BuildEntry> stack;
            nodes.push_back(BVHNode());
            BuildEntry rootEntry;
            rootEntry.start = 0;
            rootEntry.end = (int)work_prims.size();
            rootEntry.nodeIndex = 0;
            rootEntry.depth = 0;
            rootEntry.bounds = root_bounds;
            rootEntry.centroidMin = centroid_bounds.min;
            rootEntry.centroidMax = centroid_bounds.max;
            stack.push(rootEntry);

            auto surfaceAreaSafe = [](const AABB& b) {
                float sa = b.surfaceArea();
                return sa <= 0.0f ? 1e-6f : sa;
            };

            while (!stack.empty()) {
                BuildEntry e = stack.top();
                stack.pop();

                int start = e.start;
                int end = e.end;
                int count = end - start;
                int nodeIdx = e.nodeIndex;
                int depth = e.depth;
                AABB node_bounds = e.bounds;

                BVHNode& node = nodes[nodeIdx];

                if (count <= max_leaf_size || depth >= max_depth) {
                    node.left = -1;
                    node.right = -1;
                    node.start = (int)indexes.size();
                    node.count = count;
                    for (int i = start; i < end; ++i) indexes.push_back(work_prims[i]);
                    node.bounds = node_bounds;
                    continue;
                }

                vec3 cmin(FLT_MAX, FLT_MAX, FLT_MAX);
                vec3 cmax(-FLT_MAX, -FLT_MAX, -FLT_MAX);
                for (int i = start; i < end; ++i) {
                    vec3 c = work_centroids[i];
                    if (c.x() < cmin.x()) cmin.x() = c.x();
                    if (c.y() < cmin.y()) cmin.y() = c.y();
                    if (c.z() < cmin.z()) cmin.z() = c.z();
                    if (c.x() > cmax.x()) cmax.x() = c.x();
                    if (c.y() > cmax.y()) cmax.y() = c.y();
                    if (c.z() > cmax.z()) cmax.z() = c.z();
                }

                int axis = 0;
                float ex0 = cmax.x() - cmin.x();
                float ex1 = cmax.y() - cmin.y();
                float ex2 = cmax.z() - cmin.z();
                if (ex1 > ex0) axis = 1;
                if (ex2 > (axis == 0 ? ex0 : ex1)) axis = 2;

                float axisExtent = (axis == 0 ? ex0 : (axis == 1 ? ex1 : ex2));
                if (axisExtent <= 1e-9f) {
                    node.left = -1;
                    node.right = -1;
                    node.start = (int)indexes.size();
                    node.count = count;
                    node.bounds = node_bounds;
                    for (int i = start; i < end; ++i) indexes.push_back(work_prims[i]);
                    continue;
                }

                struct Bin {
                    AABB bounds;
                    int count = 0;
                };
                std::vector<Bin> bins(num_bins);

                float axisMin = (axis == 0 ? cmin.x() : (axis == 1 ? cmin.y() : cmin.z()));
                float axisMax = (axis == 0 ? cmax.x() : (axis == 1 ? cmax.y() : cmax.z()));
                float scale = axisMax > axisMin ? (float)num_bins / (axisMax - axisMin) : 0.0f;

                for (int i = start; i < end; ++i) {
                    vec3 c = work_centroids[i];
                    float ca = (axis == 0 ? c.x() : (axis == 1 ? c.y() : c.z()));
                    int b = (int)((ca - axisMin) * scale);
                    if (b < 0) b = 0;
                    if (b >= num_bins) b = num_bins - 1;
                    bins[b].count++;
                    bins[b].bounds.expand(scene->objects[work_prims[i]].getBounds());
                }

                std::vector<AABB> leftBounds(num_bins - 1);
                std::vector<int> leftCounts(num_bins - 1);
                AABB accumL;
                int cntL = 0;
                for (int i = 0; i < num_bins - 1; ++i) {
                    if (bins[i].count > 0) accumL.expand(bins[i].bounds);
                    cntL += bins[i].count;
                    leftBounds[i] = accumL;
                    leftCounts[i] = cntL;
                }

                std::vector<AABB> rightBounds(num_bins - 1);
                std::vector<int> rightCounts(num_bins - 1);
                AABB accumR;
                int cntR = 0;
                for (int i = num_bins - 1; i >= 1; --i) {
                    if (bins[i].count > 0) accumR.expand(bins[i].bounds);
                    cntR += bins[i].count;
                    rightBounds[i - 1] = accumR;
                    rightCounts[i - 1] = cntR;
                }

                float parentArea = surfaceAreaSafe(node_bounds);
                float bestCost = FLT_MAX;
                int bestBin = -1;
                for (int i = 0; i < num_bins - 1; ++i) {
                    if (leftCounts[i] == 0 || rightCounts[i] == 0) continue;
                    float leftA = surfaceAreaSafe(leftBounds[i]);
                    float rightA = surfaceAreaSafe(rightBounds[i]);
                    float cost = 1.0f + (leftA / parentArea) * leftCounts[i] + (rightA / parentArea) * rightCounts[i];
                    if (cost < bestCost) {
                        bestCost = cost;
                        bestBin = i;
                    }
                }

                float leafCost = (float)count;

                printf("Node %d depth=%d: count=%d leafCost=%.3f bestCost=%.3f bestBin=%d axis=%d\n",
                        nodeIdx, depth, count, leafCost, bestCost, bestBin, axis);

                if (bestBin == -1 || bestCost >= leafCost) {
                    node.left = -1;
                    node.right = -1;
                    node.start = (int)indexes.size();
                    node.count = count;
                    node.bounds = node_bounds;
                    for (int i = start; i < end; ++i) indexes.push_back(work_prims[i]);
                    continue;
                }

                float splitCoord = axisMin + (axisMax - axisMin) * ((bestBin + 1) / (float)num_bins);

                int mid = partitionBySplit(work_prims, work_centroids, start, end, axis, splitCoord);

                if (mid == start || mid == end) {
                    node.left = -1;
                    node.right = -1;
                    node.start = (int)indexes.size();
                    node.count = count;
                    node.bounds = node_bounds;
                    for (int i = start; i < end; ++i) indexes.push_back(work_prims[i]);
                    continue;
                }

                node.bounds = node_bounds;
                node.left = (int)nodes.size();
                nodes.push_back(BVHNode());
                node.right = (int)nodes.size();
                nodes.push_back(BVHNode());

                BuildEntry leftEntry;
                leftEntry.start = start;
                leftEntry.end = mid;
                leftEntry.nodeIndex = node.left;
                leftEntry.depth = depth + 1;
                leftEntry.bounds = AABB();
                vec3 leftCmin(FLT_MAX, FLT_MAX, FLT_MAX), leftCmax(-FLT_MAX, -FLT_MAX, -FLT_MAX);
                for (int i = leftEntry.start; i < leftEntry.end; ++i) {
                    AABB pb = scene->objects[work_prims[i]].getBounds();
                    leftEntry.bounds.expand(pb);
                    vec3 c = work_centroids[i];
                    if (c.x() < leftCmin.x()) leftCmin.x() = c.x();
                    if (c.y() < leftCmin.y()) leftCmin.y() = c.y();
                    if (c.z() < leftCmin.z()) leftCmin.z() = c.z();
                    if (c.x() > leftCmax.x()) leftCmax.x() = c.x();
                    if (c.y() > leftCmax.y()) leftCmax.y() = c.y();
                    if (c.z() > leftCmax.z()) leftCmax.z() = c.z();
                }
                leftEntry.centroidMin = leftCmin;
                leftEntry.centroidMax = leftCmax;

                BuildEntry rightEntry;
                rightEntry.start = mid;
                rightEntry.end = end;
                rightEntry.nodeIndex = node.right;
                rightEntry.depth = depth + 1;
                rightEntry.bounds = AABB();
                vec3 rightCmin(FLT_MAX, FLT_MAX, FLT_MAX), rightCmax(-FLT_MAX, -FLT_MAX, -FLT_MAX);
                for (int i = rightEntry.start; i < rightEntry.end; ++i) {
                    AABB pb = scene->objects[work_prims[i]].getBounds();
                    rightEntry.bounds.expand(pb);
                    vec3 c = work_centroids[i];
                    if (c.x() < rightCmin.x()) rightCmin.x() = c.x();
                    if (c.y() < rightCmin.y()) rightCmin.y() = c.y();
                    if (c.z() < rightCmin.z()) rightCmin.z() = c.z();
                    if (c.x() > rightCmax.x()) rightCmax.x() = c.x();
                    if (c.y() > rightCmax.y()) rightCmax.y() = c.y();
                    if (c.z() > rightCmax.z()) rightCmax.z() = c.z();
                }
                rightEntry.centroidMin = rightCmin;
                rightEntry.centroidMax = rightCmax;

                stack.push(rightEntry);
                stack.push(leftEntry);
            }

        }

        int partitionBySplit(std::vector<int>& work_prims, std::vector<vec3>& work_centroids, int start, int end, int axis, float splitCoord)
        {
            int i = start;
            for (int j = start; j < end; ++j) {
                float c = (axis == 0 ? work_centroids[j].x() : (axis == 1 ? work_centroids[j].y() : work_centroids[j].z()));
                if (c <= splitCoord) {
                    if (i != j) {
                        std::swap(work_prims[i], work_prims[j]);
                        std::swap(work_centroids[i], work_centroids[j]);
                    }
                    ++i;
                }
            }
            return i;
        }

    public:
        std::vector<BVHNode> nodes;
        std::vector<int> indexes;

        BVHBuilder(SceneDescription* desc, int max_objects_per_leaf = 4, int max_tree_depth = 32)
        {
            scene = desc;
            max_leaf_size = max_objects_per_leaf;
            max_depth = max_tree_depth;
        }

        ~BVHBuilder() { nodes.clear(); indexes.clear(); }

        void build() {
            printf("Building BVH...\n");

            const int objects_count = scene->objectsCount;

            indexes.clear();
            indexes.reserve(objects_count);

            std::vector<int> prims(objects_count);
            std::vector<vec3> centroids(objects_count);
            for (int i = 0; i < objects_count; ++i) {
                prims[i] = i;
                AABB b = scene->objects[i].getBounds();
                centroids[i] = (b.min + b.max) * 0.5f;
            }

            nodes.clear();
            nodes.reserve(objects_count * 2);

            buildRecursive(prims, centroids);

            printf("BVH builded! nodes=%zu, indexes=%zu\n", nodes.size(), indexes.size());
        }
    };

    class BVH : public Geometry {
    private:
        Geometry** objects_list;
        int objects_list_size;

        BVHNode* nodes;
        int nodes_count;

        int* indexes_list;      // lista indeksów obiektów
        int indexes_list_size;

    public:
        BVH() = default;

        __device__ BVH(Geometry** gl, int gn, BVHNode* nl, int nn, int* il, int in) {
            objects_list = gl;
            objects_list_size = gn;
            nodes = nl;
            nodes_count = nn;
            indexes_list = il;
            indexes_list_size = in;
        }

        __device__ ~BVH() {
            delete[] nodes;
            delete[] indexes_list;
        }

        __device__ AABB getBounds() const {
            if (nodes_count > 0) {
                return nodes[0].bounds;
            }
            AABB bounds;
            for (int i = 0; i < objects_list_size; ++i) {
                bounds.expand(objects_list[i]->getBounds());
            }
            return bounds;
        }

        __device__ bool hit(const Ray& ray, RayHit& hit) const {
            RayHit temp_hit;
            bool hit_anything = false;
            float closest_so_far = FLT_MAX;

            if (nodes == nullptr || nodes_count == 0) {
                for (int i = 0; i < objects_list_size; ++i) {
                    if (objects_list[i]->hit(ray, temp_hit)) {
                        if (temp_hit.hitDist < closest_so_far) {
                            hit_anything = true;
                            closest_so_far = temp_hit.hitDist;
                            hit = temp_hit;
                        }
                    }
                }
                return hit_anything;
            }

            const int MAX_STACK = 256;
            int stack[MAX_STACK];
            int stack_size = 0;

            float tmin_root, tmax_root;
            if (!nodes[0].bounds.hit(ray, tmin_root, tmax_root)) return false;

            stack[stack_size++] = 0;

            while (stack_size > 0) {
                int node_idx = stack[--stack_size];
                const BVHNode& node = nodes[node_idx];

                float tmin, tmax;
                if (!node.bounds.hit(ray, tmin, tmax) || tmin > closest_so_far) continue;

                if (node.count > 0) {
                    for (int i = 0; i < node.count; ++i) {
                        int obj_idx = indexes_list[node.start + i];
                        if (objects_list[obj_idx]->hit(ray, temp_hit)) {
                            if (temp_hit.hitDist < closest_so_far) {
                                hit_anything = true;
                                closest_so_far = temp_hit.hitDist;
                                hit = temp_hit;
                            }
                        }
                    }
                }
                else {
                    if (node.left != -1) stack[stack_size++] = node.left;
                    if (node.right != -1) stack[stack_size++] = node.right;
                }
            }

            return hit_anything;
        }
    };
}