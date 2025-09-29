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
    struct KDNode {
        AABB bounds;
        int axis;       // -1 - leaf, 0 - x, 1 - y, 2 - z
        int left;       // left child index, -1 if none
        int right;      // right child index, -1 if none
        int leaf_first; // first index in list of indexes of objects
        int leaf_count; // count of list of indexes of objects
    };

    class KDTreeBuilder {
    private:
        SceneDescription* scene;
        int max_leaf_size;
        int max_depth;

    public:
        std::vector<KDNode> nodes;
        std::vector<int> indexes;

        KDTreeBuilder(SceneDescription* desc, int max_objects_per_leaf = 2, int max_tree_depth = 32)
        { 
            scene = desc;
            max_leaf_size = max_objects_per_leaf; 
            max_depth = max_tree_depth; 
        }

        ~KDTreeBuilder() { nodes.clear(); indexes.clear(); }

        void build() {
            printf("Building KDTree...\n");

            const int objects_count = scene->objectsCount;

            indexes.clear();
            nodes.clear();

            struct BuildTask {
                int node_index;
                std::vector<int> obj_indices;
                int depth;
                int axis;
            };

            std::stack<BuildTask> stack;

            std::vector<int> all(objects_count);
            for (int i = 0; i < objects_count; ++i) all[i] = i;

            nodes.push_back({}); // root
            stack.push({ 0, all, 0, 0 });

            while (!stack.empty()) {
                BuildTask task = stack.top();
                stack.pop();

                int node_index = task.node_index;
                std::vector<int> obj_indices = std::move(task.obj_indices);
                int count = (int)obj_indices.size();
                int depth = task.depth;
                int ax = task.axis;

                nodes.reserve(nodes.size() + 2);

                KDNode& node = nodes[node_index];

                AABB bounds;
                for (int i = 0; i < obj_indices.size(); ++i) {
                    bounds.expand(scene->objects[obj_indices[i]].getBounds());
                }
                node.bounds = bounds;

                if (count <= max_leaf_size || depth >= max_depth) {
                    node.axis = -1;
                    node.left = -1;
                    node.right = -1;
                    node.leaf_first = (int)indexes.size();
                    node.leaf_count = count;
                    indexes.insert(indexes.end(), obj_indices.begin(), obj_indices.end());

                    printf("  -> leaf with %d objects\n", count);
                    continue;
                }

                node.axis = ax;

                float split_pos = 0.5f * (bounds.min[ax] + bounds.max[ax]);

                std::vector<int> left_buf, right_buf;
                for (int i = 0; i < obj_indices.size(); ++i) {
                    int idx = obj_indices[i];
                    AABB obj_bounds = scene->objects[idx].getBounds();

                    bool goesLeft = obj_bounds.min[ax] <= split_pos;
                    bool goesRight = obj_bounds.max[ax] >= split_pos;

                    if (goesLeft)  left_buf.push_back(idx);
                    if (goesRight) right_buf.push_back(idx);
                }

                if (left_buf.empty() || right_buf.empty() || left_buf.size() == count || right_buf.size() == count) {
                    node.axis = -1;
                    node.left = -1;
                    node.right = -1;
                    node.leaf_first = (int)indexes.size();
                    node.leaf_count = count;
                    indexes.insert(indexes.end(), obj_indices.begin(), obj_indices.end());

                    printf("  -> leaf with %d objects\n", count);
                    continue;
                }

                node.left = (int)nodes.size();
                nodes.push_back({});
                node.right = (int)nodes.size();
                nodes.push_back({});

                stack.push({ node.left, left_buf, depth + 1, (ax + 1) % 3 });
                stack.push({ node.right, right_buf, depth + 1, (ax + 1) % 3 });
            }

            printf("KDTree builded! nodes=%zu, indexes=%zu\n", nodes.size(), indexes.size());
        }
    };

    class KDTree : public Geometry {
    private:
        Geometry** objects_list;
        int objects_list_size;

        KDNode* nodes;
        int nodes_count;

        int* indexes_list;
        int indexes_list_size;

    public:
        KDTree() = default;
        __device__ KDTree(Geometry** gl, int gn, KDNode* nl, int nn, int* il, int in)
        {
            objects_list = gl;
            objects_list_size = gn;
            nodes = nl;
            nodes_count = nn;
            indexes_list = il;
            indexes_list_size = in;
        }

        __device__ ~KDTree() { delete[] nodes; delete[] indexes_list; }

        __device__ AABB getBounds() const {
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

            if (nodes == nullptr || nodes_count == 0 || nodes[0].axis == -1) {
                for (int i = 0; i < objects_list_size; ++i) {
                    if (objects_list[i]->hit(ray, temp_hit)) {
                        if (temp_hit.hitDist < closest_so_far) {
                            hit_anything = true;
                            closest_so_far = temp_hit.hitDist;

                            hit.isHit = temp_hit.isHit;
                            hit.hitPoint = temp_hit.hitPoint;
                            hit.hitNormal = temp_hit.hitNormal;
                            hit.hitUV = temp_hit.hitUV;
                            hit.hitDist = temp_hit.hitDist;
                            hit.hitMat = temp_hit.hitMat;
                        }
                    }
                }
                return hit_anything;
            }

            const int MAX_STACK = 256;
            
            struct StackEntry
            {
                int node; 
                float tmin; 
                float tmax; 
            };

            StackEntry stack[MAX_STACK];
            int stack_size = 0;

            float root_tmin, root_tmax;
            if (!nodes[0].bounds.hit(ray, root_tmin, root_tmax)) return false;

            stack[stack_size++] = { 0, root_tmin, root_tmax };

            while (stack_size > 0) {
                StackEntry ent = stack[--stack_size];

                if (ent.tmin > closest_so_far) continue;

                const KDNode& node = nodes[ent.node];

                if (node.axis == -1) {
                    for (int i = 0; i < node.leaf_count; ++i) {
                        int obj_idx = indexes_list[node.leaf_first + i];

                        if (objects_list[obj_idx]->hit(ray, temp_hit)) {
                            if (temp_hit.hitDist < closest_so_far) {
                                hit_anything = true;
                                closest_so_far = temp_hit.hitDist;

                                hit.isHit = temp_hit.isHit;
                                hit.hitPoint = temp_hit.hitPoint;
                                hit.hitNormal = temp_hit.hitNormal;
                                hit.hitUV = temp_hit.hitUV;
                                hit.hitDist = temp_hit.hitDist;
                                hit.hitMat = temp_hit.hitMat;
                            }
                        }
                    }
                    continue;
                }

                int axis = node.axis;
                float split_pos = 0.5f * (node.bounds.min[axis] + node.bounds.max[axis]);

                const vec3 origin = ray.getOrigin();
                const vec3 direction = ray.getDirection();

                float origin_axis = (axis == 0 ? origin.x() : (axis == 1 ? origin.y() : origin.z()));
                float dir_axis = (axis == 0 ? direction.x() : (axis == 1 ? direction.y() : direction.z()));

                float t_split = (epsilon_equal(dir_axis, 0.0f, MSTD_EPSILON<float>)) ? FLT_MAX : ((split_pos - origin_axis) / dir_axis);

                int nearChild = dir_axis >= 0.0f ? node.left : node.right;
                int farChild = dir_axis >= 0.0f ? node.right : node.left;

                if (t_split <= ent.tmin) {
                    if (farChild != -1) {
                        stack[stack_size++] = { farChild, ent.tmin, ent.tmax };
                    }
                }
                else if (t_split >= ent.tmax) {
                    if (nearChild != -1) {
                        stack[stack_size++] = { nearChild, ent.tmin, ent.tmax };
                    }
                }
                else {
                    if (farChild != -1) {
                        stack[stack_size++] = { farChild, t_split, ent.tmax };
                    }
                    if (nearChild != -1) {
                        stack[stack_size++] = { nearChild, ent.tmin, t_split };
                    }
                }
            }

            return hit_anything;
        }
    };
}