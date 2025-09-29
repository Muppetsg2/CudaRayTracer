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

    enum class KDTreeSplitType : uint8_t {
        Centroid = 0,
        SAH = 1
    };

    class KDTreeBuilder {
    private:
        KDTreeSplitType type;
        SceneDescription* scene;
        int max_leaf_size;
        int max_depth;

    public:
        std::vector<KDNode> nodes;
        std::vector<int> indexes;

        KDTreeBuilder(SceneDescription* desc, int max_objects_per_leaf = 2, int max_tree_depth = 32, KDTreeSplitType split_type = KDTreeSplitType::SAH)
        { 
            scene = desc;
            max_leaf_size = max_objects_per_leaf; 
            max_depth = max_tree_depth; 
            type = split_type;
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
            };

            std::stack<BuildTask> stack;

            std::vector<int> all(objects_count);
            for (int i = 0; i < objects_count; ++i) all[i] = i;

            nodes.push_back({}); // root
            stack.push({ 0, all, 0 });

            while (!stack.empty()) {
                BuildTask task = stack.top();
                stack.pop();

                int node_index = task.node_index;
                std::vector<int> obj_indices = std::move(task.obj_indices);
                int count = (int)obj_indices.size();
                int depth = task.depth;

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
                    continue;
                }

                switch (type) {
                    case KDTreeSplitType::Centroid: {
                        int axis = 0;
                        if (bounds.extent(1) > bounds.extent(axis)) axis = 1;
                        if (bounds.extent(2) > bounds.extent(axis)) axis = 2;
                        node.axis = axis;

                        float split_pos = 0.5f * (bounds.min[axis] + bounds.max[axis]);

                        std::vector<int> left_buf, right_buf;
                        for (int i = 0; i < obj_indices.size(); ++i) {
                            int idx = obj_indices[i];
                            AABB obj_bounds = scene->objects[idx].getBounds();

                            bool goesLeft = obj_bounds.min[axis] <= split_pos;
                            bool goesRight = obj_bounds.max[axis] >= split_pos;

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
                            continue;
                        }

                        node.left = (int)nodes.size();
                        nodes.push_back({});
                        node.right = (int)nodes.size();
                        nodes.push_back({});

                        stack.push({ node.left, left_buf, depth + 1 });
                        stack.push({ node.right, right_buf, depth + 1 });
                        break;
                    }
                    case KDTreeSplitType::SAH: {
                        const int num_bins = 16;
                        float best_cost = FLT_MAX;
                        int best_axis = -1;
                        float best_split = 0.0f;
                        std::vector<int> best_left, best_right;

                        float dx = bounds.max.x() - bounds.min.x();
                        float dy = bounds.max.y() - bounds.min.y();
                        float dz = bounds.max.z() - bounds.min.z();
                        float parent_area = 2.0f * (dx * dy + dx * dz + dy * dz);
                        if (parent_area <= 0) parent_area = 1e-6f;

                        for (int axis = 0; axis < 3; ++axis) {
                            float min_val = bounds.min[axis];
                            float max_val = bounds.max[axis];
                            if (max_val <= min_val) continue;

                            struct Bin {
                                AABB box;
                                int count = 0;
                            };
                            std::vector<Bin> bins(num_bins);

                            for (int idx : obj_indices) {
                                AABB obj_bounds = scene->objects[idx].getBounds();
                                float centroid = 0.5f * (obj_bounds.min[axis] + obj_bounds.max[axis]);
                                int b = int(((centroid - min_val) / (max_val - min_val)) * num_bins);
                                if (b < 0) b = 0;
                                if (b >= num_bins) b = num_bins - 1;
                                bins[b].count++;
                                bins[b].box.expand(obj_bounds);
                            }

                            std::vector<AABB> left_box(num_bins), right_box(num_bins);
                            std::vector<int> left_count(num_bins), right_count(num_bins);

                            AABB tmp;
                            int cnt = 0;
                            for (int i = 0; i < num_bins; ++i) {
                                cnt += bins[i].count;
                                tmp.expand(bins[i].box);
                                left_box[i] = tmp;
                                left_count[i] = cnt;
                            }
                            tmp = AABB();
                            cnt = 0;
                            for (int i = num_bins - 1; i >= 0; i--) {
                                cnt += bins[i].count;
                                tmp.expand(bins[i].box);
                                right_box[i] = tmp;
                                right_count[i] = cnt;
                            }

                            for (int i = 0; i < num_bins - 1; i++) {
                                if (left_count[i] == 0 || right_count[i + 1] == 0) continue;

                                float left_area = left_box[i].surfaceArea();
                                float right_area = right_box[i + 1].surfaceArea();
                                if (left_area <= 0) left_area = 1e-6f;
                                if (right_area <= 0) right_area = 1e-6f;

                                float cost = 1.0f +
                                    (left_area / parent_area) * left_count[i] +
                                    (right_area / parent_area) * right_count[i + 1];

                                if (cost < best_cost) {
                                    best_cost = cost;
                                    best_axis = axis;
                                    best_split = min_val + (max_val - min_val) * ((i + 1) / float(num_bins));
                                }
                            }
                        }

                        float leaf_cost = (float)count;

                        if (best_axis != -1) {
                            printf("Node %d depth=%d: leaf_cost=%.2f, best_cost=%.2f (axis=%d split=%.3f)\n",
                                node_index, depth, leaf_cost, best_cost, best_axis, best_split);
                        }
                        else {
                            printf("Node %d depth=%d: leaf_cost=%.2f, no valid split\n",
                                node_index, depth, leaf_cost);
                        }

                        if (best_axis != -1 && best_cost < leaf_cost) {
                            std::vector<int> left_buf, right_buf;
                            for (int idx : obj_indices) {
                                AABB obj_bounds = scene->objects[idx].getBounds();
                                bool goesLeft = obj_bounds.min[best_axis] <= best_split;
                                bool goesRight = obj_bounds.max[best_axis] >= best_split;
                                if (goesLeft) left_buf.push_back(idx);
                                if (goesRight) right_buf.push_back(idx);
                            }

                            printf("  -> split: left=%d right=%d\n",
                                (int)left_buf.size(), (int)right_buf.size());

                            node.axis = best_axis;
                            node.left = (int)nodes.size();
                            nodes.push_back({});
                            node.right = (int)nodes.size();
                            nodes.push_back({});

                            stack.push({ node.left, std::move(left_buf), depth + 1 });
                            stack.push({ node.right, std::move(right_buf), depth + 1 });
                            continue;
                        }
                        else {
                            printf("  -> leaf with %d objects\n", count);

                            node.axis = -1;
                            node.left = -1;
                            node.right = -1;
                            node.leaf_first = (int)indexes.size();
                            node.leaf_count = count;
                            indexes.insert(indexes.end(), obj_indices.begin(), obj_indices.end());
                        }
                        break;
                    }
                }
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

            if (nodes == nullptr || nodes_count == 0) {
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