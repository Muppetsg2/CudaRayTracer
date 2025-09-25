/**************************************************************
 *                                                            *
 *  Project:   CudaRayTracer                                  *
 *  Authors:   Muppetsg2 & MAIPA01                            *
 *  License:   MIT License                                    *
 *  Last Update: 25.09.2025                                   *
 *                                                            *
 **************************************************************/

#pragma once
#include "vec.hpp"
#include "Camera.hpp"
#include "Material.hpp"
#include "Color.hpp"
#include <yaml-cpp/yaml.h>
#include <functional>
#include <algorithm>
#include <filesystem>

using namespace MSTD_NAMESPACE;

namespace craytracer {
	enum class ObjectType : uint8_t { Sphere, Quad };
	enum class LightType : uint8_t { Area };

    struct CameraDesc {
        CameraType type;
        vec3 position;
        vec3 front;
        float fov;
        float orthoScale;
    };

    struct MaterialDesc {
        unsigned int id;
        MaterialType type;
        vec4 ambient, diffuse, specular;
        float shininess;
        float refractIndex; // for Refraction
    };

    struct SphereDesc {
        vec3 center;
        float radius;
    };

    struct QuadDesc {
        vec3 positions[4];
        vec2 texCoords[4];
        vec3 normals[4];
    };

    struct ObjectDesc {
        ObjectType type;
        unsigned int materialId;
        SphereDesc sphere;
        QuadDesc quad;
    };

    struct AreaLightDesc {
        vec3 points[4];
        vec3 rotationAxis;
        float roationAngle;
        bool twoSided;
    };

    struct LightDesc {
        LightType type;
        vec4 color;
        float intensity;
        AreaLightDesc area;
    };

    struct SceneDescription {
        CameraDesc cam;
        size_t materialsCount;
        MaterialDesc* materials;
        size_t objectsCount;
        ObjectDesc* objects;
        size_t lightsCount;
        LightDesc* lights;

        ~SceneDescription() {
            delete[] materials;
            delete[] objects;
            delete[] lights;
        }
    };

    namespace detail {
        ::std::map<::std::string, ::std::function<vec4(void)>> nameToColor = {
        { "red",[]() -> vec4 { return Color::red(); }},
        { "orange",[]() -> vec4 { return Color::orange(); }},
        { "yellow",[]() -> vec4 { return Color::yellow(); }},
        { "lime",[]() -> vec4 { return Color::lime(); }},
        { "green",[]() -> vec4 { return Color::green(); }},
        { "teal",[]() -> vec4 { return Color::teal(); }},
        { "cyan",[]() -> vec4 { return Color::cyan(); }},
        { "turquoise",[]() -> vec4 { return Color::turquoise(); }},
        { "lapis",[]() -> vec4 { return Color::lapis(); }},
        { "blue",[]() -> vec4 { return Color::blue(); }},
        { "indigo",[]() -> vec4 { return Color::indigo(); }},
        { "violet",[]() -> vec4 { return Color::violet(); }},
        { "purple",[]() -> vec4 { return Color::purple(); }},
        { "magenta",[]() -> vec4 { return Color::magenta(); }},
        { "pink",[]() -> vec4 { return Color::pink(); }},
        { "brown",[]() -> vec4 { return Color::brown(); }},
        { "maroon",[]() -> vec4 { return Color::maroon(); }},
        { "olive",[]() -> vec4 { return Color::olive(); }},
        { "gold",[]() -> vec4 { return Color::gold(); }},
        { "silver",[]() -> vec4 { return Color::silver(); }},
        { "navy",[]() -> vec4 { return Color::navy(); }},
        { "mint",[]() -> vec4 { return Color::mint(); }},
        { "beige",[]() -> vec4 { return Color::beige(); }},
        { "salmon",[]() -> vec4 { return Color::salmon(); }},
        { "coral",[]() -> vec4 { return Color::coral(); }},
        { "white",[]() -> vec4 { return Color::white(); }},
        { "gray",[]() -> vec4 { return Color::gray(); }},
        { "black",[]() -> vec4 { return Color::black(); }}
        };

        vec4 parseColor(const YAML::Node& node) {
            if (node.IsScalar()) {
                ::std::string val = node.as<::std::string>();
                ::std::transform(val.begin(), val.end(), val.begin(), [](unsigned char c) { return ::std::tolower(c); });

                auto it = nameToColor.find(val);
                if (it != nameToColor.end()) {
                    return it->second();
                }
                return vec4::zero();
            }
            if (!node.IsSequence() || node.size() != 4) return vec4::zero();
            return vec4(node[0].as<float>(), node[1].as<float>(), node[2].as<float>(), node[3].as<float>());
        }

        MaterialType parseMaterialType(const ::std::string& s) {
            if (s == "Reflect")    return MaterialType::Reflect;
            if (s == "Refractive") return MaterialType::Refractive;
            if (s == "Diffuse")    return MaterialType::Diffuse;
            return MaterialType::Diffuse;
        }

        ObjectType parseObjectType(const ::std::string& s) {
            if (s == "Sphere") return ObjectType::Sphere;
            if (s == "Quad")   return ObjectType::Quad;
            return ObjectType::Sphere;
        }

        LightType parseLightType(const ::std::string& s) {
            if (s == "Area")  return LightType::Area;
            return LightType::Area;
        }
    }

    SceneDescription* parseScene(const ::std::string& filepath) {
        if (!::std::filesystem::exists(filepath) || !::std::filesystem::is_regular_file(filepath)) {
            fprintf(stderr, "Error while trying to open world file.");
            exit(96);
        }

        YAML::Node config = YAML::LoadFile(filepath);
        SceneDescription* scene = new SceneDescription();

        // Camera
        YAML::Node camNode = config["camera"];
        scene->cam.type = camNode["is_perspective"].as<bool>() ? CameraType::PERSPECTIVE : CameraType::ORTHOGRAPHIC;
        scene->cam.position = vec3(camNode["position"][0].as<float>(), camNode["position"][1].as<float>(), camNode["position"][2].as<float>());
        scene->cam.front = vec3(camNode["front"][0].as<float>(), camNode["front"][1].as<float>(), camNode["front"][2].as<float>());
        scene->cam.fov = camNode["fov"].as<float>();
        scene->cam.orthoScale = camNode["ortho_scale"] ? camNode["ortho_scale"].as<float>() : 2.0f;

        // Materials
        YAML::Node mats = config["materials"];
        scene->materialsCount = mats.size();
        scene->materials = new MaterialDesc[scene->materialsCount];
        for (size_t i = 0; i < scene->materialsCount; ++i) {
            YAML::Node m = mats[i];
            scene->materials[i].id = m["id"].as<unsigned int>();
            ::std::string type = m["type"].as<::std::string>();
            scene->materials[i].type = detail::parseMaterialType(type);

            scene->materials[i].ambient = detail::parseColor(m["ambient"]);
            scene->materials[i].diffuse = detail::parseColor(m["diffuse"]);
            scene->materials[i].specular = detail::parseColor(m["specular"]);
            scene->materials[i].shininess = m["shininess"].as<float>();
            scene->materials[i].refractIndex = m["refract_index"] ? m["refract_index"].as<float>() : 0.0f;
        }

        // Objects
        YAML::Node objs = config["objects"];
        scene->objectsCount = objs.size();
        scene->objects = new ObjectDesc[scene->objectsCount];
        for (size_t i = 0; i < scene->objectsCount; ++i) {
            YAML::Node o = objs[i];
            ObjectType t = detail::parseObjectType(o["type"].as<::std::string>());
            scene->objects[i] = ObjectDesc(t);

            switch (scene->objects[i].type) {
                case ObjectType::Sphere: {
                    YAML::Node c = o["center"];
                    scene->objects[i].sphere.center = vec3(c[0].as<float>(), c[1].as<float>(), c[2].as<float>());
                    scene->objects[i].sphere.radius = o["radius"].as<float>();
                    break;
                }
                case ObjectType::Quad :{
                    scene->objects[i].type = ObjectType::Quad;
                    for (int v = 0; v < 4; ++v) {
                        YAML::Node vv = o["positions"][v];
                        YAML::Node vt = o["tex_coords"][v];
                        YAML::Node vn = o["normals"][v];
                        scene->objects[i].quad.positions[v] = vec3(vv[0].as<float>(), vv[1].as<float>(), vv[2].as<float>());
                        scene->objects[i].quad.texCoords[v] = vec2(vt[0].as<float>(), vt[1].as<float>());
                        scene->objects[i].quad.normals[v] = vec3(vn[0].as<float>(), vn[1].as<float>(), vn[2].as<float>());
                    }
                    break;
                }
            }
            scene->objects[i].materialId = o["material_id"].as<unsigned int>();
        }

        // Lights
        YAML::Node lights = config["lights"];
        scene->lightsCount = lights.size();
        scene->lights = new LightDesc[scene->lightsCount];
        for (size_t i = 0; i < scene->lightsCount; ++i) {
            YAML::Node l = lights[i];
            LightType t = detail::parseLightType(l["type"].as<::std::string>());
            scene->lights[i] = LightDesc(t);

            switch (scene->lights[i].type) {
                case LightType::Area: {
                    for (int v = 0; v < 4; ++v) {
                        YAML::Node vv = l["points"][v];
                        scene->lights[i].area.points[v] = vec3(vv[0].as<float>(), vv[1].as<float>(), vv[2].as<float>());
                    }
                    YAML::Node ra = l["rotation_axis"];
                    scene->lights[i].area.rotationAxis = vec3(ra[0].as<float>(), ra[1].as<float>(), ra[2].as<float>());
                    scene->lights[i].area.roationAngle = l["rotation_angle"].as<float>();
                    scene->lights[i].area.twoSided = l["two_sided"].as<bool>();
                    break;
                }
            }
            scene->lights[i].color = detail::parseColor(l["color"]);
            scene->lights[i].intensity = l["intensity"].as<float>();
        }

        return scene;
    }

    void deleteScene(SceneDescription* scene) {
        delete scene;
    }
}