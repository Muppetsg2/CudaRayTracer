/**************************************************************
 *                                                            *
 *  Project:   CudaRayTracer                                  *
 *  Authors:   Muppetsg2 & MAIPA01                            *
 *  License:   MIT License                                    *
 *  Last Update: 09.10.2025                                   *
 *                                                            *
 **************************************************************/

#pragma once
#include <vector>
#include <string>
#include <filesystem>
#include <functional>
#include <type_traits>
#include <variant>
#include <cstdint>
#include <chrono>
#include <ctime>
#include <iomanip>

namespace fs = ::std::filesystem;

namespace craytracer {
    class Settings {
    private:
        bool _render_all_at_once;
        unsigned int _blocks_per_draw;
        unsigned int _image_width;
        unsigned int _image_height;
        unsigned int _aa_iter;
        unsigned int _ref_iter;
        unsigned int _gl_iter;
        unsigned int _ind_rays;
        unsigned int _shadow_samples;
        ::std::string _file_name;
        fs::path _output_path;
        fs::path _world_file_path;

        enum class ValueType : uint8_t {
            BOOL = 0,
            INT = 1,
            UINT = 2,
            STRING = 3,
            PATH = 4,
            ARRAY_BOOL = 5,
            ARRAY_INT = 6,
            ARRAY_UINT = 7,
            ARRAY_STRING = 8,
            ARRAY_PATH = 9
        };

        template<ValueType T>
        struct Value {
            ::std::string name;

            using type = ::std::conditional_t<T == ValueType::BOOL, bool,
                ::std::conditional_t<T == ValueType::INT, int,
                ::std::conditional_t<T == ValueType::UINT, unsigned int,
                ::std::conditional_t<T == ValueType::STRING, ::std::string,
                ::std::conditional_t<T == ValueType::PATH, fs::path,
                ::std::conditional_t<T == ValueType::ARRAY_BOOL, ::std::vector<bool>,
                ::std::conditional_t<T == ValueType::ARRAY_INT, ::std::vector<int>,
                ::std::conditional_t<T == ValueType::ARRAY_UINT, ::std::vector<unsigned int>,
                ::std::conditional_t<T == ValueType::ARRAY_STRING, ::std::vector<::std::string>,
                ::std::vector<fs::path>>>>>>>>>>;

            type defaultValue;
            ::std::string description;
            ::std::function<void(type)> addFunc;
        };

        using ValueVariant = ::std::variant<
            Value<ValueType::BOOL>,
            Value<ValueType::INT>,
            Value<ValueType::UINT>,
            Value<ValueType::STRING>,
            Value<ValueType::PATH>,
            Value<ValueType::ARRAY_BOOL>,
            Value<ValueType::ARRAY_INT>,
            Value<ValueType::ARRAY_UINT>,
            Value<ValueType::ARRAY_STRING>,
            Value<ValueType::ARRAY_PATH>
        >;

        struct Category {
            ::std::string name;
            ::std::vector<ValueVariant> values;
        };

        const ::std::vector<Category> _Settings = {
            Category{ "Draw",
                {
                    Value<ValueType::BOOL>{
                        "render_all_at_once",
                        true,
                        "Indicates that the image will be drawn in one block.",
                        [&](bool x) { _render_all_at_once = x; }
                    },
                    Value<ValueType::UINT>{
                        "blocks_per_draw",
                        200,
                        "If `render_all_at_once` is false then in one draw will be used that many blocks.",
                        [&](unsigned int x) { _blocks_per_draw = x; }
                    },
                    Value<ValueType::PATH>{
                        "world_file_path",
                        "./world.yaml",
                        "The path to file containing info about world. This can be relative to the exe file or absolute.",
                        [&](fs::path x) { _world_file_path = x.lexically_normal(); }
                    }
                }
            },
            Category{ "Image",
                {
                    Value<ValueType::UINT>{
                        "image_width",
                        720,
                        "Size of output image.",
                        [&](unsigned int x) { _image_width = x; }
                    },
                    Value<ValueType::UINT>{
                        "image_height",
                        720,
                        "",
                        [&](unsigned int x) { _image_height = x; }
                    },
                    Value<ValueType::STRING>{
                        "file_name",
                        "file-%H-%M-%S",
                        "Name of the output file. You can include time tags compatible with `strftime`.",
                        [&](::std::string x) {
                            x.erase(::std::remove_if(x.begin(), x.end(), ::isspace), x.end());
                            auto now = ::std::chrono::system_clock::now();
                            ::std::time_t t = ::std::chrono::system_clock::to_time_t(now);

                            ::std::ostringstream oss;
                            if (!x.empty()) {
                                oss << ::std::put_time(::std::localtime(&t), x.c_str());
                            }
                            else {
                                oss << ::std::put_time(::std::localtime(&t), "file-%H-%M-%S");
                            }
                            _file_name = oss.str();
                        }
                    },
                    Value<ValueType::PATH>{
                        "output_path",
                        ".",
                        "The path to save the output file. This can be relative to the exe file or absolute.",
                        [&](fs::path x) { _output_path = x; }
                    }
                }
            },
            Category{ "Quality",
                {
                    Value<ValueType::UINT>{
                        "aa_iter",
                        1,
                        "Number of AntyAliasing Iterations.",
                        [&](unsigned int x) { _aa_iter = x; }
                    },
                    Value<ValueType::UINT>{
                        "ref_iter",
                        4,
                        "Number of iterations used when calculating refraction i reflection.",
                        [&](unsigned int x) { _ref_iter = x; }
                    },
                    Value<ValueType::UINT>{
                        "gl_iter",
                        1,
                        "Number of Global Illumination iterations.",
                        [&](unsigned int x) { _gl_iter = x; }
                    },
                    Value<ValueType::UINT>{
                        "ind_rays",
                        75,
                        "Number of indirect rays casted from diffuse hit point.",
                        [&](unsigned int x) { _ind_rays = x; }
                    },
                    Value<ValueType::UINT>{
                        "shadow_samples",
                        50,
                        "Number of shadow samples.",
                        [&](unsigned int x) { _shadow_samples = x; }
                    }
                }
            }
        };

        void _setDefaultValues(const fs::path& iniPath) {
            fs::path baseDir = iniPath.parent_path();

            for (const Category& category : _Settings) {
                for (const ValueVariant& value : category.values) {

                    ::std::visit([&](auto&& val) {
                        using T = ::std::decay_t<decltype(val.defaultValue)>;

                        if constexpr (::std::is_same_v<T, fs::path>) {
                            if (val.defaultValue == fs::path(".")) {
                                val.addFunc(baseDir);
                            }
                            else if (val.defaultValue.is_relative()) {
                                val.addFunc(baseDir / val.defaultValue);
                            }
                            else {
                                val.addFunc(val.defaultValue);
                            }
                        }
                        else if constexpr (::std::is_same_v<T, ::std::vector<fs::path>>) {
                            ::std::vector<fs::path> resolved;
                            resolved.reserve(val.defaultValue.size());

                            for (const fs::path& p : val.defaultValue) {
                                if (p == fs::path(".")) {
                                    resolved.push_back(baseDir);
                                }
                                else if (p.is_relative()) {
                                    resolved.push_back(baseDir / p);
                                }
                                else {
                                    resolved.push_back(p);
                                }
                            }

                            val.addFunc(resolved);
                        }
                        else {
                            val.addFunc(val.defaultValue);
                        }

                    }, value);
                }
            }
        }

        void _createDefault(const fs::path& iniPath) {
            ::std::ofstream file(iniPath);
            fs::path baseDir = iniPath.parent_path();

            bool first = true;
            for (const Category& category : _Settings) {
                if (!first) file << "\n";
                file << "[" << category.name << "]\n";
                first = false;

                for (const ValueVariant& value : category.values) {
                    
                    ::std::visit([&](auto&& val) {
                        if (!val.description.empty()) {
                            file << "# " << val.description << "\n";
                        }

                        file << val.name << "=";

                        using T = ::std::decay_t<decltype(val.defaultValue)>;

                        if constexpr (::std::is_same_v<T, bool>) {
                            file << (val.defaultValue ? "true" : "false");
                        }
                        else if constexpr (::std::is_same_v<T, fs::path>) {
                            fs::path p = val.defaultValue;

                            if (p == fs::path(".")) {
                                p = baseDir;
                            }
                            else if (p.is_relative()) {
                                p = baseDir / p;
                            }

                            file << p.string();
                        }
                        else if constexpr (::std::is_same_v<T, ::std::vector<bool>>) {
                            for (size_t i = 0; i < val.defaultValue.size(); ++i) {
                                file << (val.defaultValue[i] ? "true" : "false");
                                if (i + 1 < val.defaultValue.size()) file << ";";
                            }
                        }
                        else if constexpr (::std::is_same_v<T, ::std::vector<int>>          ||
                                           ::std::is_same_v<T, ::std::vector<unsigned int>> ||
                                           ::std::is_same_v<T, ::std::vector<::std::string>>) {
                            for (size_t i = 0; i < val.defaultValue.size(); ++i) {
                                file << val.defaultValue[i];
                                if (i + 1 < val.defaultValue.size()) file << ";";
                            }
                        }
                        else if constexpr (::std::is_same_v<T, ::std::vector<fs::path>>) {
                            for (size_t i = 0; i < val.defaultValue.size(); ++i) {
                                fs::path p = val.defaultValue[i];

                                if (p == fs::path(".")) {
                                    p = baseDir;
                                }
                                else if (p.is_relative()) {
                                    p = baseDir / p;
                                }

                                file << p.string();
                                if (i + 1 < val.defaultValue.size()) file << ";";
                            }
                        }
                        else {
                            file << val.defaultValue;
                        }

                    }, value);

                    file << "\n";
                }
            }
        }

        void _parseLine(const ::std::string& line, const ::std::string& exeDir) {
            if (line.empty() || line[0] == '#' || line[0] == '[')
                return;

            auto eqPos = line.find('=');
            if (eqPos == ::std::string::npos)
                return;

            ::std::string key = line.substr(0, eqPos);
            ::std::string value = line.substr(eqPos + 1);

            for (const Category& category : _Settings) {
                for (const ValueVariant& var : category.values) {

                    ::std::visit([&](auto&& val) {
                        if (val.name != key) return; // check next

                        using T = ::std::decay_t<decltype(val.defaultValue)>;

                        if constexpr (::std::is_same_v<T, bool>) {
                            bool parsed = value == "true";
                            val.addFunc(parsed);
                        }
                        else if constexpr (::std::is_same_v<T, int>) {
                            val.addFunc(::std::stoi(value));
                        }
                        else if constexpr (::std::is_same_v<T, unsigned int>) {
                            val.addFunc(static_cast<unsigned int>(::std::stoul(value)));
                        }
                        else if constexpr (::std::is_same_v<T, ::std::string>) {
                            val.addFunc(value);
                        }
                        else if constexpr (::std::is_same_v<T, fs::path>) {
                            fs::path p = value;
                            if (p == ".") {
                                p = exeDir;
                            }
                            else if (p.is_relative()) {
                                p = fs::path(exeDir) / p;
                            }
                            val.addFunc(p.lexically_normal());
                        }
                        else if constexpr (::std::is_same_v<T, ::std::vector<bool>>) {
                            ::std::vector<bool> parsed;
                            ::std::stringstream ss(value);
                            ::std::string token;
                            while (::std::getline(ss, token, ';')) {
                                parsed.push_back(token == "true");
                            }
                            val.addFunc(parsed);
                        }
                        else if constexpr (::std::is_same_v<T, ::std::vector<int>>) {
                            ::std::vector<int> parsed;
                            ::std::stringstream ss(value);
                            ::std::string token;
                            while (::std::getline(ss, token, ';')) {
                                parsed.push_back(::std::stoi(token));
                            }
                            val.addFunc(parsed);
                        }
                        else if constexpr (::std::is_same_v<T, ::std::vector<unsigned int>>) {
                            ::std::vector<unsigned int> parsed;
                            ::std::stringstream ss(value);
                            ::std::string token;
                            while (::std::getline(ss, token, ';')) {
                                parsed.push_back(static_cast<unsigned int>(::std::stoul(token)));
                            }
                            val.addFunc(parsed);
                        }
                        else if constexpr (::std::is_same_v<T, ::std::vector<::std::string>>) {
                            ::std::vector<::std::string> parsed;
                            ::std::stringstream ss(value);
                            ::std::string token;
                            while (::std::getline(ss, token, ';')) {
                                parsed.push_back(token);
                            }
                            val.addFunc(parsed);
                        }
                        else if constexpr (::std::is_same_v<T, ::std::vector<fs::path>>) {
                            ::std::vector<fs::path> parsed;
                            ::std::stringstream ss(value);
                            ::std::string token;
                            while (::std::getline(ss, token, ';')) {
                                fs::path p = token;
                                if (p == ".") {
                                    p = exeDir;
                                }
                                else if (p.is_relative()) {
                                    p = fs::path(exeDir) / p;
                                }
                                parsed.push_back(p.lexically_normal());
                            }
                            val.addFunc(parsed);
                        }
                    }, var);
                }
            }
        }

    public:
        // Load settings from settings.ini or creates default
        bool load(const ::std::string& exeDir) {
            fs::path settingsPath = fs::path(exeDir) / "settings.ini";

            _setDefaultValues(settingsPath);

            if (!fs::exists(settingsPath)) {
                _createDefault(settingsPath);
                return true;
            }

            ::std::ifstream file(settingsPath);
            if (!file.is_open())
                return false;

            ::std::string line;
            while (::std::getline(file, line)) {
                _parseLine(line, exeDir);
            }

            file.close();

            return true;
        }

        const bool& getRenderAllAtOnce() const { return _render_all_at_once; }
        const unsigned int& getBlocksPerDraw() const { return _blocks_per_draw; }
        const unsigned int& getImageWidth() const { return _image_width; }
        const unsigned int& getImageHeight() const { return _image_height; }
        const unsigned int& getAAIterations() const { return _aa_iter; }
        const unsigned int& getRefIterations() const { return _ref_iter; }
        const unsigned int& getGlobalIlluminationIterations() const { return _gl_iter; }
        const unsigned int& getIndirectRays() const { return _ind_rays; }
        const unsigned int& getShadowSamples() const { return _shadow_samples; }
        const ::std::string& getFileName() const { return _file_name; }
        const fs::path& getOutputPath() const { return _output_path; }
        const fs::path& getWorldFilePath() const { return _world_file_path; }
    };
}
