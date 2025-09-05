/**************************************************************
 *                                                            *
 *  Project:   CudaRayTracer                                  *
 *  Authors:   Muppetsg2 & MAIPA01                            *
 *  License:   MIT License                                    *
 *  Last Update: 06.09.2025                                   *
 *                                                            *
 **************************************************************/

#pragma once

namespace fs = std::filesystem;

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
        fs::path _output_path;

        void _createDefault(const fs::path& iniPath) {
            std::ofstream file(iniPath);
            file << "\n[Draw]\n";
            file << "# Indicates that the image will be drawn in one block.\n";
            file << "render_all_at_once=true\n";
            file << "# If `render_all_at_once` is false then in one draw will be used that many blocks.\n";
            file << "blocks_per_draw=200\n";
            file << "[Image]\n";
            file << "# Size of output image.\n";
            file << "image_width=720\n";
            file << "image_height=720\n";
            file << "# The path to save the output file. This can be relative to the exe file or absolute.\n";
            file << "output_path=" << iniPath.parent_path().string() << "\n";
            file << "[Quality]\n";
            file << "# Number of AntyAliasing Iterations.\n";
            file << "aa_iter=1\n";
            file << "# Number of iterations used when calculating refraction i reflection.\n";
            file << "ref_iter=4\n";
            file << "# Number of Global Illumination iterations.\n";
            file << "gl_iter=0\n";
            file << "# Number of indirect rays casted from diffuse hit point..\n";
            file << "ind_rays=75\n";
            file << "# Number of shadow samples.\n";
            file << "shadow_samples=50";
        }

        void _parseLine(const std::string& line, const std::string& exeDir) {
            if (line.empty() || line[0] == '#' || line[0] == '[')
                return;

            auto eqPos = line.find('=');
            if (eqPos == std::string::npos)
                return;

            std::string key = line.substr(0, eqPos);
            std::string value = line.substr(eqPos + 1);

            /*
            if (key == "serialization_base_classes") {
                std::stringstream ss(value);
                std::string cls;
                while (std::getline(ss, cls, ';')) {
                    cls.erase(std::remove_if(cls.begin(), cls.end(), ::isspace), cls.end());
                    if (!cls.empty())
                        _serializationBaseClasses.emplace(cls);
                }
            }
            else if (key == "parser_include_dirs") {
                std::stringstream ss(value);
                std::string cls;
                while (std::getline(ss, cls, ';')) {
                    fs::path p = fs::path(cls);
                    if (p.is_relative())
                        p = fs::path(exeDir) / p;

                    std::string s = p.string();
                    if (!check_directory(s.c_str()))
                        _parserIncludeDirs.emplace(p);
                }
            }
            else if (key == "output_path") {
                fs::path p = fs::path(value);
                if (p.is_relative())
                    p = fs::path(exeDir) / p;

                std::string s = p.string();
                if (check_directory(s.c_str()))
                    _outputPath = p;
                else
                    _outputPath = exeDir;
            }
            else if (key == "output_class_name") {
                value.erase(std::remove_if(value.begin(), value.end(), ::isspace), value.end());
                if (!value.empty())
                    _outputClassName = value;
                else
                    _outputClassName = "Serializer";
            }
            */
        }

    public:
        // Load settings from settings.ini or creates default
        bool load(const std::string& exeDir) {
            namespace fs = std::filesystem;
            fs::path settingsPath = fs::path(exeDir) / "settings.ini";

            _render_all_at_once = true;
            _blocks_per_draw = 200;
            _image_width = 720;
            _image_height = 720;
            _aa_iter = 1;
            _ref_iter = 4;
            _gl_iter = 0;
            _ind_rays = 75;
            _shadow_samples = 50;
            _output_path = fs::path(exeDir);

            if (!fs::exists(settingsPath)) {
                _createDefault(settingsPath);
                return true;
            }

            std::ifstream file(settingsPath);
            if (!file.is_open())
                return false;

            std::string line;
            while (std::getline(file, line)) {
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
        const fs::path& getOutputPath() const { return _output_path; }
    };
}