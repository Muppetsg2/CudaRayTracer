# CudaRayTracer

**CudaRayTracer** is a project that ports a custom Ray Tracer implementation, originally developed during university studies, to run on the GPU using **CUDA**.  
The main goal was to explore GPU-based rendering techniques and gain hands-on experience with CUDA programming.

## ✨ Features

- **Adaptive Antialiasing**
  - Pixels are split into 4 parts (*boxes*).
  - Each box can be subdivided recursively (up to 4 levels – CUDA limitation).
  - If all 4 corners of a box have the same color, subdivision stops (*early termination*).

- **Global Illumination**
  - Realistic lighting with reflections from other objects.

- **Area Light with LTC (Linearly Transformed Cosines)**
  - Accurate and efficient surface light representation.

- **Colored Materials**
  - **Diffuse** – scatters light.
  - **Reflective** – reflects light (mirror-like).
  - **Refractive** – bends light (e.g., glass).
  - Each material can have its own color.

- **Scene**
  - Default scene is a **Cornell Box** with two spheres:
    - One **refractive**.
    - One **reflective**.
  - Scene is defined **directly in the source code**.

- **File Output**
  - Rendered image is saved in `.hdr` format as `file.hdr`.

## 🧰 Libraries Used

- [**SFML**](https://www.sfml-dev.org/) – window and display handling.
- [**stb_image_write**](https://github.com/nothings/stb) – saving images to files.
- [**stb_image**](https://github.com/nothings/stb) – loading textures (LTC).
- [**mstd**](https://github.com/MAIPA01/mstd) – math library (modified for CUDA compatibility).

📦 **Installation via vcpkg**

All external libraries are managed using [**vcpkg**](https://github.com/microsoft/vcpkg):

```bash
vcpkg install sfml stb
```

## ⚙️ Requirements

- **Windows 10/11**
- **Visual Studio 2022**
- **CUDA Toolkit** (recommended 12.9)
- **NVIDIA GPU with CUDA support**
- **C++20** (MSVC compiler)

## 🛠️ Build and Run

1. Open `CudaRayTracer.sln` in **Visual Studio 2022**.
2. Set configuration to `Release` or `Debug`.
3. Ensure the project uses CUDA Runtime.
4. Run (`Ctrl+F5`).

After rendering, the image will first be displayed in a window, and upon closing it, saved as `file.hdr`.

## ⚙️ Rendering Settings

- **renderAllAtOnce** *(bool)* – if `true`, rendering is done in one kernel and the entire image is displayed at once.  
  If `false`, the image is rendered in multiple kernels, each responsible for a specific part, with results shown after each.
- **blocksPerDraw** *(int)* – relevant only when `renderAllAtOnce = false`; defines how many blocks one kernel processes.
- **nx**, **ny** *(int)* – image resolution.
- **tx**, **ty** *(int)* – block size. For CUDA devices supporting 1024 threads per block, the largest square block is 19×19.
- **aa_iter** *(int)* – number of antialiasing iterations (max 4).
- **ref_iter** *(int)* – number of ray iterations for refraction and reflection.
- **gl_iter** *(int)* – number of iterations for global illumination calculations.
- **ind_rays** *(int)* – number of rays per hemisphere for global illumination (more rays = less noise).
- **shadowSamples** *(int)* – number of rays used for shadow calculation.

## 🖼️ Render Preview

Example output (Cornell Box):  

![](./renders/render.png)  

The `.hdr` file can be opened in HDR viewers or converted to `.png` and other formats.

## 📂 Project Structure

```
.
├── renders/            # Generated images
├── CudaRayTracer.sln   # Visual Studio 2022 solution
├── LICENSE             # License
└── README.md
```

## 📌 Notes

- Scene is defined in code (no external file loading).
- Adaptive antialiasing limited to 4 levels (CUDA).
- Performance depends on the GPU used.
- Rendering settings are modified in the source code.

## 👥 Authors

Project created by two students while learning and exploring ray tracing techniques with CUDA:

- [**Muppetsg2**](https://github.com/Muppetsg2)
- [**MAIPA01**](https://github.com/MAIPA01)

## 📜 License

📝 This project is licensed under the MIT License.  

- ✅ Free to use, modify, and distribute.  
- ✅ Suitable for commercial and non-commercial use.  
- ❗ Must include the original license and copyright.  

See the [LICENSE](./LICENSE) file for details.