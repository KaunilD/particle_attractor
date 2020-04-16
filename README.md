# Lucas Kanade based Optical Flow simulation using a particle engine.

#### Dependencies:

1. [CMake v3.8+](https://cmake.org/download/) [for CUDA support within CMake]
2. [CUDA v9.0+](https://developer.nvidia.com/cuda-92-download-archive) 
3. [OpenCV 4.3.0](https://github.com/opencv/opencv/archive/4.3.0.tar.gz)
4. [GLFW](https://github.com/glfw/glfw)
5. [GLEW](https://github.com/nigels-com/glew/archive/glew-2.1.0.tar.gz) (Only Windows)
6. [GLM](https://github.com/g-truc/glm/archive/0.9.9.8.tar.gz)

#### Installation

1. Let `$PROJECT_ROOT` be the root directory of the project.
2. Download an example video from [here](https://drive.google.com/open?id=17ydViQMNjSS5pO2UBRHf9ntapH9-HCjR) and place it in `$PROJECT_ROOT/src`.
3. After installing the dependencies please follow the cmake build steps in the code block below.
4. After the binary has been compiled, place it in `$PROJECT_ROOT/src` directory besides the video file you downloaded in step 2.

```cmake
cd $PROJECT_ROOT
mkdir build
cd build
cmake ..
make -j4
```

#### Roadmap:

- [x] CUDA based particle engine.
- [x] Simulate gravity using CUDA OpenGL Interop.
- [x] Integrating OpenCV with the particle engine to render frames captured from the video file.
- [ ] Compute Shader for creating Laplacian image pyramid.
- [ ] Compute Shader for LK based velocity vector computation.
- [ ] Update particle engine physics to simulate Hooke's Law.