#ifndef LIBS_H
#define LIBS_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <exception>
#include <random>
#include <cstdint>
#include <algorithm>
#include <iomanip>
#include <random>
#include <math.h>
#include <list>
#include <chrono>
// CUDA
#include <cuda.h>
#include <cuda_runtime.h>
// GL
#ifdef _WIN32
#include "GL/glew.h"
#endif // _WIN32
// GLFW
#include <GLFW/glfw3.h>
// OpenMP
#include <omp.h>
// GLM
#include <glm/glm.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
// OpenCV
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#define LOG(x) {std::cout << x << std::endl;}

constexpr auto VIEWPORT_WIDTH = 800;
constexpr auto VIEWPORT_HEIGHT = 800;

using std::shared_ptr;
using std::make_shared;

using std::unique_ptr;
using std::make_unique;

#endif