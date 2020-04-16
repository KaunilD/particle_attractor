cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# CUDA
find_package( CUDA 9.0 REQUIRED )
if(CUDA_FOUND)
	message(STATUS "CUDA library status:")
	message(STATUS "    NVCC_EXECUTABLE: ${CUDA_NVCC_EXECUTABLE}")
	message(STATUS "    INCLUDE_PATH: ${CUDA_INCLUDE_DIRS}")
	message(STATUS "    TOOLKIT_DIR: ${CUDA_TOOLKIT_ROOT_DIR}")
	set(CUDA_NVCC_FLAGS "-gencode;arch=compute_61,code=sm_61;")
	set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")
else(CUDA_FOUND)
	message("CUDA is not installed on this system.")
endif()
include_directories( ${CUDA_INCLUDE_DIRS})


# OpenGL
find_package(OpenGL REQUIRED)

# GLEW
if(WIN32)
include_directories(${PROJECT_SOURCE_DIR}/external/glew-2.1.0/include)
link_directories(${PROJECT_SOURCE_DIR}/external/glew-2.1.0/lib/Release/x64/)
endif()

# GLM
include_directories(${PROJECT_SOURCE_DIR}/external/glm)

# GLFW
include_directories(${PROJECT_SOURCE_DIR}/external/glfw-3.3.2/include)
link_directories(${PROJECT_SOURCE_DIR}/external/glfw-3.3.2/lib/Release/x64/)

# OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    VERSION: ${OpenCV_VERSION}")
message(STATUS "    LIBS: ${OpenCV_LIBS}")
message(STATUS "    INCLUDE_PATH: ${OpenCV_INCLUDE_DIRS}")

if(CMAKE_VERSION VERSION_LESS "2.8.11")
	# Add OpenCV headers location to include paths
	include_directories(${OpenCV_INCLUDE_DIRS})
endif()

set(SRCS
	"main.cpp"
	"shaderprogram.cpp"
	"camera.cpp"
	"objreader.cpp"
	"algorithms.cpp"

	"renderer/renderer.cpp"
	"renderer/sunrenderer.cpp"
	"renderer/particlerenderer.cpp"

	"gameobject/gameobject.cpp"
	"gameobject/mesh.cpp"
	"gameobject/material.cpp"

	"window.cpp"
)

set(HEADERS
	${PROJECT_SOURCE_DIR}/include/libs.hpp
	
	${PROJECT_SOURCE_DIR}/include/vertex.hpp
	${PROJECT_SOURCE_DIR}/include/objreader.hpp

	${PROJECT_SOURCE_DIR}/include/shaderprogram.hpp
	
	${PROJECT_SOURCE_DIR}/include/gameobject/gameobject.hpp
	${PROJECT_SOURCE_DIR}/include/gameobject/sunobject.hpp
	${PROJECT_SOURCE_DIR}/include/gameobject/particleobject.hpp
	${PROJECT_SOURCE_DIR}/include/gameobject/mesh.hpp
	${PROJECT_SOURCE_DIR}/include/gameobject/material.hpp
	
	${PROJECT_SOURCE_DIR}/include/algorithms/algorithms.hpp
	
	${PROJECT_SOURCE_DIR}/include/camera.hpp

	${PROJECT_SOURCE_DIR}/include/scene.hpp
	
	${PROJECT_SOURCE_DIR}/include/renderer/renderer.hpp
	${PROJECT_SOURCE_DIR}/include/renderer/sunrenderer.hpp
	${PROJECT_SOURCE_DIR}/include/renderer/particlerenderer.hpp
	
	${PROJECT_SOURCE_DIR}/include/window.hpp
	
	${PROJECT_SOURCE_DIR}/include/events/events.hpp
	${PROJECT_SOURCE_DIR}/include/events/updatable.hpp
	${PROJECT_SOURCE_DIR}/include/events/mouseevents.hpp
	${PROJECT_SOURCE_DIR}/include/events/windowevents.hpp
	${PROJECT_SOURCE_DIR}/include/events/keyboardevents.hpp
	
	${PROJECT_SOURCE_DIR}/include/cuda/update_gameobject.hpp
	${PROJECT_SOURCE_DIR}/include/cuda/update_models.hpp


)

set (KERNELS
	"cuda/update_gameobject.cu"
	"cuda/update_models.cu"
)

if(WIN32)
	add_executable(particle_attractor WIN32
		${KERNELS}
		${SRCS}
		${HEADERS}
	)
else()
	add_executable(particle_attractor
		${KERNELS}
		${SRCS}
		${HEADERS}
	)
endif()

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

set_target_properties( ${PROJECT_NAME}
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                      )
target_include_directories(
	particle_attractor 
		PUBLIC 
			${OPENGL_INCLUDE_DIR} 
			$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
			$<INSTALL_INTERFACE:include>

)
if(WIN32)
target_link_libraries( 
	particle_attractor 
		PUBLIC
		glew32
		glfw
		${OPENGL_gl_LIBRARY} 
		${CUDA_LIBRARIES}
		${OpenCV_LIBS}
)
else()
target_link_libraries( 
	particle_attractor 
		PUBLIC
		glew32
		glfw
		${OPENGL_gl_LIBRARY} 
		${CUDA_LIBRARIES}
)
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
	LINK_FLAGS /SUBSYSTEM:CONSOLE
)

