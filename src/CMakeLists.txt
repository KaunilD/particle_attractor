cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# OpenGL
find_package(OpenGL REQUIRED)
# GLEW
include_directories(${PROJECT_SOURCE_DIR}/external/glew-2.1.0/include)
link_directories(${PROJECT_SOURCE_DIR}/external/glew-2.1.0/lib/Release/x64/)
# GLM
include_directories(${PROJECT_SOURCE_DIR}/external/glm)
# GLFW
include_directories(${PROJECT_SOURCE_DIR}/external/glfw-3.3.2/include)
link_directories(${PROJECT_SOURCE_DIR}/external/glfw-3.3.2/lib/Release/x64/)


set(SRCS
	"main.cpp"
	"shaderprogram.cpp"
	"gameobject.cpp"
	"camera.cpp"
	"objreader.cpp"
	"algorithms.cpp"
	"renderer/renderer.cpp"
	"renderer/sunrenderer.cpp"
	"renderer/particlerenderer.cpp"

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
	
	${PROJECT_SOURCE_DIR}/include/renderer/renderer.hpp
	${PROJECT_SOURCE_DIR}/include/renderer/sunrenderer.hpp
	${PROJECT_SOURCE_DIR}/include/renderer/particlerenderer.hpp
	
	${PROJECT_SOURCE_DIR}/include/window.hpp
	
	${PROJECT_SOURCE_DIR}/include/events/events.hpp
	${PROJECT_SOURCE_DIR}/include/events/updatable.hpp
	${PROJECT_SOURCE_DIR}/include/events/mouseevents.hpp
	${PROJECT_SOURCE_DIR}/include/events/keyboardevents.hpp

)

if(WIN32)
	add_executable(particle_attractor WIN32
		${SRCS}
		${HEADERS}
	)
else()
	add_executable(particle_attractor
		${SRCS}
		${HEADERS}
	)
endif()

target_include_directories(
	particle_attractor 
		PUBLIC 
			${OPENGL_INCLUDE_DIR} 
			$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
			$<INSTALL_INTERFACE:include>

)

target_link_libraries( 
	particle_attractor 
		PUBLIC
		glew32
		glfw
		${OPENGL_gl_LIBRARY} 
)


set_target_properties(${PROJECT_NAME} PROPERTIES
	LINK_FLAGS /SUBSYSTEM:CONSOLE
)

