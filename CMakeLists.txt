cmake_minimum_required (VERSION 3.10)

project ("particle_attractor" 
	VERSION 1.1 
	DESCRIPTION "My experiment with particle physics in Qt5 and OpenGL 4.4" 
	LANGUAGES CXX
)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
endif()

# build the following
add_subdirectory ("src")