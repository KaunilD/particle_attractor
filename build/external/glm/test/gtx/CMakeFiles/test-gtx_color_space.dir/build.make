# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/dhruv/development/git/particle_attractor

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/dhruv/development/git/particle_attractor/build

# Include any dependencies generated for this target.
include external/glm/test/gtx/CMakeFiles/test-gtx_color_space.dir/depend.make

# Include the progress variables for this target.
include external/glm/test/gtx/CMakeFiles/test-gtx_color_space.dir/progress.make

# Include the compile flags for this target's objects.
include external/glm/test/gtx/CMakeFiles/test-gtx_color_space.dir/flags.make

external/glm/test/gtx/CMakeFiles/test-gtx_color_space.dir/gtx_color_space.cpp.o: external/glm/test/gtx/CMakeFiles/test-gtx_color_space.dir/flags.make
external/glm/test/gtx/CMakeFiles/test-gtx_color_space.dir/gtx_color_space.cpp.o: ../external/glm/test/gtx/gtx_color_space.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dhruv/development/git/particle_attractor/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object external/glm/test/gtx/CMakeFiles/test-gtx_color_space.dir/gtx_color_space.cpp.o"
	cd /home/dhruv/development/git/particle_attractor/build/external/glm/test/gtx && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test-gtx_color_space.dir/gtx_color_space.cpp.o -c /home/dhruv/development/git/particle_attractor/external/glm/test/gtx/gtx_color_space.cpp

external/glm/test/gtx/CMakeFiles/test-gtx_color_space.dir/gtx_color_space.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test-gtx_color_space.dir/gtx_color_space.cpp.i"
	cd /home/dhruv/development/git/particle_attractor/build/external/glm/test/gtx && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dhruv/development/git/particle_attractor/external/glm/test/gtx/gtx_color_space.cpp > CMakeFiles/test-gtx_color_space.dir/gtx_color_space.cpp.i

external/glm/test/gtx/CMakeFiles/test-gtx_color_space.dir/gtx_color_space.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test-gtx_color_space.dir/gtx_color_space.cpp.s"
	cd /home/dhruv/development/git/particle_attractor/build/external/glm/test/gtx && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dhruv/development/git/particle_attractor/external/glm/test/gtx/gtx_color_space.cpp -o CMakeFiles/test-gtx_color_space.dir/gtx_color_space.cpp.s

external/glm/test/gtx/CMakeFiles/test-gtx_color_space.dir/gtx_color_space.cpp.o.requires:

.PHONY : external/glm/test/gtx/CMakeFiles/test-gtx_color_space.dir/gtx_color_space.cpp.o.requires

external/glm/test/gtx/CMakeFiles/test-gtx_color_space.dir/gtx_color_space.cpp.o.provides: external/glm/test/gtx/CMakeFiles/test-gtx_color_space.dir/gtx_color_space.cpp.o.requires
	$(MAKE) -f external/glm/test/gtx/CMakeFiles/test-gtx_color_space.dir/build.make external/glm/test/gtx/CMakeFiles/test-gtx_color_space.dir/gtx_color_space.cpp.o.provides.build
.PHONY : external/glm/test/gtx/CMakeFiles/test-gtx_color_space.dir/gtx_color_space.cpp.o.provides

external/glm/test/gtx/CMakeFiles/test-gtx_color_space.dir/gtx_color_space.cpp.o.provides.build: external/glm/test/gtx/CMakeFiles/test-gtx_color_space.dir/gtx_color_space.cpp.o


# Object files for target test-gtx_color_space
test__gtx_color_space_OBJECTS = \
"CMakeFiles/test-gtx_color_space.dir/gtx_color_space.cpp.o"

# External object files for target test-gtx_color_space
test__gtx_color_space_EXTERNAL_OBJECTS =

external/glm/test/gtx/test-gtx_color_space: external/glm/test/gtx/CMakeFiles/test-gtx_color_space.dir/gtx_color_space.cpp.o
external/glm/test/gtx/test-gtx_color_space: external/glm/test/gtx/CMakeFiles/test-gtx_color_space.dir/build.make
external/glm/test/gtx/test-gtx_color_space: external/glm/test/gtx/CMakeFiles/test-gtx_color_space.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dhruv/development/git/particle_attractor/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test-gtx_color_space"
	cd /home/dhruv/development/git/particle_attractor/build/external/glm/test/gtx && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test-gtx_color_space.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
external/glm/test/gtx/CMakeFiles/test-gtx_color_space.dir/build: external/glm/test/gtx/test-gtx_color_space

.PHONY : external/glm/test/gtx/CMakeFiles/test-gtx_color_space.dir/build

external/glm/test/gtx/CMakeFiles/test-gtx_color_space.dir/requires: external/glm/test/gtx/CMakeFiles/test-gtx_color_space.dir/gtx_color_space.cpp.o.requires

.PHONY : external/glm/test/gtx/CMakeFiles/test-gtx_color_space.dir/requires

external/glm/test/gtx/CMakeFiles/test-gtx_color_space.dir/clean:
	cd /home/dhruv/development/git/particle_attractor/build/external/glm/test/gtx && $(CMAKE_COMMAND) -P CMakeFiles/test-gtx_color_space.dir/cmake_clean.cmake
.PHONY : external/glm/test/gtx/CMakeFiles/test-gtx_color_space.dir/clean

external/glm/test/gtx/CMakeFiles/test-gtx_color_space.dir/depend:
	cd /home/dhruv/development/git/particle_attractor/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dhruv/development/git/particle_attractor /home/dhruv/development/git/particle_attractor/external/glm/test/gtx /home/dhruv/development/git/particle_attractor/build /home/dhruv/development/git/particle_attractor/build/external/glm/test/gtx /home/dhruv/development/git/particle_attractor/build/external/glm/test/gtx/CMakeFiles/test-gtx_color_space.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : external/glm/test/gtx/CMakeFiles/test-gtx_color_space.dir/depend

