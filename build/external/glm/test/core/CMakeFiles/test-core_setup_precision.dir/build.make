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
include external/glm/test/core/CMakeFiles/test-core_setup_precision.dir/depend.make

# Include the progress variables for this target.
include external/glm/test/core/CMakeFiles/test-core_setup_precision.dir/progress.make

# Include the compile flags for this target's objects.
include external/glm/test/core/CMakeFiles/test-core_setup_precision.dir/flags.make

external/glm/test/core/CMakeFiles/test-core_setup_precision.dir/core_setup_precision.cpp.o: external/glm/test/core/CMakeFiles/test-core_setup_precision.dir/flags.make
external/glm/test/core/CMakeFiles/test-core_setup_precision.dir/core_setup_precision.cpp.o: ../external/glm/test/core/core_setup_precision.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dhruv/development/git/particle_attractor/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object external/glm/test/core/CMakeFiles/test-core_setup_precision.dir/core_setup_precision.cpp.o"
	cd /home/dhruv/development/git/particle_attractor/build/external/glm/test/core && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test-core_setup_precision.dir/core_setup_precision.cpp.o -c /home/dhruv/development/git/particle_attractor/external/glm/test/core/core_setup_precision.cpp

external/glm/test/core/CMakeFiles/test-core_setup_precision.dir/core_setup_precision.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test-core_setup_precision.dir/core_setup_precision.cpp.i"
	cd /home/dhruv/development/git/particle_attractor/build/external/glm/test/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dhruv/development/git/particle_attractor/external/glm/test/core/core_setup_precision.cpp > CMakeFiles/test-core_setup_precision.dir/core_setup_precision.cpp.i

external/glm/test/core/CMakeFiles/test-core_setup_precision.dir/core_setup_precision.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test-core_setup_precision.dir/core_setup_precision.cpp.s"
	cd /home/dhruv/development/git/particle_attractor/build/external/glm/test/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dhruv/development/git/particle_attractor/external/glm/test/core/core_setup_precision.cpp -o CMakeFiles/test-core_setup_precision.dir/core_setup_precision.cpp.s

external/glm/test/core/CMakeFiles/test-core_setup_precision.dir/core_setup_precision.cpp.o.requires:

.PHONY : external/glm/test/core/CMakeFiles/test-core_setup_precision.dir/core_setup_precision.cpp.o.requires

external/glm/test/core/CMakeFiles/test-core_setup_precision.dir/core_setup_precision.cpp.o.provides: external/glm/test/core/CMakeFiles/test-core_setup_precision.dir/core_setup_precision.cpp.o.requires
	$(MAKE) -f external/glm/test/core/CMakeFiles/test-core_setup_precision.dir/build.make external/glm/test/core/CMakeFiles/test-core_setup_precision.dir/core_setup_precision.cpp.o.provides.build
.PHONY : external/glm/test/core/CMakeFiles/test-core_setup_precision.dir/core_setup_precision.cpp.o.provides

external/glm/test/core/CMakeFiles/test-core_setup_precision.dir/core_setup_precision.cpp.o.provides.build: external/glm/test/core/CMakeFiles/test-core_setup_precision.dir/core_setup_precision.cpp.o


# Object files for target test-core_setup_precision
test__core_setup_precision_OBJECTS = \
"CMakeFiles/test-core_setup_precision.dir/core_setup_precision.cpp.o"

# External object files for target test-core_setup_precision
test__core_setup_precision_EXTERNAL_OBJECTS =

external/glm/test/core/test-core_setup_precision: external/glm/test/core/CMakeFiles/test-core_setup_precision.dir/core_setup_precision.cpp.o
external/glm/test/core/test-core_setup_precision: external/glm/test/core/CMakeFiles/test-core_setup_precision.dir/build.make
external/glm/test/core/test-core_setup_precision: external/glm/test/core/CMakeFiles/test-core_setup_precision.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dhruv/development/git/particle_attractor/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test-core_setup_precision"
	cd /home/dhruv/development/git/particle_attractor/build/external/glm/test/core && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test-core_setup_precision.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
external/glm/test/core/CMakeFiles/test-core_setup_precision.dir/build: external/glm/test/core/test-core_setup_precision

.PHONY : external/glm/test/core/CMakeFiles/test-core_setup_precision.dir/build

external/glm/test/core/CMakeFiles/test-core_setup_precision.dir/requires: external/glm/test/core/CMakeFiles/test-core_setup_precision.dir/core_setup_precision.cpp.o.requires

.PHONY : external/glm/test/core/CMakeFiles/test-core_setup_precision.dir/requires

external/glm/test/core/CMakeFiles/test-core_setup_precision.dir/clean:
	cd /home/dhruv/development/git/particle_attractor/build/external/glm/test/core && $(CMAKE_COMMAND) -P CMakeFiles/test-core_setup_precision.dir/cmake_clean.cmake
.PHONY : external/glm/test/core/CMakeFiles/test-core_setup_precision.dir/clean

external/glm/test/core/CMakeFiles/test-core_setup_precision.dir/depend:
	cd /home/dhruv/development/git/particle_attractor/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dhruv/development/git/particle_attractor /home/dhruv/development/git/particle_attractor/external/glm/test/core /home/dhruv/development/git/particle_attractor/build /home/dhruv/development/git/particle_attractor/build/external/glm/test/core /home/dhruv/development/git/particle_attractor/build/external/glm/test/core/CMakeFiles/test-core_setup_precision.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : external/glm/test/core/CMakeFiles/test-core_setup_precision.dir/depend

