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
include external/glm/test/bug/CMakeFiles/test-bug_ms_vec_static.dir/depend.make

# Include the progress variables for this target.
include external/glm/test/bug/CMakeFiles/test-bug_ms_vec_static.dir/progress.make

# Include the compile flags for this target's objects.
include external/glm/test/bug/CMakeFiles/test-bug_ms_vec_static.dir/flags.make

external/glm/test/bug/CMakeFiles/test-bug_ms_vec_static.dir/bug_ms_vec_static.cpp.o: external/glm/test/bug/CMakeFiles/test-bug_ms_vec_static.dir/flags.make
external/glm/test/bug/CMakeFiles/test-bug_ms_vec_static.dir/bug_ms_vec_static.cpp.o: ../external/glm/test/bug/bug_ms_vec_static.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dhruv/development/git/particle_attractor/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object external/glm/test/bug/CMakeFiles/test-bug_ms_vec_static.dir/bug_ms_vec_static.cpp.o"
	cd /home/dhruv/development/git/particle_attractor/build/external/glm/test/bug && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test-bug_ms_vec_static.dir/bug_ms_vec_static.cpp.o -c /home/dhruv/development/git/particle_attractor/external/glm/test/bug/bug_ms_vec_static.cpp

external/glm/test/bug/CMakeFiles/test-bug_ms_vec_static.dir/bug_ms_vec_static.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test-bug_ms_vec_static.dir/bug_ms_vec_static.cpp.i"
	cd /home/dhruv/development/git/particle_attractor/build/external/glm/test/bug && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dhruv/development/git/particle_attractor/external/glm/test/bug/bug_ms_vec_static.cpp > CMakeFiles/test-bug_ms_vec_static.dir/bug_ms_vec_static.cpp.i

external/glm/test/bug/CMakeFiles/test-bug_ms_vec_static.dir/bug_ms_vec_static.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test-bug_ms_vec_static.dir/bug_ms_vec_static.cpp.s"
	cd /home/dhruv/development/git/particle_attractor/build/external/glm/test/bug && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dhruv/development/git/particle_attractor/external/glm/test/bug/bug_ms_vec_static.cpp -o CMakeFiles/test-bug_ms_vec_static.dir/bug_ms_vec_static.cpp.s

external/glm/test/bug/CMakeFiles/test-bug_ms_vec_static.dir/bug_ms_vec_static.cpp.o.requires:

.PHONY : external/glm/test/bug/CMakeFiles/test-bug_ms_vec_static.dir/bug_ms_vec_static.cpp.o.requires

external/glm/test/bug/CMakeFiles/test-bug_ms_vec_static.dir/bug_ms_vec_static.cpp.o.provides: external/glm/test/bug/CMakeFiles/test-bug_ms_vec_static.dir/bug_ms_vec_static.cpp.o.requires
	$(MAKE) -f external/glm/test/bug/CMakeFiles/test-bug_ms_vec_static.dir/build.make external/glm/test/bug/CMakeFiles/test-bug_ms_vec_static.dir/bug_ms_vec_static.cpp.o.provides.build
.PHONY : external/glm/test/bug/CMakeFiles/test-bug_ms_vec_static.dir/bug_ms_vec_static.cpp.o.provides

external/glm/test/bug/CMakeFiles/test-bug_ms_vec_static.dir/bug_ms_vec_static.cpp.o.provides.build: external/glm/test/bug/CMakeFiles/test-bug_ms_vec_static.dir/bug_ms_vec_static.cpp.o


# Object files for target test-bug_ms_vec_static
test__bug_ms_vec_static_OBJECTS = \
"CMakeFiles/test-bug_ms_vec_static.dir/bug_ms_vec_static.cpp.o"

# External object files for target test-bug_ms_vec_static
test__bug_ms_vec_static_EXTERNAL_OBJECTS =

external/glm/test/bug/test-bug_ms_vec_static: external/glm/test/bug/CMakeFiles/test-bug_ms_vec_static.dir/bug_ms_vec_static.cpp.o
external/glm/test/bug/test-bug_ms_vec_static: external/glm/test/bug/CMakeFiles/test-bug_ms_vec_static.dir/build.make
external/glm/test/bug/test-bug_ms_vec_static: external/glm/test/bug/CMakeFiles/test-bug_ms_vec_static.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dhruv/development/git/particle_attractor/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test-bug_ms_vec_static"
	cd /home/dhruv/development/git/particle_attractor/build/external/glm/test/bug && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test-bug_ms_vec_static.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
external/glm/test/bug/CMakeFiles/test-bug_ms_vec_static.dir/build: external/glm/test/bug/test-bug_ms_vec_static

.PHONY : external/glm/test/bug/CMakeFiles/test-bug_ms_vec_static.dir/build

external/glm/test/bug/CMakeFiles/test-bug_ms_vec_static.dir/requires: external/glm/test/bug/CMakeFiles/test-bug_ms_vec_static.dir/bug_ms_vec_static.cpp.o.requires

.PHONY : external/glm/test/bug/CMakeFiles/test-bug_ms_vec_static.dir/requires

external/glm/test/bug/CMakeFiles/test-bug_ms_vec_static.dir/clean:
	cd /home/dhruv/development/git/particle_attractor/build/external/glm/test/bug && $(CMAKE_COMMAND) -P CMakeFiles/test-bug_ms_vec_static.dir/cmake_clean.cmake
.PHONY : external/glm/test/bug/CMakeFiles/test-bug_ms_vec_static.dir/clean

external/glm/test/bug/CMakeFiles/test-bug_ms_vec_static.dir/depend:
	cd /home/dhruv/development/git/particle_attractor/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dhruv/development/git/particle_attractor /home/dhruv/development/git/particle_attractor/external/glm/test/bug /home/dhruv/development/git/particle_attractor/build /home/dhruv/development/git/particle_attractor/build/external/glm/test/bug /home/dhruv/development/git/particle_attractor/build/external/glm/test/bug/CMakeFiles/test-bug_ms_vec_static.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : external/glm/test/bug/CMakeFiles/test-bug_ms_vec_static.dir/depend
