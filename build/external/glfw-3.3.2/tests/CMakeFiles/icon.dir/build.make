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
include external/glfw-3.3.2/tests/CMakeFiles/icon.dir/depend.make

# Include the progress variables for this target.
include external/glfw-3.3.2/tests/CMakeFiles/icon.dir/progress.make

# Include the compile flags for this target's objects.
include external/glfw-3.3.2/tests/CMakeFiles/icon.dir/flags.make

external/glfw-3.3.2/tests/CMakeFiles/icon.dir/icon.c.o: external/glfw-3.3.2/tests/CMakeFiles/icon.dir/flags.make
external/glfw-3.3.2/tests/CMakeFiles/icon.dir/icon.c.o: ../external/glfw-3.3.2/tests/icon.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dhruv/development/git/particle_attractor/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object external/glfw-3.3.2/tests/CMakeFiles/icon.dir/icon.c.o"
	cd /home/dhruv/development/git/particle_attractor/build/external/glfw-3.3.2/tests && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/icon.dir/icon.c.o   -c /home/dhruv/development/git/particle_attractor/external/glfw-3.3.2/tests/icon.c

external/glfw-3.3.2/tests/CMakeFiles/icon.dir/icon.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/icon.dir/icon.c.i"
	cd /home/dhruv/development/git/particle_attractor/build/external/glfw-3.3.2/tests && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/dhruv/development/git/particle_attractor/external/glfw-3.3.2/tests/icon.c > CMakeFiles/icon.dir/icon.c.i

external/glfw-3.3.2/tests/CMakeFiles/icon.dir/icon.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/icon.dir/icon.c.s"
	cd /home/dhruv/development/git/particle_attractor/build/external/glfw-3.3.2/tests && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/dhruv/development/git/particle_attractor/external/glfw-3.3.2/tests/icon.c -o CMakeFiles/icon.dir/icon.c.s

external/glfw-3.3.2/tests/CMakeFiles/icon.dir/icon.c.o.requires:

.PHONY : external/glfw-3.3.2/tests/CMakeFiles/icon.dir/icon.c.o.requires

external/glfw-3.3.2/tests/CMakeFiles/icon.dir/icon.c.o.provides: external/glfw-3.3.2/tests/CMakeFiles/icon.dir/icon.c.o.requires
	$(MAKE) -f external/glfw-3.3.2/tests/CMakeFiles/icon.dir/build.make external/glfw-3.3.2/tests/CMakeFiles/icon.dir/icon.c.o.provides.build
.PHONY : external/glfw-3.3.2/tests/CMakeFiles/icon.dir/icon.c.o.provides

external/glfw-3.3.2/tests/CMakeFiles/icon.dir/icon.c.o.provides.build: external/glfw-3.3.2/tests/CMakeFiles/icon.dir/icon.c.o


external/glfw-3.3.2/tests/CMakeFiles/icon.dir/__/deps/glad_gl.c.o: external/glfw-3.3.2/tests/CMakeFiles/icon.dir/flags.make
external/glfw-3.3.2/tests/CMakeFiles/icon.dir/__/deps/glad_gl.c.o: ../external/glfw-3.3.2/deps/glad_gl.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dhruv/development/git/particle_attractor/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object external/glfw-3.3.2/tests/CMakeFiles/icon.dir/__/deps/glad_gl.c.o"
	cd /home/dhruv/development/git/particle_attractor/build/external/glfw-3.3.2/tests && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/icon.dir/__/deps/glad_gl.c.o   -c /home/dhruv/development/git/particle_attractor/external/glfw-3.3.2/deps/glad_gl.c

external/glfw-3.3.2/tests/CMakeFiles/icon.dir/__/deps/glad_gl.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/icon.dir/__/deps/glad_gl.c.i"
	cd /home/dhruv/development/git/particle_attractor/build/external/glfw-3.3.2/tests && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/dhruv/development/git/particle_attractor/external/glfw-3.3.2/deps/glad_gl.c > CMakeFiles/icon.dir/__/deps/glad_gl.c.i

external/glfw-3.3.2/tests/CMakeFiles/icon.dir/__/deps/glad_gl.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/icon.dir/__/deps/glad_gl.c.s"
	cd /home/dhruv/development/git/particle_attractor/build/external/glfw-3.3.2/tests && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/dhruv/development/git/particle_attractor/external/glfw-3.3.2/deps/glad_gl.c -o CMakeFiles/icon.dir/__/deps/glad_gl.c.s

external/glfw-3.3.2/tests/CMakeFiles/icon.dir/__/deps/glad_gl.c.o.requires:

.PHONY : external/glfw-3.3.2/tests/CMakeFiles/icon.dir/__/deps/glad_gl.c.o.requires

external/glfw-3.3.2/tests/CMakeFiles/icon.dir/__/deps/glad_gl.c.o.provides: external/glfw-3.3.2/tests/CMakeFiles/icon.dir/__/deps/glad_gl.c.o.requires
	$(MAKE) -f external/glfw-3.3.2/tests/CMakeFiles/icon.dir/build.make external/glfw-3.3.2/tests/CMakeFiles/icon.dir/__/deps/glad_gl.c.o.provides.build
.PHONY : external/glfw-3.3.2/tests/CMakeFiles/icon.dir/__/deps/glad_gl.c.o.provides

external/glfw-3.3.2/tests/CMakeFiles/icon.dir/__/deps/glad_gl.c.o.provides.build: external/glfw-3.3.2/tests/CMakeFiles/icon.dir/__/deps/glad_gl.c.o


# Object files for target icon
icon_OBJECTS = \
"CMakeFiles/icon.dir/icon.c.o" \
"CMakeFiles/icon.dir/__/deps/glad_gl.c.o"

# External object files for target icon
icon_EXTERNAL_OBJECTS =

external/glfw-3.3.2/tests/icon: external/glfw-3.3.2/tests/CMakeFiles/icon.dir/icon.c.o
external/glfw-3.3.2/tests/icon: external/glfw-3.3.2/tests/CMakeFiles/icon.dir/__/deps/glad_gl.c.o
external/glfw-3.3.2/tests/icon: external/glfw-3.3.2/tests/CMakeFiles/icon.dir/build.make
external/glfw-3.3.2/tests/icon: external/glfw-3.3.2/src/libglfw3.a
external/glfw-3.3.2/tests/icon: /usr/lib/x86_64-linux-gnu/libm.so
external/glfw-3.3.2/tests/icon: /usr/lib/x86_64-linux-gnu/librt.so
external/glfw-3.3.2/tests/icon: /usr/lib/x86_64-linux-gnu/libm.so
external/glfw-3.3.2/tests/icon: /usr/lib/x86_64-linux-gnu/libX11.so
external/glfw-3.3.2/tests/icon: external/glfw-3.3.2/tests/CMakeFiles/icon.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dhruv/development/git/particle_attractor/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking C executable icon"
	cd /home/dhruv/development/git/particle_attractor/build/external/glfw-3.3.2/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/icon.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
external/glfw-3.3.2/tests/CMakeFiles/icon.dir/build: external/glfw-3.3.2/tests/icon

.PHONY : external/glfw-3.3.2/tests/CMakeFiles/icon.dir/build

external/glfw-3.3.2/tests/CMakeFiles/icon.dir/requires: external/glfw-3.3.2/tests/CMakeFiles/icon.dir/icon.c.o.requires
external/glfw-3.3.2/tests/CMakeFiles/icon.dir/requires: external/glfw-3.3.2/tests/CMakeFiles/icon.dir/__/deps/glad_gl.c.o.requires

.PHONY : external/glfw-3.3.2/tests/CMakeFiles/icon.dir/requires

external/glfw-3.3.2/tests/CMakeFiles/icon.dir/clean:
	cd /home/dhruv/development/git/particle_attractor/build/external/glfw-3.3.2/tests && $(CMAKE_COMMAND) -P CMakeFiles/icon.dir/cmake_clean.cmake
.PHONY : external/glfw-3.3.2/tests/CMakeFiles/icon.dir/clean

external/glfw-3.3.2/tests/CMakeFiles/icon.dir/depend:
	cd /home/dhruv/development/git/particle_attractor/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dhruv/development/git/particle_attractor /home/dhruv/development/git/particle_attractor/external/glfw-3.3.2/tests /home/dhruv/development/git/particle_attractor/build /home/dhruv/development/git/particle_attractor/build/external/glfw-3.3.2/tests /home/dhruv/development/git/particle_attractor/build/external/glfw-3.3.2/tests/CMakeFiles/icon.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : external/glfw-3.3.2/tests/CMakeFiles/icon.dir/depend

