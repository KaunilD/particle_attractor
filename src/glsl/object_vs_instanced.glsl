#version 440 core

layout(location = 0) in vec3 vertex_position;
layout(location = 1) in vec3 vertex_normal;
layout(location = 2) in mat4 instance_matrix;
layout(location = 6) in vec4 instance_position;
layout(location = 7) in vec4 instance_velocity;

uniform highp mat4 projViewMat;

out vec4 vs_position;
out vec3 vs_normal;
out vec4 vs_velocity;

void main() {
	gl_Position = projViewMat * instance_matrix * vec4(vertex_position, 1.f);

	vs_position = instance_position;
	vs_normal	= vertex_normal;
	vs_velocity	= instance_velocity;
}