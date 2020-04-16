#version 440 core

layout (location = 0) in vec3 vertex_position;
layout (location = 1) in vec3 vertex_normal;

uniform highp mat4 projViewMat;
uniform highp mat4 modelMat;

out vec4 vs_position;
out vec3 vs_normal;

void main(){
	gl_Position = projViewMat * modelMat * vec4(vertex_position, 1.f);
	vs_position = modelMat * vec4(vertex_position, 1.f);
	vs_normal	= vertex_normal;
}