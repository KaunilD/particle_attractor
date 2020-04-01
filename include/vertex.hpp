#include "glm/glm.hpp"
#ifndef VERTEX_H
#define VERTEX_H

struct Vertex {
	glm::vec3 position; // 12bytes
	glm::vec3 normals;
	glm::vec2 texture;
	glm::vec3 color; // 12 bytes
	glm::vec3 tangent;
	glm::vec3 bitTangent;
};
#endif