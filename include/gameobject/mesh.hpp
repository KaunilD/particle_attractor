#ifndef MESH_H
#define MESH_H

#include "libs.hpp"
#include "objreader.hpp"

class Mesh {
public:
	size_t i_count = 0, v_count = 0;
	GLuint m_VAO, m_VBO, m_EBO;
	
	Mesh() = default;

	Mesh(Utils::ObjReaderRet obj): i_count(obj.i.size()), v_count(obj.v.size()) {

		glGenVertexArrays(1, &m_VAO);
		glGenBuffers(1, &m_VBO);
		glGenBuffers(1, &m_EBO);

		glBindVertexArray(m_VAO);

		glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
		glBufferData(GL_ARRAY_BUFFER, obj.v.size() * sizeof(Vertex), &obj.v[0], GL_STATIC_DRAW);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_EBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, obj.i.size() * sizeof(GLuint), &obj.i[0], GL_STATIC_DRAW);

	};

	~Mesh(){
		LOG("Mesh::Destroyed")
	}
};

#endif MESH_H