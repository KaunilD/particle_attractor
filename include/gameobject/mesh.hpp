#ifndef MESH_H
#define MESH_H

#include "libs.hpp"
#include "objreader.hpp"

class Mesh {
public:
	size_t i_count = 0, v_count = 0;
	GLuint m_VAO, m_VBO, m_EBO;
	
	Mesh() = default;
	enum Buffers {
		POSITION,
		NORMAL,
		TEXTURE,
		COLOR
	};
	Mesh(Utils::ObjReaderRet obj): i_count(obj.i.size()), v_count(obj.v.size()) {

		glGenVertexArrays(1, &m_VAO);
		glBindVertexArray(m_VAO);

		glGenBuffers(1, &m_VBO);
		glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
		glBufferData(GL_ARRAY_BUFFER, obj.v.size() * sizeof(Vertex), &obj.v[0], GL_STATIC_DRAW);

		// set up vertex attributes, specify layout
		glEnableVertexAttribArray(Buffers::POSITION);
		glVertexAttribPointer(Buffers::POSITION, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, position));
		
		glEnableVertexAttribArray(Buffers::NORMAL);
		glVertexAttribPointer(Buffers::NORMAL, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normals));

		glGenBuffers(1, &m_EBO);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_EBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, obj.i.size() * sizeof(GLuint), &obj.i[0], GL_STATIC_DRAW);

	};

	void draw() {
		
		glBindVertexArray(m_VAO);	
		glDrawElements(GL_TRIANGLES, i_count, GL_UNSIGNED_INT, 0);
		glBindVertexArray(0);
	}

	void draw(shared_ptr<std::vector<glm::mat4x4>> models) {

		GLuint buffer;
		glGenBuffers(1, &buffer);
		glBindBuffer(GL_ARRAY_BUFFER, buffer);
		glBufferData(GL_ARRAY_BUFFER, models->size() * sizeof(glm::mat4), &models->at(0), GL_STATIC_DRAW);

		glBindVertexArray(m_VAO);

		std::size_t vec4Size = sizeof(glm::vec4);

		glEnableVertexAttribArray(2);
		glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 4 * vec4Size, (void*)0);

		glEnableVertexAttribArray(3);
		glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 4 * vec4Size, (void*)(1 * vec4Size));

		glEnableVertexAttribArray(4);
		glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, 4 * vec4Size, (void*)(2 * vec4Size));

		glEnableVertexAttribArray(5);
		glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, 4 * vec4Size, (void*)(3 * vec4Size));

		glVertexAttribDivisor(2, 1);
		glVertexAttribDivisor(3, 1);
		glVertexAttribDivisor(4, 1);
		glVertexAttribDivisor(5, 1);

		glDrawElementsInstanced(
			GL_TRIANGLES, i_count, GL_UNSIGNED_INT, 0, models->size());

	}
	~Mesh(){
		LOG("Mesh::Destroyed")
	}
};

#endif MESH_H