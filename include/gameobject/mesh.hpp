#ifndef MESH_H
#define MESH_H

#include "libs.hpp"
#include "objreader.hpp"
#include "cuda_gl_interop.h"

class Mesh {
public:
	size_t i_count = 0, v_count = 0;
	GLuint m_VAO, m_VBO, m_EBO, m_models;
	struct cudaGraphicsResource* m_models_CUDA;
	float4* d_modelBuffer;
	size_t num_bytes;

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
		glBufferData(GL_ARRAY_BUFFER, v_count * sizeof(Vertex), &obj.v[0], GL_STATIC_DRAW);
		
		glGenBuffers(1, &m_EBO);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_EBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, i_count * sizeof(GLuint), &obj.i[0], GL_STATIC_DRAW);

		// set up vertex attributes, specify layout
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, position));
		
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normals));

		glBindVertexArray(0);


	};

	void prepareInstances(const std::vector<glm::mat4>& models) {
		glBindVertexArray(m_VAO);

		glGenBuffers(1, &m_models);
		glBindBuffer(GL_ARRAY_BUFFER, m_models);
		glBufferData(GL_ARRAY_BUFFER, models.size() * sizeof(glm::mat4), &models.at(0) , GL_STATIC_DRAW);

		auto vec4Size = sizeof(glm::vec4);

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


		glBindVertexArray(0);
		

		cudaGraphicsGLRegisterBuffer(&m_models_CUDA, m_models, cudaGraphicsRegisterFlagsWriteDiscard);

	}

	void getCUDAptr() {
		cudaGraphicsMapResources(1, &m_models_CUDA, 0);
		cudaGraphicsResourceGetMappedPointer((void**)&d_modelBuffer, &num_bytes, m_models_CUDA);
	}

	void drawInstanced(int numInstances) {

		cudaGraphicsUnmapResources(1, &m_models_CUDA, 0);

		glBindVertexArray(m_VAO);
		glDrawElementsInstanced(
			GL_TRIANGLES, i_count, GL_UNSIGNED_INT, 0, numInstances);
		glBindVertexArray(0);
		
	}


	~Mesh(){
		LOG("Mesh::Destroyed");

		glDisableVertexAttribArray(m_VAO);
		glDisableVertexAttribArray(m_VBO);
		glDisableVertexAttribArray(m_EBO);
	}
};

#endif MESH_H