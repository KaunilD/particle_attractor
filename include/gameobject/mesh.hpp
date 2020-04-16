#ifndef MESH_H
#define MESH_H

#include "libs.hpp"
#include "objreader.hpp"
#include "cuda_gl_interop.h"

class Mesh {
public:
	size_t i_count = 0, v_count = 0;
	GLuint m_VAO, m_VBO, m_EBO, m_models, m_positions;
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
	Mesh(Utils::ObjReaderRet obj);
	void prepareInstances(const std::vector<glm::mat4>& models);
	void getCUDAptr();
	void drawInstanced(int numInstances);
	

	~Mesh(){
		LOG("Mesh::Destroyed");

		glDisableVertexAttribArray(m_VAO);
		glDisableVertexAttribArray(m_VBO);
		glDisableVertexAttribArray(m_EBO);
	}
};

#endif MESH_H