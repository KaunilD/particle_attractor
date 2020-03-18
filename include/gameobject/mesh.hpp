#ifndef MESH_H
#define MESH_H

#include "libs.hpp"
#include "objreader.hpp"

class Mesh {
public:
	size_t i_count = 0, v_count = 0;
	QOpenGLBuffer *m_VAO, *m_IBO;
	
	Mesh() = default;

	Mesh(Utils::ObjReaderRet obj): i_count(obj.i.count()), v_count(obj.v.count()) {
		m_VAO = new QOpenGLBuffer();
		m_VAO->create();
		m_VAO->bind();
		m_VAO->allocate(
			obj.v.constData(),
			obj.v.count() * static_cast<int>(sizeof(Vertex))
		);

		m_IBO = new QOpenGLBuffer(QOpenGLBuffer::IndexBuffer);
		m_IBO->create();
		m_IBO->bind();
		m_IBO->allocate(
			obj.i.constData(),
			obj.i.count() * static_cast<int>(sizeof(GL_UNSIGNED_INT))
		);
	};

	~Mesh(){
		LOG("Mesh::Destroyed")
		m_IBO->destroy();
		m_VAO->destroy();
	}
};

#endif MESH_H