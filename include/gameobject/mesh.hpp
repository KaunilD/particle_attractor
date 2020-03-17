#ifndef MESH_H
#define MESH_H

#include "libs.hpp"
#include "objreader.hpp"
#include "gameobject/gameobject.hpp"

class Mesh {
public:
	QOpenGLBuffer *m_VAO, *m_IBO;
	size_t i_count = 0, v_count = 0;

	Mesh() = default;
	
	Mesh(const Mesh& t_mesh) :
		m_VAO(t_mesh.m_VAO),
		m_IBO(t_mesh.m_IBO),
		i_count(t_mesh.i_count),
		v_count(t_mesh.v_count)
	{};

	Mesh(engine::ObjReader::ObjReaderRet obj) {
		m_VAO = new QOpenGLBuffer();
		m_VAO->create();
		m_VAO->bind();
		m_VAO->allocate(
			obj.v.constData(),
			obj.v.count() * static_cast<int>(sizeof(Vertex))
		);
		v_count = obj.v.count();

		m_IBO = new QOpenGLBuffer(QOpenGLBuffer::IndexBuffer);
		m_IBO->create();
		m_IBO->bind();
		m_IBO->allocate(
			obj.i.constData(),
			obj.i.count() * static_cast<int>(sizeof(GL_UNSIGNED_INT))
		);
		i_count = obj.i.count();
	};

	~Mesh(){
		m_IBO->destroy();
		m_VAO->destroy();
	}
};

#endif