#ifndef MATERIAL_H
#define MATERIAL_H

#include "libs.hpp"

class Material {
public:
	QString m_texturePath;
	unique_ptr<QOpenGLTexture> m_texture;
	Material() = default;
	Material(const Material& t_material): m_texturePath(t_material.m_texturePath){

		m_texture = make_unique<QOpenGLTexture>(
			QImage(t_material.m_texturePath).mirrored()
		);
	}

	Material(QString texturePath): m_texturePath(texturePath) {
		m_texture = make_unique<QOpenGLTexture>(
			QImage(texturePath).mirrored()
		);
	}

	~Material() {
		LOG("Material::Destroyed")
	};
};

#endif MATERIAL_H