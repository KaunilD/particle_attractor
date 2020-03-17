#ifndef MATERIAL_H
#define MATERIAL_H

#include "libs.hpp"

class Material {
public:
	QOpenGLTexture *texture;

	Material() = default;
	Material(const Material& t_material):
		texture(t_material.texture)
	{

	}
	Material(QString texturePath) {
		texture = new QOpenGLTexture(
			QImage(texturePath).mirrored()
		);
	}

	~Material() = default;
};

#endif MATERIAL_H