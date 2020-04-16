#ifndef MATERIAL_H
#define MATERIAL_H

#include "libs.hpp"

class Material {
public:
	std::string m_texturePath;
    GLuint m_texture;
    
    Material() = default;
    Material(const Material& t_material){

	}

    Material(const std::string& texturePath);

	~Material() {
		LOG("Material::Destroyed")
	};
};
#endif MATERIAL_H