#ifndef MATERIAL_H
#define MATERIAL_H

#include "libs.hpp"
class Material {
public:
	std::string m_texturePath;
    GLuint m_texture;
	GLuint m_pyramid[3];
    
    Material() = default;
	Material(const Material& t_material);
    Material(const std::string& texturePath);
	Material(int, int);
	void updateFrame(cv::Mat);

	~Material() {
		LOG("Material::Destroyed")
	};
};
#endif MATERIAL_H