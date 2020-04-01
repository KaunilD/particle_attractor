#include "renderer/renderer.hpp"
#ifndef PARTICLERENDERER_H
#define PARTICLERENDERER_H

class ParticleRenderer : public Renderer{
public:
	ParticleRenderer() {};
	virtual void render(
		ShaderProgram& shaderProgram, 
		GameObject& gameObject, 
		Camera& camera) override;
};

#endif PARTICLERENDERER_H