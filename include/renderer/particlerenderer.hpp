#include "renderer/renderer.hpp"
#ifndef PARTICLERENDERER_H
#define PARTICLERENDERER_H

class ParticleRenderer : public Renderer{
public:
	ParticleRenderer() {};
	virtual void render(
		shared_ptr<ShaderProgram> shaderProgram, 
		shared_ptr<GameObject> gameObject,
		shared_ptr<Camera> camera) override;
};

#endif PARTICLERENDERER_H