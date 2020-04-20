#include "renderer/renderer.hpp"
#ifndef PARTICLERENDERER_H
#define PARTICLERENDERER_H

class ParticleRenderer : public Renderer {
public:
	ParticleRenderer() {
		LOG("ParticleRenderer::c_tor()");
	};
	virtual void render(
		shared_ptr<ShaderProgram> shaderProgram,
		shared_ptr<GameObject> gameObject,
		shared_ptr<Camera> camera) override;

	virtual void render(
		shared_ptr<ShaderProgram> shaderProgram,
		shared_ptr<std::vector<shared_ptr<GameObject>>> gameObject,
		shared_ptr<Camera> camera) override;
	virtual void render(
		shared_ptr<ShaderProgram> shaderProgram,
		shared_ptr<Scene> scene,
		shared_ptr<Camera> camera) override;
	virtual void render(
		shared_ptr<ShaderProgram> shaderProgram,
		shared_ptr<Scene> scene, GLuint material, const OpticalFlow& optflow,
		shared_ptr<Camera> camera) override;
};

#endif PARTICLERENDERER_H