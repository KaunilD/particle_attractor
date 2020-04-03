#include "renderer/renderer.hpp"
#ifndef SUNRENDERER_H
#define SUNRENDERER_H

class SunRenderer : public Renderer {
public:
	SunRenderer() {};
	virtual void render(
		shared_ptr<ShaderProgram> shaderProgram,
		shared_ptr<GameObject> gameObject,
		shared_ptr<Camera> camera) override;
};

#endif SUNRENDERER_H