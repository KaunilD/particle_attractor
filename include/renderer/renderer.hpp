#ifndef RENDERER_H
#define RENDERER_H

#include "libs.hpp"
#include "camera.hpp"
#include "shaderprogram.hpp"
#include "gameobject/gameobject.hpp"
#include "scene.hpp"
#include "cuda/optflow.hpp"
class Renderer {

public:
	Renderer() = default;

	virtual void render(
		shared_ptr<ShaderProgram> shaderProgram,
		shared_ptr<Scene> scene, 
		shared_ptr<Material> material, 
		const OpticalFlow& optflow,
		shared_ptr<Camera> camera) = 0;

};

#endif