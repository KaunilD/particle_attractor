#ifndef RENDERER_H
#define RENDERER_H

#include "libs.hpp"
#include "camera.hpp"
#include "shaderprogram.hpp"
#include "gameobject/gameobject.hpp"

class Renderer: protected QOpenGLFunctions {

public:
	Renderer() {
		initializeOpenGLFunctions();
	}

	virtual void render(
		shared_ptr<ShaderProgram> shaderProgram,
		shared_ptr<GameObject> gameObject,
		shared_ptr<Camera> camera
	) = 0;

};

#endif