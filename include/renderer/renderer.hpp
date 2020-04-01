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
		ShaderProgram& shaderProgram,
		GameObject& gameObject,
		Camera& camera
	) = 0;

};

#endif