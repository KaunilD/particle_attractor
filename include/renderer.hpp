#ifndef RENDERER_H
#define RENDERER_H

#include "libs.hpp"

class Renderer: protected QOpenGLFunctions {

public:
	Renderer() {
		initializeOpenGLFunctions();
	}
	void render(
		const ShaderProgram & shaderProgram,
		QOpenGLBuffer & attributeBuffer,
		QOpenGLBuffer & indexBuffer,
		QOpenGLTexture & texture
	) {
		
		GLuint textureAttribLoc, positionAttribLoc, normalAttribLoc, colorAttribLoc;
		attributeBuffer.bind();
		indexBuffer.bind();

		// POSITION
		positionAttribLoc = shaderProgram.program->attributeLocation("vertex_position");
		shaderProgram.program->enableAttributeArray(positionAttribLoc);
		shaderProgram.program->setAttributeBuffer(positionAttribLoc, GL_FLOAT, (int)offsetof(Vertex, position), 3, sizeof(Vertex));

		// NORMAL
		normalAttribLoc = shaderProgram.program->attributeLocation("vertex_normal");
		shaderProgram.program->enableAttributeArray(normalAttribLoc);
		shaderProgram.program->setAttributeBuffer(normalAttribLoc, GL_FLOAT, (int)offsetof(Vertex, normals), 3, sizeof(Vertex));

		// COLOR
		colorAttribLoc = shaderProgram.program->attributeLocation("vertex_color");
		shaderProgram.program->enableAttributeArray(colorAttribLoc);
		shaderProgram.program->setAttributeBuffer(colorAttribLoc, GL_FLOAT, (int)offsetof(Vertex, color), 3, sizeof(Vertex));


		glDrawElements(GL_TRIANGLES, attributeBuffer.size(), GL_UNSIGNED_INT, 0);

		glDisableVertexAttribArray(positionAttribLoc);
		glDisableVertexAttribArray(normalAttribLoc);
		glDisableVertexAttribArray(colorAttribLoc);
	};
};

#endif