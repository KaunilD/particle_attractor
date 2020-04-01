#include "renderer/particlerenderer.hpp"

void ParticleRenderer::render(ShaderProgram& shaderProgram, GameObject& gameObject, Camera& camera) {
	{

		shaderProgram.activate();
		
		shaderProgram.setUniform(
			"projectionMatrix",
			camera.getProjectionMatrix()
		);
		shaderProgram.setUniform(
			"modelMatrix",
			gameObject.getModelMatrix()
		);
		shaderProgram.setUniform(
			"viewMatrix",
			camera.getViewMatrix()
		);

		shaderProgram.setUniform(
			"cameraEye",
			camera.posVector
		);

		GLuint textureAttribLoc, positionAttribLoc, normalAttribLoc, colorAttribLoc;
		gameObject.m_mesh->m_VAO->bind();
		gameObject.m_mesh->m_IBO->bind();

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


		glDrawElements(GL_TRIANGLES, gameObject.m_mesh->i_count, GL_UNSIGNED_INT, 0);

		glDisableVertexAttribArray(positionAttribLoc);
		glDisableVertexAttribArray(normalAttribLoc);
		glDisableVertexAttribArray(colorAttribLoc);
	};

}