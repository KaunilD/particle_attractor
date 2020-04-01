#include "renderer/particlerenderer.hpp"

void ParticleRenderer::render(shared_ptr<ShaderProgram> shaderProgram, shared_ptr<GameObject> gameObject, shared_ptr<Camera> camera) {
	{

		shaderProgram->activate();
		
		shaderProgram->setMat4(
			"projectionMatrix",
			camera->getProjectionMatrix()
		);
		shaderProgram->setMat4(
			"modelMatrix",
			gameObject->getModelMatrix()
		);
		shaderProgram->setMat4(
			"viewMatrix",
			camera->getViewMatrix()
		);

		shaderProgram->setVec3(
			"cameraEye",
			camera->posVector
		);
		shaderProgram->setVec3(
			"lightPos",
			glm::vec3(0.f, 0.f, 0.f)
		);
		shaderProgram->setVec3(
			"lightColor",
			glm::vec3(1.0f, 0.0f, 1.0f)
		);

		shaderProgram->setVec3(
			"lightAmbient",
			glm::vec3(1.0f, 0.0f, 1.0f)

		);


		GLuint textureAttribLoc, positionAttribLoc, normalAttribLoc, colorAttribLoc;
		glBindVertexArray(gameObject->m_mesh->m_VAO);
		
		// POSITION
		positionAttribLoc = glGetAttribLocation(shaderProgram->program, "vertex_position");
		glEnableVertexAttribArray(positionAttribLoc);
		glVertexAttribPointer(positionAttribLoc, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*)offsetof(Vertex, position));

		// NORMAL
		normalAttribLoc = glGetAttribLocation(shaderProgram->program, "vertex_normal");
		glEnableVertexAttribArray(normalAttribLoc);
		glVertexAttribPointer(normalAttribLoc, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*)offsetof(Vertex, normals));

		// COLOR
		colorAttribLoc = glGetAttribLocation(shaderProgram->program, "vertex_color");
		glEnableVertexAttribArray(colorAttribLoc);
		glVertexAttribPointer(colorAttribLoc, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*)offsetof(Vertex, color));


		glDrawElements(GL_TRIANGLES, gameObject->m_mesh->i_count, GL_UNSIGNED_INT, 0);

		glDisableVertexAttribArray(positionAttribLoc);
		glDisableVertexAttribArray(normalAttribLoc);
		glDisableVertexAttribArray(colorAttribLoc);
	};

}