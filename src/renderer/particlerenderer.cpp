#include "renderer/particlerenderer.hpp"

void ParticleRenderer::render(shared_ptr<ShaderProgram> shaderProgram, shared_ptr<GameObject> gameObject, shared_ptr<Camera> camera) {
	{

		shaderProgram->activate();
		
		shaderProgram->setMat4("modelMat", gameObject->getModelMatrix());
		shaderProgram->setMat4("normalMat", glm::mat3x3(
			glm::transpose(glm::inverse(gameObject->getModelMatrix()))));
		shaderProgram->setMat4("projViewMat", camera->getProjectionMatrix() * camera->getViewMatrix());
		
		shaderProgram->setVec3("color", gameObject->getColor());

		shaderProgram->setVec3("lightPos", glm::vec3(0.f, 0.f, 0.f));
		shaderProgram->setVec3("lightColor", glm::vec3(1.0f, 1.0f, 1.0f));
		shaderProgram->setVec3("lightAmbient", glm::vec3(1.0f, 0.0f, 1.0f));

		shaderProgram->setVec3("cameraEye", camera->m_posVector);


		GLuint textureAttribLoc, positionAttribLoc, normalAttribLoc, colorAttribLoc;
		glBindVertexArray(gameObject->m_mesh->m_VAO);
		
		// POSITION
		positionAttribLoc = glGetAttribLocation(shaderProgram->program, "vertex_position");
		glVertexAttribPointer(positionAttribLoc, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*)offsetof(Vertex, position));
		glEnableVertexAttribArray(positionAttribLoc);

		// NORMAL
		normalAttribLoc = glGetAttribLocation(shaderProgram->program, "vertex_normal");
		glVertexAttribPointer(normalAttribLoc, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*)offsetof(Vertex, normals));
		glEnableVertexAttribArray(normalAttribLoc);

		glDrawElements(GL_TRIANGLES, gameObject->m_mesh->i_count, GL_UNSIGNED_INT, 0);

		glDisableVertexAttribArray(positionAttribLoc);
		glDisableVertexAttribArray(normalAttribLoc);
	};

}