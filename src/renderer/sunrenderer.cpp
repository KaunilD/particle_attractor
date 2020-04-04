#include "renderer/sunrenderer.hpp"

void SunRenderer::render(shared_ptr<ShaderProgram> shaderProgram, shared_ptr<GameObject> gameObject, shared_ptr<Camera> camera) {
	{
		shaderProgram->activate();
		
		shaderProgram->setMat4("projViewMat", camera->getProjectionMatrix() * camera->getViewMatrix());
		shaderProgram->setMat4("modelMat", gameObject->getModelMatrix());

		shaderProgram->setVec3("color", gameObject->getColor());


		glm::vec3 lightColor(sin(glfwGetTime()), 0, 1.0f);

		
		shaderProgram->setVec3("color", lightColor);
		//shaderProgram->setVec3("light.diffuse", diffuseColor);

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

		shaderProgram->deactivate();
	};

}