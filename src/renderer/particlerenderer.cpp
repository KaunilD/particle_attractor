#include "renderer/particlerenderer.hpp"

void ParticleRenderer::render(shared_ptr<ShaderProgram> shaderProgram, shared_ptr<GameObject> gameObject, shared_ptr<Camera> camera) {

	shaderProgram->activate();

	shaderProgram->setMat4("modelMat", gameObject->getModelMatrix());
	
	shaderProgram->setMat4("normalMat", glm::mat3x3(
		glm::transpose(glm::inverse(gameObject->getModelMatrix()))));

	shaderProgram->setMat4("projViewMat", camera->getProjectionMatrix() * camera->getViewMatrix());
	/* object */
	shaderProgram->setVec3("color", gameObject->getColor());

	/* light */
	shaderProgram->setVec3("light.position",	glm::vec3(0.f));
	shaderProgram->setVec3("light.ambient",		glm::vec3(0.5f));
	shaderProgram->setVec3("light.diffuse",		glm::vec3(0.2f));
	shaderProgram->setVec3("light.specular",	glm::vec3(1.0f));

	/* material */
	shaderProgram->setFloat("material.shininess",	32.0f);
	shaderProgram->setVec3("material.ambient",		glm::vec3(0.5f));
	shaderProgram->setVec3("material.diffuse",		glm::vec3(1.0f, 0.0f, 1.0f));
	shaderProgram->setVec3("material.specular",		glm::vec3(0.5f));

	shaderProgram->setVec3("cameraEye", camera->m_posVector);

	gameObject->m_mesh->draw();

	shaderProgram->deactivate();
};


void ParticleRenderer::render(shared_ptr<ShaderProgram> shaderProgram, shared_ptr<std::vector<shared_ptr<GameObject>>> gameObjects, shared_ptr<Camera> camera) {

	
	unique_ptr<std::vector<glm::mat4x4>> model_mats = make_unique<std::vector<glm::mat4x4>>();
	
	for (int i = 1; i < gameObjects->size(); i++) {
		model_mats->push_back(
			gameObjects->at(i)->getModelMatrix()
		);
	}

	
	shaderProgram->activate();

	shaderProgram->setMat4("projViewMat", camera->getProjectionMatrix() * camera->getViewMatrix());
	/* object */
	
	/* light */

	/* light */
	shaderProgram->setVec3("light.position",	glm::vec3(0.f));
	shaderProgram->setVec3("light.ambient",		glm::vec3(0.2f));
	shaderProgram->setVec3("light.diffuse",		glm::vec3(1.0f));
	shaderProgram->setVec3("light.specular",	glm::vec3(1.0f));

	/* material */
	shaderProgram->setFloat("material.shininess",	32.0f);
	shaderProgram->setVec3("material.ambient",		glm::vec3(1.0f, 0.5f, 0.31f));
	shaderProgram->setVec3("material.diffuse",		glm::vec3(1.0f, 0.5f, 0.31f));
	shaderProgram->setVec3("material.specular",		glm::vec3(0.5f));

	shaderProgram->setVec3("cameraEye", camera->m_posVector);

	gameObjects->at(0)->m_mesh->draw(std::move(model_mats));

	shaderProgram->deactivate();
};