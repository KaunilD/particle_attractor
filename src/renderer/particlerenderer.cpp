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
	shaderProgram->setVec3("light.position",	camera->m_posVector);
	shaderProgram->setVec3("light.ambient",		glm::vec3(0.6f));
	shaderProgram->setVec3("light.diffuse",		glm::vec3(1.0f, 0.0f, 1.0f));
	shaderProgram->setVec3("light.specular",	glm::vec3(1.0f));

	/* material */
	shaderProgram->setFloat("material.shininess",	32.0f);
	shaderProgram->setVec3("material.ambient",		glm::vec3(0.5f));
	shaderProgram->setVec3("material.diffuse",		glm::vec3(1.0f, 0.0f, 1.0f));
	shaderProgram->setVec3("material.specular",		glm::vec3(0.5f));

	shaderProgram->setVec3("cameraEye", camera->m_posVector);

	gameObject->m_mesh->drawInstanced(100);

	shaderProgram->deactivate();
};


void ParticleRenderer::render(shared_ptr<ShaderProgram> shaderProgram, shared_ptr<std::vector<shared_ptr<GameObject>>> gameObjects, shared_ptr<Camera> camera) {

	
	shaderProgram->activate();

	shaderProgram->setMat4("projViewMat", camera->getProjectionMatrix() * camera->getViewMatrix());

	/* light */
	shaderProgram->setVec3("light.position",	-camera->m_posVector);
	shaderProgram->setVec3("light.ambient",		glm::vec3(0.2f));
	shaderProgram->setVec3("light.diffuse",		glm::vec3(0.5f));
	shaderProgram->setVec3("light.specular",	glm::vec3(1.00f));

	/* material */
	shaderProgram->setFloat("material.shininess",	32.0f);
	shaderProgram->setVec3("material.ambient",		glm::vec3(0.2f));
	shaderProgram->setVec3("material.diffuse",		glm::vec3(1.0f));
	shaderProgram->setVec3("material.specular",		glm::vec3(1.0f));

	shaderProgram->setVec3("cameraEye", camera->m_posVector);

	gameObjects->at(0)->m_mesh->drawInstanced(100);

	shaderProgram->deactivate();
};


void ParticleRenderer::render(shared_ptr<ShaderProgram> shaderProgram, shared_ptr<Scene> scene, shared_ptr<Camera> camera) {


	shaderProgram->activate();

	shaderProgram->setMat4("projViewMat", camera->getProjectionMatrix() * camera->getViewMatrix());
	/* object */

	/* light */
	shaderProgram->setVec3("light.position", -camera->m_posVector);
	shaderProgram->setVec3("light.ambient", glm::vec3(0.2f));
	shaderProgram->setVec3("light.diffuse", glm::vec3(0.5f));
	shaderProgram->setVec3("light.specular", glm::vec3(1.00f));

	/* material */
	shaderProgram->setFloat("material.shininess", 32.0f);
	shaderProgram->setVec3("material.ambient", glm::vec3(0.2f));
	shaderProgram->setVec3("material.diffuse", glm::vec3(0.5, 1, 1.0f));
	shaderProgram->setVec3("material.specular", glm::vec3(1.0f));

	shaderProgram->setVec3("cameraEye", camera->m_posVector);

	scene->update();

	shaderProgram->deactivate();
};