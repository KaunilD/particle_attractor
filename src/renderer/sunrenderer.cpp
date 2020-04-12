#include "renderer/sunrenderer.hpp"

void SunRenderer::render(shared_ptr<ShaderProgram> shaderProgram, shared_ptr<GameObject> gameObject, shared_ptr<Camera> camera) {
	shaderProgram->activate();
		
	shaderProgram->setMat4("projViewMat", camera->getProjectionMatrix() * camera->getViewMatrix());
	shaderProgram->setMat4("modelMat", gameObject->getModelMatrix());

	shaderProgram->setVec3("color", gameObject->getColor());

	glm::vec3 lightColor(
		1.00 + (1 + sin(2 * 3.14 * glfwGetTime() * 0.30)) * 0.250,
		0.65 + (1 + sin(2 * 3.14 * glfwGetTime() * 0.30)) * 0.200,
		0.00
	);

	shaderProgram->setVec3("color", lightColor);
	//shaderProgram->setVec3("light.diffuse", diffuseColor);
	gameObject->m_mesh->drawInstanced(100);

	shaderProgram->deactivate();

}

void SunRenderer::render(shared_ptr<ShaderProgram> shaderProgram, shared_ptr<std::vector<shared_ptr<GameObject>>> gameObject, shared_ptr<Camera> camera) {
}