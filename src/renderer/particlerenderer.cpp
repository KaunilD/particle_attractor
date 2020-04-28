#include "renderer/particlerenderer.hpp"


void ParticleRenderer::render(
	shared_ptr<ShaderProgram> shaderProgram,
	shared_ptr<Scene> scene, 
	shared_ptr<Material> material, 
	const OpticalFlow& optflow,
	shared_ptr<Camera> camera
) {
	shaderProgram->activate();
	shaderProgram->setMat4("projViewMat", camera->getProjectionMatrix() * camera->getViewMatrix());
	shaderProgram->setVec3("cameraEye", camera->m_posVector);

	shaderProgram->setVec3("light.position", glm::vec3(1, 1, 1));
	shaderProgram->setVec3("light.ambient", glm::vec3(0.2f));
	shaderProgram->setVec3("light.diffuse", glm::vec3(0.5f));
	shaderProgram->setVec3("light.specular", glm::vec3(0.50f));

	/* material */
	shaderProgram->setFloat("material.shininess", 0.088);
	shaderProgram->setVec3("material.ambient", glm::vec3(0.25, 0.20725, 0.20725));
	shaderProgram->setVec3("material.diffuse", glm::vec3(1, 0.829, 0.829));
	shaderProgram->setVec3("material.specular", glm::vec3(0.296648, 0.296648, 0.296648));


	material->bind();
	scene->update(optflow.d_uv3);

	shaderProgram->deactivate();
};