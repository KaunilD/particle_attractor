#include "libs.hpp"
#include "camera.hpp"
#include "shaderprogram.hpp"
#include "renderer/particlerenderer.hpp"
#include "renderer/sunrenderer.hpp"
#include "gameobject/gameobject.hpp"
#include "gameobject/sunobject.hpp"
#include "gameobject/particleobject.hpp"
#include "gameobject/mesh.hpp"
#include "algorithms/algorithms.hpp"
#include "window.hpp"

void initShaders() {
	shared_ptr<ShaderProgram> particleShader, sunShader;
	shared_ptr<ParticleRenderer> particleRenderer, sunRenderer;

	particleShader = make_shared<ShaderProgram>();
	particleShader->loadShaders(
		"C:\\Users\\dhruv\\Development\\git\\particle_attractor\\src\\resources\\glsl\\object_vs.glsl",
		"C:\\Users\\dhruv\\Development\\git\\particle_attractor\\src\\resources\\glsl\\object_fs.glsl"
	);

	sunShader = make_shared<ShaderProgram>();
	sunShader->loadShaders(
		"C:\\Users\\dhruv\\Development\\git\\particle_attractor\\src\\resources\\glsl\\object_vs.glsl",
		"C:\\Users\\dhruv\\Development\\git\\particle_attractor\\src\\resources\\glsl\\sun_fs.glsl"
	);

	particleRenderer = make_shared<ParticleRenderer>();
	sunRenderer = make_shared<ParticleRenderer>();

}

void initGameObejcts(shared_ptr<std::vector<shared_ptr<GameObject>>> gameObjects, shared_ptr<Mesh> mesh) {
	
	// setup sun
	unique_ptr<SunObject> sunObject = make_unique<SunObject>(true);
	sunObject->setMass(100.0f);
	sunObject->setMesh(mesh);
	sunObject->setColor(glm::vec3(1.0f));
	sunObject->setScale(glm::vec3(1.00f));
	sunObject->setPosition(glm::vec3(0.0f));
	gameObjects->push_back(std::move(sunObject));

	// setup particles

	shared_ptr<ParticleObject> particleObject;
	for (int i = 1; i < 100; i++) {
		particleObject = make_shared<ParticleObject>(false);
		particleObject->setMass(
			Algorithms::randomInt(10) / 10.0f + 0.01
		);
		particleObject->setMesh(mesh);
		//particleObject->setMaterial(m_sphereMaterial);
		particleObject->setScale(
			glm::vec3(0.5f)
		);
		particleObject->setColor(glm::vec3(1.0, 0.0, 1.0));
		particleObject->setPosition(glm::vec3(
			Algorithms::randomInt(10) - 5,
			Algorithms::randomInt(10) - 5,
			Algorithms::randomInt(10) - 5
		));

		gameObjects->push_back(std::move(particleObject));
	}
}

void updateObjects(float dt,
	shared_ptr<std::vector<shared_ptr<GameObject>>> gameObjects) {
	#pragma omp parallel for
	for (int i = 1; i < gameObjects->size(); i++) {
		gameObjects->at(i)->updateObject(
			dt, *gameObjects->at(0)
		);
	}
}
/*
	NEEDSTO BE REFACTORED INTO SCENES
*/
void renderObjects(
	shared_ptr<std::vector<shared_ptr<GameObject>>> gameObjects,
	shared_ptr<ShaderProgram> particleShader, shared_ptr<ShaderProgram> sunShader,
	shared_ptr<ParticleRenderer> particleRenderer, shared_ptr<SunRenderer> sunRenderer,
	shared_ptr<Camera> camera) {
	for (int i = 0; i < gameObjects->size(); i++) {
		particleRenderer->render(
			particleShader, gameObjects->at(i), camera
		);
	}
	sunRenderer->render(
		sunShader, gameObjects->at(0), camera
	);
}

int main(int argc, char *argv[])
{
	float dt = 1/20.0f;
	Window window(1000, 1000, "Particle Attractor");
	window.makeCurrent();

	shared_ptr<MouseEvent> mouseEvent(new MouseEvent());
	shared_ptr<KeyboardEvent> keyboardEvent(new KeyboardEvent());

	window.attatchEventHandlers(mouseEvent, keyboardEvent);

	shared_ptr<Camera> camera(new Camera(
		glm::vec3(0.0f, 0.0f, 10.0f),
		glm::vec3(0.0f, 0.0f, -1.0f),
		glm::vec3(0.0f, 1.0f, 0.0f),
		45.0f, 1000, 1000, 0.001f, 100.f
	));
	camera->setSpeed(0.05);

	shared_ptr<Mesh> m_shpereMesh(new Mesh(
		Utils::readObj(
			"C:\\Users\\dhruv\\Development\\git\\particle_attractor\\src\\resources\\objects\\sphere.obj"
		)
	));

	shared_ptr<ShaderProgram> particleShader(new ShaderProgram());
	particleShader->loadShaders(
		"C:\\Users\\dhruv\\Development\\git\\particle_attractor\\src\\resources\\glsl\\object_vs_instanced.glsl",
		"C:\\Users\\dhruv\\Development\\git\\particle_attractor\\src\\resources\\glsl\\object_fs.glsl"
	);
	shared_ptr<ShaderProgram> sunShader(new ShaderProgram());
	sunShader->loadShaders(
		"C:\\Users\\dhruv\\Development\\git\\particle_attractor\\src\\resources\\glsl\\object_vs.glsl",
		"C:\\Users\\dhruv\\Development\\git\\particle_attractor\\src\\resources\\glsl\\sun_fs.glsl"
	);

	shared_ptr<ParticleRenderer> particleRenderer(new ParticleRenderer());
	shared_ptr<SunRenderer> sunRenderer(new SunRenderer());


	shared_ptr<std::vector<shared_ptr<GameObject>>> gameObjects(
		new std::vector<shared_ptr<GameObject>>());
	initGameObejcts(gameObjects, m_shpereMesh);

	
	mouseEvent->addListener(camera);

	while (!glfwWindowShouldClose(window.window)){

		glfwPollEvents();
		mouseEvent->dispatchEvents();
		updateObjects(dt, gameObjects);
		window.clearCanvas();
		// render particles
		
		particleRenderer->render(
			particleShader, gameObjects, camera
		);
		
		window.update();
		
	}
	window.destroy();
	return 0;
}