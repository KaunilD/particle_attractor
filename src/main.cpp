#include "libs.hpp"
#include "camera.hpp"
#include "shaderprogram.hpp"
#include "renderer/particlerenderer.hpp"
#include "gameobject/gameobject.hpp"
#include "gameobject/sunobject.hpp"
#include "gameobject/particleobject.hpp"
#include "gameobject/mesh.hpp"
#include "algorithms/algorithms.hpp"
#include "window.hpp"

// FPS CONTROLLER
bool running = true;
double lastTime = 0.0;
double deltaTime;
double currentTime;
int frame = 0;
const double maxFPS = 60.0f;
const double maxPeriod = 1.0 / maxFPS;

// CAMERA
float cameraSpeed = 0.05f; // adjust accordingly
float fov = 45.0f;

// GAMEOBJECTS
shared_ptr<Camera> camera;
shared_ptr<ShaderProgram> particleShader, sunShader;
shared_ptr<ParticleRenderer> particleRenderer, sunRenderer;
shared_ptr<std::vector<shared_ptr<GameObject>>> gameObjects;
shared_ptr<Mesh> m_sphereMesh;

// MOUSE
float lastX = 400, lastY = 300;
float yaw = 0, pitch = 0;
bool firstMouse = true;

void initGlRenderFlags() {
	glEnable(GL_DEPTH_TEST);
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
}

void initCSR() {
	camera = make_shared<Camera>(
		glm::vec3(-10.0f, 0.0f, 5.0f),
		glm::vec3(0.0f, 0.0f, -1.0f),
		glm::vec3(0.0f, 1.0f, 0.0f),
		fov, VIEWPORT_WIDTH, VIEWPORT_WIDTH, 0.001f, 100.f
		);
	camera->setSpeed(0.5);

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

void initObjects() {

	gameObjects = make_shared<std::vector<shared_ptr<GameObject>>>();

	m_sphereMesh = make_shared<Mesh>(
		Utils::readObj(
			"C:\\Users\\dhruv\\Development\\git\\particle_attractor\\src\\resources\\objects\\sphere.obj"
		)
	);

	//m_sphereMaterial = make_shared<Material>(QString(".//resources//objects//blatt.png"));

	// setup sun
	shared_ptr<SunObject> sunObject = make_shared<SunObject>(true);
	sunObject->setMass(100.0f);
	sunObject->setMesh(m_sphereMesh);
	sunObject->setColor(glm::vec3(1.0f));
	sunObject->setScale(glm::vec3(10.0f));
	sunObject->setPosition(glm::vec3(0.0f));
	gameObjects->push_back(std::move(sunObject));

	// setup particles

	shared_ptr<ParticleObject> particleObject;
	for (int i = 1; i < 100; i++) {
		particleObject = make_shared<ParticleObject>(false);
		particleObject->setMass(
			Algorithms::randomInt(10)/10.0f +0.01
		);
		particleObject->setMesh(m_sphereMesh);
		//particleObject->setMaterial(m_sphereMaterial);
		particleObject->setScale(
			glm::vec3(particleObject->m_mass)
		);
		particleObject->setColor(glm::vec3(1.0, 0.0, 1.0));
		particleObject->setPosition(
			glm::vec3(
				Algorithms::randomInt(10) - 5,
				Algorithms::randomInt(10) - 5,
				Algorithms::randomInt(10) - 5
			)
		);

		gameObjects->push_back(std::move(particleObject));
	}

}

void updateGameObjects(float dt) {

#pragma omp parallel for
	for (int i = 1; i < gameObjects->size(); i++) {
		gameObjects->at(i)->updateObject(
			dt, *gameObjects->at(0)
		);
	}

	sunRenderer->render(
		sunShader,
		gameObjects->at(0),
		camera
	);

	for (int i = 1; i < gameObjects->size(); i++) {

		particleRenderer->render(
			particleShader,
			gameObjects->at(i),
			camera
		);
	}
}

void display(GLFWwindow* window) {
	glfwSwapBuffers(window);
	glFlush();
}

void mouseCallback(GLFWwindow* window, double xpos, double ypos){
	if (firstMouse)
	{
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}

	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos; // reversed since y-coordinates range from bottom to top
	lastX = xpos;
	lastY = ypos;

	const float sensitivity = 0.05f;
	xoffset *= sensitivity;
	yoffset *= sensitivity;


	yaw += xoffset;
	pitch += yoffset;

	if (pitch > 89.0f)
		pitch = 89.0f;
	if (pitch < -89.0f)
		pitch = -89.0f;

}

void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
	if (fov >= 1.0f && fov <= 90.0f)
		fov -= yoffset;
	else if (fov < 1.0f)
		fov = 1.0f;
	else if (fov > 90.0f)
		fov = 89.0f;
}


int main(int argc, char *argv[])
{
	Window window(1000, 1000, "Particle Attractor");
	
	shared_ptr<MouseEvent> mouseEvent(new MouseEvent());
	shared_ptr<KeyboardEvent> keyboardEvent(new KeyboardEvent());

	window.attatchEventHandlers(mouseEvent, keyboardEvent);
	window.makeCurrent();
	
	mouseEvent->addListener(camera);

	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK) {
		std::cout << "ERROR::MAIN.CPP::GLEW_INIT_FAILED" << std::endl;
		glfwTerminate();
	}

	initCSR();
	initObjects();
	while (!glfwWindowShouldClose(window.window)){

		currentTime = glfwGetTime();
		deltaTime = currentTime - lastTime;

		glfwPollEvents();
		camera->update(yaw, pitch, fov);
		
		mouseEvent->dispatchEvents();
		
		if (deltaTime >= maxPeriod) {
			glClearColor(0, 0, 0, 0);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			/*
				update and render gameObjects
			*/

			updateGameObjects(deltaTime/1000.0f);

			frame += 1;
		}
		display(window.window);

	}

	glfwDestroyWindow(window.window);
	glfwTerminate();

	return 0;
}