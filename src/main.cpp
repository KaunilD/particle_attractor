#include "libs.hpp"
#include "camera.hpp"
#include "shaderprogram.hpp"
#include "renderer/particlerenderer.hpp"
#include "gameobject/gameobject.hpp"
#include "gameobject/sunobject.hpp"
#include "gameobject/particleobject.hpp"
#include "gameobject/mesh.hpp"
#include "algorithms/algorithms.hpp"

int frameBufferHeight = 0;
int frameBufferWidth = 0;

// FPS CONTROLLER
bool running = true;
double lastTime = 0.0;
double deltaTime;
double currentTime;
int frame = 0;
const double maxFPS = 360.0f;
const double maxPeriod = 1.0 / maxFPS;

// CAMERA
float cameraSpeed = 0.05f; // adjust accordingly

// GAMEOBJECTS
shared_ptr<Camera> camera;
shared_ptr<ShaderProgram> particleObjShader;
shared_ptr<ParticleRenderer> particleRenderer;
shared_ptr<std::vector<shared_ptr<GameObject>>> gameObjects;
shared_ptr<Mesh> m_sphereMesh;

void frameBufferResizeCb(GLFWwindow* window, int fbW, int fbH) {

	frameBufferWidth = fbW;
	frameBufferHeight = fbW;
	glViewport(0, 0, fbW, fbH);
}

void initGlRenderFlags() {
	glEnable(GL_DEPTH_TEST);

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

}


void initCSR() {
	camera = make_shared<Camera>(
		glm::vec3(0.0f, 0.0f, 10.0f),
		glm::vec3(0.0f, 0.0f, -1.0f),
		glm::vec3(0.0f, 1.0f, 0.0f),
		90.0f, VIEWPORT_WIDTH, VIEWPORT_WIDTH, 0.001f, 100.f
		);
	camera->setSpeed(0.5);

	particleObjShader = make_shared<ShaderProgram>();
	particleObjShader->loadShaders(
		"C:\\Users\\dhruv\\Development\\git\\particle_attractor\\src\\resources\\glsl\\object_vs.glsl",
		"C:\\Users\\dhruv\\Development\\git\\particle_attractor\\src\\resources\\glsl\\object_fs.glsl"
	);

	particleRenderer = make_shared<ParticleRenderer>();

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
	sunObject->setMass(1000.0f);
	sunObject->setMesh(m_sphereMesh);
	//sunObject->setMaterial(m_sphereMaterial);
	sunObject->setScale(glm::vec3(10.0f, 10.0f, 10.0f));
	gameObjects->push_back(std::move(sunObject));

	// setup particles

	shared_ptr<ParticleObject> particleObject;
	for (int i = 1; i < 100; i++) {
		particleObject = make_shared<ParticleObject>(false);
		particleObject->setMass(
			Algorithms::randomInt(10) / 10.f + 0.01
		);
		particleObject->setMesh(m_sphereMesh);
		//particleObject->setMaterial(m_sphereMaterial);
		particleObject->setScale(
			glm::vec3(
				particleObject->m_mass, particleObject->m_mass, particleObject->m_mass
			)
		);

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

void display(GLFWwindow* window) {
	glfwSwapBuffers(window);
	glFlush();
}

void updateInput(GLFWwindow* window) {
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GLFW_TRUE);
	}

	cameraSpeed = 0.1f * (float)deltaTime;
	/*
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
		camera.update(W, cameraSpeed);
		slrCamera.processKB(W, cameraSpeed);
	}
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
		camera.processKB(S, cameraSpeed);
		slrCamera.processKB(S, cameraSpeed);
	}
	if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
		camera.processKB(Q, cameraSpeed);
		slrCamera.processKB(Q, cameraSpeed);
	}
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
		camera.processKB(A, cameraSpeed);
		slrCamera.processKB(A, cameraSpeed);
	}
	if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS) {
		camera.processKB(Z, cameraSpeed);
		slrCamera.processKB(Z, cameraSpeed);
	}
	if (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS) {
		camera.processKB(X, cameraSpeed);
		slrCamera.processKB(X, cameraSpeed);
	}
	if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
		camera.processKB(UP, cameraSpeed);
		slrCamera.processKB(UP, cameraSpeed);
	}
	if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
		camera.processKB(DOWN, cameraSpeed);
		slrCamera.processKB(DOWN, cameraSpeed);
	}
	if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
		camera.processKB(LEFT, cameraSpeed);
		slrCamera.processKB(LEFT, cameraSpeed);
	}
	if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
		camera.processKB(RIGHT, cameraSpeed);
		slrCamera.processKB(RIGHT, cameraSpeed);
	}
	*/
}


int main(int argc, char *argv[])
{
	if (!glfwInit())
		return -1;

	
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
	glfwWindowHint(GLFW_SAMPLES, 4); // 4x antialiasing
	glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);

	GLFWwindow* window;
	window = glfwCreateWindow(
		VIEWPORT_WIDTH, VIEWPORT_HEIGHT,
		"PC Viewer", NULL, NULL
	);

	if (!window)
	{
		glfwTerminate();
		return -1;
	}

	glfwGetFramebufferSize(window, &frameBufferWidth, &frameBufferHeight);
	glViewport(0, 0, frameBufferWidth, frameBufferHeight);
	glfwSetFramebufferSizeCallback(window, frameBufferResizeCb);

	glfwMakeContextCurrent(window);
	// INIT OPENGL FLAGS

	// INITIALIZE GLEW
	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK) {
		std::cout << "ERROR::MAIN.CPP::GLEW_INIT_FAILED" << "/n";
		glfwTerminate();
	}

	initCSR();
	initObjects();
	while (!glfwWindowShouldClose(window)){

		currentTime = glfwGetTime();
		deltaTime = currentTime - lastTime;

		glfwPollEvents();

		updateInput(window);


		if (deltaTime >= maxPeriod) {
			glClearColor(0, 0, 0, 0);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			/*
				update and render gameObjects
			*/
			#pragma omp parallel for
			for (int i = 1; i < gameObjects->size(); i++) {
				gameObjects->at(i)->updateObject(
					frame, *gameObjects->at(0)
				);
			}

			for (int i = 0; i < gameObjects->size(); i++) {

				particleRenderer->render(
					particleObjShader,
					gameObjects->at(i),
					camera
				);
			}
			frame += 1;
		}
		display(window);

	}

	glfwDestroyWindow(window);
	glfwTerminate();

	return 0;
}