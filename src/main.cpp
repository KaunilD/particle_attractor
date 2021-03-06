#include "libs.hpp"
#include "camera.hpp"
#include "shaderprogram.hpp"
#include "renderer/particlerenderer.hpp"
#include "gameobject/gameobject.hpp"
#include "gameobject/sunobject.hpp"
#include "gameobject/particleobject.hpp"
#include "gameobject/mesh.hpp"
#include "algorithms/algorithms.hpp"
#include "scene.hpp"
#include "window.hpp"
#include "cuda/optflow.hpp"

int main(int argc, char* argv[])
{

	int height = 1080, width = 1920;
	double lastTime = glfwGetTime();
	int frameCount = 0;
	cv::Mat lastFrame, currentFrame;

	/*
		Initialize Window
	*/
	Window window(height, width, "Particle Attractor");
	window.makeCurrent();

	shared_ptr<MouseEvent> mouseEvent(new MouseEvent());
	shared_ptr<WindowEvent> windowEvent(new WindowEvent(height, width));

	window.attachEventHandler(mouseEvent);
	window.attachEventHandler(windowEvent);

	cudaSetDevice(0);
	cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
	/*
		Initialiize Scene
	*/
	shared_ptr<Scene> scene(new Scene(height, width));
	scene->init();

	/*
		Initialiize Meshes
	*/
	shared_ptr<Material> m_currentMaterial(new Material(height, width));

	shared_ptr<Mesh> m_shpereMesh(new Mesh(
		Utils::readObj(
			"./resources/objects/sphere.obj"
		)
	));

	/*
		Add mesh to scene
	*/
	scene->addMesh(m_shpereMesh);

	/*
		Initialiize Renderer
	*/
	shared_ptr<ParticleRenderer> particleRenderer(new ParticleRenderer());

	/*
		Initialiize Shader
	*/

	shared_ptr<ShaderProgram> particleShader(new ShaderProgram());
	particleShader->loadShaders(
		"./glsl/object_vs_instanced.glsl",
		"./glsl/object_fs.glsl"
	);

	/*
		Initialiize Camera
	*/
	shared_ptr<Camera> camera(new Camera(
		glm::vec3(0.0f, 0.0f, 2.0f),
		glm::vec3(0.0f, 0.0f, -1.0f),
		glm::vec3(0.0f, 1.0f, 0.0f),
		45.0f, height, width, 0.1f, 100.f
	));
	camera->setSpeed(0.05);
	/*
		Add Mouse and Window event-listeners
		to Camera
	*/
	mouseEvent->addListener(camera);
	windowEvent->addListener(camera);


	cv::VideoCapture cap;
	if (!cap.open("./resources/candle_high.mp4")) {
		return 0;
	}

	OpticalFlow optFlow(height, width);

	cap >> currentFrame;

	while (!glfwWindowShouldClose(window.window)) {
		frameCount++;

		glfwPollEvents();
		mouseEvent->dispatchEvents();
		windowEvent->dispatchEvents();


		window.clearCanvas();
		lastFrame = std::move(currentFrame);
		cap >> currentFrame;

		if (lastFrame.empty() || currentFrame.empty()) {
			cap.set(cv::CAP_PROP_POS_FRAMES, 0);
			continue;
		}

		m_currentMaterial->updateFrame(lastFrame);


		optFlow.copy(lastFrame, currentFrame);

		launch_fill(optFlow.d_uv1, 0, height * width);
		launch_convert(
			optFlow.d_f1ptr,
			optFlow.d_f1ptrf32,
			height * width
		);

		launch_convert(
			optFlow.d_f2ptr,
			optFlow.d_f2ptrf32,
			height * width
		);
		launch_gray(
			optFlow.d_f1ptrf32,
			optFlow.d_f1ptrf32gray,
			height * width
		);

		launch_gray(
			optFlow.d_f2ptrf32,
			optFlow.d_f2ptrf32gray,
			height * width
		);

		launch_partials(
			optFlow.d_f1ptrf32gray,
			optFlow.d_f1dx, optFlow.d_f1dy,
			height, width
		);

		launch_partials(
			optFlow.d_f2ptrf32gray,
			optFlow.d_f2dx, optFlow.d_f2dy,
			height, width
		);

		launch_sub(
			optFlow.d_f1ptrf32gray, optFlow.d_f2ptrf32gray,
			optFlow.d_dt,
			height, width
		);

		launch_optflow(
			optFlow.d_f1dx, optFlow.d_f1dy, optFlow.d_f2dx, optFlow.d_f2dy, optFlow.d_dt,
			optFlow.d_uv1, optFlow.d_uv2,
			height, width
		);
		for (int i = 0; i < 2; i++) {
			launch_convection(optFlow.d_uv2, optFlow.d_uv3, optFlow.p, height, width);

		}

		particleRenderer->render(
			particleShader,
			scene,
			m_currentMaterial, optFlow,
			camera
		);

		window.update();
		frameCount++;
	}

	window.destroy();
	return 0;
}
