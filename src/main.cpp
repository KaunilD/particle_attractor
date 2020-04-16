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
#include "scene.hpp"
#include "window.hpp"


int main(int argc, char* argv[])
{
	/*
		Initialize Window
	*/
	Window window(1280, 720, "Particle Attractor");
	window.makeCurrent();

	shared_ptr<MouseEvent> mouseEvent(new MouseEvent());
	shared_ptr<WindowEvent> windowEvent(new WindowEvent(1280, 720));

	window.attachEventHandler(mouseEvent);
	window.attachEventHandler(windowEvent);

	cudaGLSetGLDevice(0);

	/*
		Initialiize Scene
	*/
	shared_ptr<Scene> scene(new Scene(360, 640));
	scene->init();

	/*
		Initialiize Meshes
	*/

	shared_ptr<Material> m_material(new Material(
		"resources\\250.png"
		)
	);
	
	shared_ptr<Mesh> m_shpereMesh(new Mesh(
		Utils::readObj(
			"resources\\objects\\sphere.obj"
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
		"glsl\\object_vs_instanced.glsl",
		"glsl\\object_fs.glsl"
	);

	/*
		Initialiize Camera
	*/
	shared_ptr<Camera> camera(new Camera(
		glm::vec3(0.0f, 0.0f, 2.0f),
		glm::vec3(0.0f, 0.0f, -1.0f),
		glm::vec3(0.0f, 1.0f, 0.0f),
		45.0f, 1000, 1000, 0.001f, 100.f
	));
	camera->setSpeed(0.05);
	/*
		Add Mouse and Window event-listeners 
		to Camera
	*/
	mouseEvent->addListener(camera);
	windowEvent->addListener(camera);


	cv::VideoCapture cap;
	if (!cap.open("movie.mp4")) {
		return 0;
	}
	// the camera will be closed automatically upon exit
	// cap.close();


	while (!glfwWindowShouldClose(window.window)){

		glfwPollEvents();
		mouseEvent->dispatchEvents();
		windowEvent->dispatchEvents();
		
		window.clearCanvas();
			
		cv::Mat frame;
		cap >> frame;
		if (!frame.empty()) {
			m_material->updateFrame(frame);
		}
		particleRenderer->render(
			particleShader, scene, m_material, camera
		);
		window.update();
	
	}

	window.destroy();
	return 0;
}