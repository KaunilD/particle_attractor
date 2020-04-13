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


int main(int argc, char *argv[])
{
	/*
		Initialize Window
	*/
	Window window(1000, 1000, "Particle Attractor");
	window.makeCurrent();

	shared_ptr<MouseEvent> mouseEvent(new MouseEvent());
	shared_ptr<WindowEvent> windowEvent(new WindowEvent(1000, 1000));

	window.attachEventHandler(mouseEvent);
	window.attachEventHandler(windowEvent);

	cudaSetDevice(0);

	/*
		Initialiize Scene
	*/
	shared_ptr<Scene> scene(new Scene(1024, 1024));
	scene->init();

	/*
		Initialiize Meshes
	*/
	shared_ptr<Mesh> m_shpereMesh(new Mesh(
		Utils::readObj(
			"C:\\Users\\dhruv\\Development\\git\\particle_attractor\\src\\resources\\objects\\sphere.obj"
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
		"C:\\Users\\dhruv\\Development\\git\\particle_attractor\\src\\resources\\glsl\\object_vs_instanced.glsl",
		"C:\\Users\\dhruv\\Development\\git\\particle_attractor\\src\\resources\\glsl\\object_fs.glsl"
	);

	/*
		Initialiize Camera
	*/
	shared_ptr<Camera> camera(new Camera(
		glm::vec3(0.0f, 0.0f, 10.0f),
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



	while (!glfwWindowShouldClose(window.window)){

		glfwPollEvents();
		mouseEvent->dispatchEvents();
		windowEvent->dispatchEvents();
		
		window.clearCanvas();
			
		particleRenderer->render(
			particleShader, scene, camera
		);
		window.update();
	
	}

	window.destroy();
	return 0;
}