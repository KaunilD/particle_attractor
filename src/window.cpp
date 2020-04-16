#include "window.hpp"
Window::Window() {};
Window::Window(int t_width, int t_height, std::string t_title)
	:m_width(t_width), m_height(t_height), m_title(t_title)
{

	if (!glfwInit()) {
		return;
	}


	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
	glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);


	this->window = glfwCreateWindow(
		m_width, m_height, m_title.c_str(), NULL, NULL
	);
	
	if (!this->window)
	{
		glfwTerminate();
		return;
	}


	glfwSetInputMode(this->window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	glfwGetFramebufferSize(this->window, &m_width, &m_height);
	glViewport(0, 0, m_width, m_height);

	glfwSetWindowUserPointer(this->window, reinterpret_cast<void*>(this));

};

void Window::attachEventHandler(shared_ptr<MouseEvent> t_mouseEvent) {
	mouseEvent = t_mouseEvent;

	glfwSetCursorPosCallback(
		this->window, [](GLFWwindow* t_window, double x, double y) {
			Window* window = reinterpret_cast<Window*>(glfwGetWindowUserPointer(t_window));
			if (window) {
				window->mouseEvent->props->setXPos(x);
				window->mouseEvent->props->setYPos(y);
			}
		});

	glfwSetScrollCallback(this->window, [](GLFWwindow* t_window, double x, double y) {
		Window* window = reinterpret_cast<Window*>(glfwGetWindowUserPointer(t_window));
		if (window) {
			window->mouseEvent->props->setXOff(x);
			window->mouseEvent->props->setYOff(y);
		}
		});

};

void Window::attachEventHandler(shared_ptr<WindowEvent> t_windowEvent) {
	windowEvent = t_windowEvent;
	glfwSetFramebufferSizeCallback(this->window, [](GLFWwindow* t_window, int h, int w) {
		Window* window = reinterpret_cast<Window*>(glfwGetWindowUserPointer(t_window));
		if (window) {
			window->m_width = w;
			window->m_height = h;
			glViewport(0, 0, w, h);
		}
		});

};

void Window::clearCanvas() {
	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
};

void Window::update() {
	frame_count += 1;
	glfwSwapBuffers(this->window);
	glFlush();
};

void Window::makeCurrent() {
	glfwMakeContextCurrent(this->window);
#ifdef _WIN32
	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK) {
		std::cout << "ERROR::MAIN.CPP::GLEW_INIT_FAILED" << std::endl;
		glfwTerminate();
	}
#endif
	glEnable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_STENCIL_TEST);
};

void Window::destroy() {
	glfwDestroyWindow(this->window);
	glfwTerminate();
};