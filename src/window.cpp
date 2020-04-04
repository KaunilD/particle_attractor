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
	glfwWindowHint(GLFW_SAMPLES, 4); // 4x antialiasing
	glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);


	this->window = glfwCreateWindow(
		m_width, m_height, m_title.c_str(), nullptr, nullptr
	);
	
	if (!this->window)
	{
		glfwTerminate();
		return;
	}

	glfwSetWindowUserPointer(this->window, reinterpret_cast<void*>(this));

	glfwSetInputMode(this->window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	glfwGetFramebufferSize(this->window, &m_width, &m_height);
	glViewport(0, 0, m_width, m_height);

	


};

void Window::handleWindowResize(GLFWwindow* window, int w, int h) {
	m_width = w;
	m_height = h;
}