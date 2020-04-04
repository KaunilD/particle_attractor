#ifndef WINDOW_H
#define WINDOW_H

#include "libs.hpp"

#include "events/events.hpp"
#include "events/keyboardevents.hpp"
#include "events/mouseevents.hpp"

class Window {

public:

	int m_width = 0, m_height = 0;
	int frame_count = 0;
	std::string m_title;
	
	GLFWwindow* window;
	shared_ptr<MouseEvent> mouseEvent;
	shared_ptr<KeyboardEvent> keyboardEvent;

	Window();
	Window(int width, int height, std::string title);

	void handleWindowResize(GLFWwindow*, int, int);

	void attatchEventHandlers(shared_ptr<MouseEvent> t_mouseEvent, shared_ptr<KeyboardEvent> t_keyboardEvent) {
		mouseEvent = t_mouseEvent;
		keyboardEvent = t_keyboardEvent;

		glfwSetCursorPosCallback(
			this->window, [](GLFWwindow * t_window, double x, double y) {
				Window* window = reinterpret_cast<Window*>(glfwGetWindowUserPointer(t_window));
				if (window) {
					window->mouseEvent->props->setXPos(x);
					window->mouseEvent->props->setYPos(y);
				}
			}
		);

		glfwSetScrollCallback(this->window, [](GLFWwindow* t_window, double x, double y) {
				Window* window = reinterpret_cast<Window*>(glfwGetWindowUserPointer(t_window));
				if (window) {
					window->mouseEvent->props->setXOff(x);
					window->mouseEvent->props->setYOff(y);
				}
			}
		);
	};

	void clearCanvas() {
		glClearColor(0, 0, 0, 0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}

	void update() {
		frame_count += 1;
		glfwSwapBuffers(this->window);
		glFlush();
	}
	
	void makeCurrent() {
		glfwMakeContextCurrent(this->window);
		glewExperimental = GL_TRUE;
		if (glewInit() != GLEW_OK) {
			std::cout << "ERROR::MAIN.CPP::GLEW_INIT_FAILED" << std::endl;
			glfwTerminate();
		}

	};

	void destroy() {
		glfwDestroyWindow(this->window);
		glfwTerminate();
	}
};

#endif WINDOW_H