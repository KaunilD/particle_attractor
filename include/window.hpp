#ifndef WINDOW_H
#define WINDOW_H

#include "libs.hpp"

#include "events/events.hpp"
#include "events/keyboardevents.hpp"
#include "events/mouseevents.hpp"

class Window {

public:

	int m_width = 0, m_height = 0;

	std::string m_title;
	
	GLFWwindow* window;
	shared_ptr<MouseEvent> mouseEvent;
	shared_ptr<KeyboardEvent> keyboardEvent;

	Window();
	Window(int width, int height, std::string title);

	void handleWindowResize(GLFWwindow*, int, int);
	void handleScroll(GLFWwindow* window, double x, double y) {

	}
	void attatchEventHandlers(shared_ptr<MouseEvent> t_mouseEvent, shared_ptr<KeyboardEvent> t_keyboardEvent) {
		mouseEvent = t_mouseEvent;
		keyboardEvent = t_keyboardEvent;

		glfwSetCursorPosCallback(
			this->window, [](GLFWwindow * t_window, double x, double y) {
				Window* window = reinterpret_cast<Window*>(glfwGetWindowUserPointer(t_window));
				if (window) {
					window->mouseEvent->props->setX(x);
					window->mouseEvent->props->setY(y);
				}
		});
	};

	
	void makeCurrent() {
		glfwMakeContextCurrent(this->window);
	};
};

#endif WINDOW_H