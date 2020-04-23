#ifndef WINDOW_H
#define WINDOW_H

#include "libs.hpp"

#include "events/events.hpp"
#include "events/keyboardevents.hpp"
#include "events/mouseevents.hpp"
#include "events/windowevents.hpp"

class Window {

public:

	int m_width = 0, m_height = 0;
	int frame_count = 0;
	std::string m_title;

	GLFWwindow* window;
	shared_ptr<MouseEvent> mouseEvent;
	shared_ptr<WindowEvent> windowEvent;

	Window();
	Window(int height, int width, std::string title);

	void attachEventHandler(shared_ptr<MouseEvent> t_mouseEvent);
	void attachEventHandler(shared_ptr<WindowEvent> t_windowEvent);
	void clearCanvas();
	void update();
	void makeCurrent();
	void destroy();

};

#endif
