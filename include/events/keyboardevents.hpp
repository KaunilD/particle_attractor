#include "events/events.hpp"

#ifndef KEYBOARDEVENT_H
#define KEYBOARDEVENT_H

class KeyboardEvent: public Event {
public:
	KeyboardEvent() : Event() {};
	
	virtual void handleEvent() {
		LOG("KeyboardEvent:: Keyboard event detected");
	}
};

#endif KEYBOARDEVENT_H
