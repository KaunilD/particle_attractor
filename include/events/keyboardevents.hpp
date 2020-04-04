#ifndef KEYBOARDEVENT_H
#define KEYBOARDEVENT_H

#include "events/events.hpp"

class KeyboardEvent: public Event {
public:
	KeyboardEvent() : Event() {};
	
	virtual void handleEvent() {
		LOG("KeyboardEvent:: Keyboard event detected");
	}
	
	virtual void dispatchEvents(){
		for (int i = 0; i < observers->size(); i++) {
			//observers->at(i)->update();
		}
	}
};

#endif KEYBOARDEVENT_H
