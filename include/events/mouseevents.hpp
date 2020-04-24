#ifndef MOUSEEVENT_H
#define MOUSEEVENT_H
#include "events/events.hpp"
#include "events/mouseprops.hpp"

class MouseEvent : public Event {
public:
	MouseEvent() : Event() {
		props = make_unique<MouseProps>();
	};
	~MouseEvent() = default;

	virtual void handleEvent() {
		LOG("MouseEvent:: Mouse Event Detected.");
	}


	virtual void dispatchEvents() {
		for (int i = 0; i < observers->size(); i++) {
			observers->at(i)->update(*props.get());
		}
		props->setYOff(0);
		props->setXOff(0);
	}
	unique_ptr<MouseProps> props;

};
#endif
