#ifndef WINDOWEVENT_H
#define WINDOWEVENT_H
#include "events.hpp"
#include "windowprops.hpp"

class WindowEvent : public Event {
public:
	WindowEvent() : Event() {
		props = make_unique<WindowProps>();
	};
	WindowEvent(int w, int h) : Event() {
		props = make_unique<WindowProps>(w, h);
	};
	~WindowEvent() = default;

	virtual void handleEvent() {
		LOG("WindowEvent:: Mouse Event Detected.");
	}


	virtual void dispatchEvents() {
		for (int i = 0; i < observers->size(); i++) {
			observers->at(i)->update(*props.get());
		}
	}
	unique_ptr<WindowProps> props;

};
#endif
