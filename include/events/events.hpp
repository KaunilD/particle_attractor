#ifndef EVENTS_H
#define EVENTS_H

#include "libs.hpp"
#include "events/updatable.hpp"

class Event {
public:
	unique_ptr<std::vector<shared_ptr<Updatable>>> observers;

	Event() {
		LOG("Event::default c_tor()");
		observers = make_unique<std::vector<shared_ptr<Updatable>>>() ;
	}

	void addListener(shared_ptr<Updatable> t_updatable) {
		observers->push_back(t_updatable);
	}

	virtual void dispatchEvents() = 0;
};

#endif
