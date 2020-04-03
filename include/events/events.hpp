#ifndef EVENT_H
#define EVENT_H

#include "libs.hpp"

class Event {
public:

	unique_ptr<std::list<shared_ptr<Updatable>>> observers;
	Event() {
		observers = make_unique<std::list<shared_ptr<Updatable>>>() ;
	}

	void addListener(shared_ptr<Updatable> t_updatable) {
		observers->push_back(t_updatable);
	}

	void dispatchEvents() {
		auto it = observers->begin();
		while (it!=observers->end()) {
			LOG("EVENT::dispatchEvents()");
			(*it)->update();
			it++;
		}
	}


};

#endif EVENT_H
