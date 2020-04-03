#include "events/events.hpp"

#ifndef MOUSEEVENT_H
#define MOUSEEVENT_H

namespace {
	struct Props {
		double xpos = 0, ypos = 0;
		void setX(double x) {
			this->xpos = x;
		}
		void setY(double y) {
			this->ypos = y;
		}
	};

	class MouseEvent : public Event {
	public:
		MouseEvent() : Event() {
			props = make_unique<Props>();
		};
		~MouseEvent() = default;
		
		virtual void handleEvent() {
			LOG("MouseEvent:: Mouse Event Detected.");
		}
		
		unique_ptr<Props> props;
	};
}
#endif MOUSEEVENT_H
