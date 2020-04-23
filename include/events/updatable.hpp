#ifndef UPDATABLE_H
#define UPDATABLE_H
#include "events/mouseprops.hpp"
#include "events/windowprops.hpp"

class Updatable {
public:

	virtual void update(MouseProps) = 0;
	virtual void update(WindowProps) = 0;
};

#endif
