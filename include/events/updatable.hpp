#ifndef UPDATABLE_H
#define UPDATABLE_H
#include "events/mouseprops.hpp"

class Updatable {
public:

	virtual void update(MouseProps) = 0;
};

#endif UPDATABLE_H
