#ifndef MOUSEPROPS_H
#define MOUSEPROPS_H
struct MouseProps {
	double xpos = 0, ypos = 0;
	double xoff = 0, yoff = 0;
	void setXPos(double x) {
		this->xpos = x;
	}
	void setYPos(double y) {
		this->ypos = y;
	}
	void setXOff(double x) {
		this->xoff = x;
	}
	void setYOff(double y) {
		this->yoff = y;
	}

};

#endif