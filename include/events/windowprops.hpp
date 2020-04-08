#ifndef WINDOWPROPS_H
#define WINDOWPROPS_H

struct WindowProps {
	int fbW = 0, fbH = 0;
	WindowProps() = default;
	WindowProps(int w, int h): fbW(w), fbH(h) {}

	void setfbH(int h) {
		this->fbH = h;
	}

	void setfbW(double w) {
		this->fbW = w;
	}
};

#endif