#ifndef ALGORITHMS_H
#define ALGORITHMS_H

#include "libs.hpp"

class Algorithms {

public:

	Algorithms();
	~Algorithms();

	int xyToIndex(int x, int y);
	vector<int> indexToxy(int index);
	bool inBounds(int x, int y);
	int randomInt(int max);
};

#endif