#ifndef ALGORITHMS_H
#define ALGORITHMS_H

#include "libs.hpp"

namespace Algorithms {

	struct twoInts {
		int n1, n2;
	};

	int xyToIndex(int x, int y);
	twoInts indexToxy(int index);
	bool inBounds(int x, int y);
	int randomInt(int max);

	template <typename T>
	T bound(T v, T lo, T up) {
		return std::min(up, std::max(lo, v));
	}
};

#endif
