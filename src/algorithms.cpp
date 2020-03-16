#include "algorithms/algorithms.hpp"

Algorithms::Algorithms(){

	
}

Algorithms::~Algorithms() {
}
int Algorithms::randomInt(int max) {
	std::random_device dev;
	std::mt19937 rng(dev());
	std::uniform_int_distribution<std::mt19937::result_type> dist(0, max); // distribution in range [1, 6]

	return dist(rng);
}
