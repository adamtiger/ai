#include "ActionSelectors.h"
#include <random>

void native::alg::ActionSelectors::EpsilonGreedy(int suggested, int* choosen){

	int r = std::rand() % 100 + 1;

	int sum = 0; bool cont = true;
	for (int i = 0; cont && i < 4; ++i) {
		sum += (i == suggested ? 85 : 5);
		if (sum >= r) {
			*choosen = i;
			cont = false;
		}
	}
}

void native::alg::ActionSelectors::Softmax(int suggested, int* choosen){

}
