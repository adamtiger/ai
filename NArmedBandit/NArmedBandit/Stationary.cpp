#include "Stationary.h"
#include <random>
#include <math.h>

using namespace std;
using namespace bandit::stationary;


Stationary::Stationary() : BaseEnvironment() {}

Stationary::~Stationary() {}

void Stationary::refresh_values(int action, float reward) {

	visits_[action] += 1;

	float ratio = 1.0f / visits_[action];

	action_value_[action] = action_value_[action] + ratio * (reward - action_value_[action]);
}




