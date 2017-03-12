#include "NonStationary.h"

using namespace bandit::nonstationary;
using namespace std;

NonStationary::NonStationary() : BaseEnvironment(), alpha_(0.2), state_(0) {}
NonStationary::~NonStationary() {}

void NonStationary::ChangeEnv() {

	int rm = state_ % 5;
	float* deviations;

	switch (rm) {
	case 0:
		deviations = new float[num_]{ 1, 2, -1, 1, 0, -2, 1, 2, 0, 3};
		break;
	case 1:
		deviations = new float[num_]{ -1, -1, 1, 0, 2, -1, 3, 0,-2, -3 };
		break;
	case 2:
		deviations = new float[num_] { 1, 1, 0, 1, 0, -1, 2, 2, 0, 4};
		break;
	case 3:
		deviations = new float[num_] { 0, 0, 2, 0, -1, 1, 0, 0, 1, 1 };
		break;
	case 4:
		deviations = new float[num_] { 2, -1, 1, -2, 0, 2, -4, -3, 1, -3 };
		break;
	default:
		deviations = new float[num_] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	}

	for (int i = 0; i < num_; ++i)
		machines_[i]->ModifyMu(deviations[i]);

	delete[] deviations;
}

float NonStationary::GetMu(int idx) const {

	return machines_[idx]->GetMu();
}

void NonStationary::refresh_values(int action, float reward) {

	visits_[action] += 1;

	action_value_[action] = action_value_[action] + alpha_ * (reward - action_value_[action]);
}