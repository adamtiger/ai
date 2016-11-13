#include "Stationary.h"
#include <random>
#include <math.h>

using namespace std;
using namespace bandit::stationary;

NormalDistribution::NormalDistribution() :
	mu_(0.0f), szigma_(1.0f) {}

NormalDistribution::NormalDistribution(float mu, float szigma) :
	mu_(mu), szigma_(szigma) {}

float NormalDistribution::Generate() const {

	float u_r = (rand() % 1000) / 1000.0f;
	float u_fi = (rand() % 1000) / 1000.0f;

	float x_gauss = szigma_ * sqrtf(-2*logf(1 - u_r)) * cosf(2 * 3.14 * u_fi) + mu_;

	return x_gauss;
}


Stationary::Stationary() {

	num_ = 10;
	float mus[10] = { 3.5, 1.9, 6.7, 3.2, 4.9, 10.5, 16.7, 8.6, 12.4, 17.1};

	machines_.resize(num_);
	action_value_.resize(num_);
	visits_.resize(num_);

	for (int i = 0; i < num_; ++i) {
		machines_[i] = new NormalDistribution(mus[i], 1);
		action_value_[i] = 0.0f;
		visits_[i] = 0;
	}
}

Stationary::~Stationary() {

	for (auto m : machines_)
	{
		delete m;
	}
}

void Stationary::DoTrial() {

	int action = choose_action();

	float rw = observe_reward(action);

	refresh_values(action, rw);
}

float Stationary::GetValue(int idx) const {

	return action_value_[idx];
}

int Stationary::choose_action() const {

	int rd = rand() % 100 + 1;
	int choosen = 0;

	int max_idx = 0;
	for (int idx = 0; idx < num_; ++idx) {
		if (action_value_[max_idx] < action_value_[idx])
			max_idx = idx;
	}

	if (rd <= 91) {

		choosen = max_idx;
	}
	else {
		rd = rd - 92;

		if (rd >= max_idx)
			choosen = rd + 1;
		else
			choosen = rd;
	}

	return choosen;
}

float Stationary::observe_reward(int action) const {

	float rw = machines_[action]->Generate();

	return rw;
}

void Stationary::refresh_values(int action, float reward) {

	visits_[action] += 1;

	float ratio = 1.0f / visits_[action];

	action_value_[action] = action_value_[action] + ratio * (reward - action_value_[action]);
}

