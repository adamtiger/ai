#include "Environment.h"

#include <iostream>

using namespace std;
using namespace bandit;

NormalDistribution::NormalDistribution() :
	mu_(0.0f), szigma_(1.0f) {}

NormalDistribution::NormalDistribution(float mu, float szigma) :
	mu_(mu), szigma_(szigma) {}

float NormalDistribution::Generate() const {

	float u_r = (rand() % 1000) / 1000.0f;
	float u_fi = (rand() % 1000) / 1000.0f;

	float x_gauss = szigma_ * sqrtf(-2 * logf(1 - u_r)) * cosf(2 * 3.14 * u_fi) + mu_;

	return x_gauss;
}

void NormalDistribution::ModifyMu(float delta_mu) {
	mu_ += delta_mu;
}

float NormalDistribution::GetMu() const {

	return mu_;
}


BaseEnvironment::BaseEnvironment() {

	num_ = 10;
	float mus[10] = { 3.5, 1.9, 6.7, 3.2, 4.9, 10.5, 16.7, 8.6, 12.4, 17.1 };
	float szigmas[10] = { 1.0, 1.3, 0.5, 2.5, 4.6, 1.0, 0.5, 0.5, 1.0, 1.0 };

	machines_.resize(num_);
	action_value_.resize(num_);
	visits_.resize(num_);

	for (int i = 0; i < num_; ++i) {
		machines_[i] = new NormalDistribution(mus[i], 1);
		action_value_[i] = 0.0f;
		visits_[i] = 0;
	}
}

BaseEnvironment::~BaseEnvironment() {

	for (auto m : machines_)
	{
		delete m;
	}
}

float BaseEnvironment::GetValue(int idx) const {

	return action_value_[idx];
}

void BaseEnvironment::DoTrial() {

	int action = choose_action();

	float rw = observe_reward(action);

	refresh_values(action, rw);
}

int BaseEnvironment::choose_action() const {

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

float BaseEnvironment::observe_reward(int action) const {

	float rw = machines_[action]->Generate();

	return rw;
}
