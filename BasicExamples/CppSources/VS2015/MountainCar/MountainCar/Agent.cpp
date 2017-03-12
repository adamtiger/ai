#include "stdafx.h"
#include "Agent.h"

Agent::Agent(int nm_features, double alpha, double gamma){

	nm_features_ = nm_features;
	Initialize();

	alpha_ = alpha;
	gamma_ = gamma;
}

void Agent::Initialize(){

	state_ = 0;

	if (left_thetas_.empty())
		left_thetas_.resize(nm_features_);

	if (right_thetas_.empty())
		right_thetas_.resize(nm_features_);
	
	for (int i = 0; i < nm_features_; ++i) {
		left_thetas_[i] = 0.0;
		right_thetas_[i] = 0.0;
	}
}

void Agent::UpdateThetaValues(int state, Action action, double reward, int next_state, Action next_action){

	double new_value = (1 - alpha_) * get_th(state, action) + alpha_ * (reward + gamma_ * get_th(next_state, next_action));

	set_th(state, action, new_value);
}

Action Agent::Policy(){

	return left_thetas_[state_] < right_thetas_[state_] ? RIGHT : LEFT;
}

double Agent::get_th(int state, Action action) const{

	return action == LEFT ? left_thetas_[state] : right_thetas_[state];
}

void Agent::set_th(int state, Action action, double new_value){

	action == LEFT ? left_thetas_[state] = new_value : right_thetas_[state] = new_value;
}
