#include "LearningCoordinator.h"

#include <random>

#define XMAX 3.14

LearningCoordinator::LearningCoordinator(int r_slices, int phi_slices, double a_car, double alpha, double gamma){
	
	engine_ = PhysicsEngine::CreateDefault(a_car);
	coarseCoding_ = new CoarseCoding(XMAX, engine_->CalcMaxSpeed(), r_slices, phi_slices);
	agent_ = new Agent(r_slices * phi_slices, alpha, gamma);
	logger_ = new Logger();
}

LearningCoordinator::~LearningCoordinator(){
	delete engine_;
	delete coarseCoding_;
	delete agent_;
	delete logger_;
}

void LearningCoordinator::DoLearning(int nm_episodes){

	for (int eps = 0; eps < nm_episodes; ++eps) {

		reset_system();
		run_epsiode();
	}
}

void LearningCoordinator::TestAgent(){

	reset_system();

	int  max_iter = 5000;
	int  it = 0;
	bool exit = false;

	int state = 0;
	Action action;

	double x, y, x_dot;

	while (!exit && it < max_iter) {

		x = engine_->GetCurrentX();
		x_dot = engine_->GetCurrentXdot();
		y = engine_->GetCurrentY();


		state = coarseCoding_->GetFeatureVectorAt(x, x_dot);
		action = agent_->Policy();

		engine_->ExecutingAction(action);

		if (fabs(x) > XMAX) {
			exit = true;
		}

		it += 1;
	}
}

Logger * LearningCoordinator::GetLogger(){

	return logger_;
}

void LearningCoordinator::reset_system(){

	engine_->ResetSystem();
	agent_->Initialize();
}

void LearningCoordinator::run_epsiode(){

	int  max_iter = 5000;
	int  it       = 0;
	bool exit     = false;

	int state = 0, next_state;
	Action action, next_action;
	double reward;

	double x, y, x_dot;

	x = engine_->GetCurrentX();
	x_dot = engine_->GetCurrentXdot();

	state = coarseCoding_->GetFeatureVectorAt(x, x_dot);
	action = epsilon_greedy();

	while (!exit && it < max_iter) {

		engine_->ExecutingAction(action);

		x = engine_->GetCurrentX();
		x_dot = engine_->GetCurrentXdot();

		next_state = coarseCoding_->GetFeatureVectorAt(x, x_dot);
		next_action = epsilon_greedy();

		if (fabs(x) > XMAX) {
			reward = 100.0;
			exit = true;
		}
		else {
			reward = -0.001;
		}
			

		agent_->UpdateThetaValues(state, action, reward, next_state, next_action);

		state = next_state;
		action = next_action;
		it += 1;
	}
}

Action LearningCoordinator::epsilon_greedy(){

	Action action = agent_->Policy();

	// Randomly decide whether to do the other action

	bool other = rand() % 100 + 1 > 90;

	if (other)
		action = action == LEFT ? RIGHT : LEFT;

	return action;
}
