#ifndef __LEARNING_COORDINATOR__
#define __LEARNING_COORDINATOR__

#include "Agent.h"
#include "PhysicsEngine.h"
#include "CoarseCoding.h"
#include "Logger.h"


/*
	This class is responsible for managing the learning
	in an episodic way then test the agent at the end.
*/

class LearningCoordinator {

public:
	LearningCoordinator(
		int r_slices,
		int phi_slices,
		double a_car,
		double alpha,
		double gamma
	);

	~LearningCoordinator();

	void DoLearning(int nm_episodes);

	void TestAgent();

private:

	void reset_system();
	void run_epsiode();

	Action epsilon_greedy();

private:

	Agent* agent_;
	CoarseCoding* coarseCoding_;
	PhysicsEngine* engine_;
	//Logger* logger;
};

#endif // __LEARNING_COORDINATOR__