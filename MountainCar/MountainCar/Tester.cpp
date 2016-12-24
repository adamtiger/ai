#include "Tester.h"
#include <assert.h>

#include "CoarseCoding.h"
#include "PhysicsEngine.h"
#include "Agent.h"

void Tester::RunAllTests(){

	test_UpdateFeatureVectorAt();
	test_ExecutingAction();
	test_Agent();
}

void Tester::test_result_printer(bool success, std::string name){

	std::cout << "Test of " << name << " was:" << (success ? " Successful. :)" : "Failed!!! :(")  << std::endl;
}

void Tester::test_UpdateFeatureVectorAt(){

	int idx;
	CoarseCoding coarseCoding(15.0, 25.0, 10, 16);

	coarseCoding.UpdateFeatureVectorAt(0,0);
	idx = coarseCoding.GetFeatureVector();
	assert(idx == 0);

	coarseCoding.UpdateFeatureVectorAt(1.7, 0);
	idx = coarseCoding.GetFeatureVector();
	assert(idx == 16);

	coarseCoding.UpdateFeatureVectorAt(-1.7, 0);
	idx = coarseCoding.GetFeatureVector();
	assert(idx == 24);

	coarseCoding.UpdateFeatureVectorAt(0.0, 5.2);
	idx = coarseCoding.GetFeatureVector();
	assert(idx == 36);

	test_result_printer(true, "UpdateFeatureVectorAt");
}

void Tester::test_ExecutingAction(){

	PhysicsEngine* physicsEngine = PhysicsEngine::CreateDefault(1.2);
	assert(physicsEngine->GetCurrentX() == 0.0);
	assert(physicsEngine->GetCurrentXdot() == 0.0);

	physicsEngine->ExecutingAction(1);
	assert(physicsEngine->GetCurrentX() == 0.0);
	assert(physicsEngine->GetCurrentXdot() != 0.0);


	delete physicsEngine;

	test_result_printer(true, "ExecutingAction");
}

void Tester::test_Agent(){

	Agent agent(10, 0.01, 0.95);
	assert(agent.Policy() == LEFT);

	test_result_printer(true, "Agent");
}
