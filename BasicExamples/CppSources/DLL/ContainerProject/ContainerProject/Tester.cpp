#include "Tester.h"
#include <assert.h>

#include "CoarseCoding.h"
#include "PhysicsEngine.h"
#include "Agent.h"
#include "Logger.h"

void Tester::RunAllTests(){

	test_UpdateFeatureVectorAt();
	test_ExecutingAction();
	test_Agent();
	test_Logger();
}

void Tester::test_result_printer(bool success, std::string name){

	std::cout << "Test of " << name << " was:" << (success ? " Successful. :)" : "Failed!!! :(")  << std::endl;
}

void Tester::test_UpdateFeatureVectorAt(){

	int idx;
	CoarseCoding coarseCoding(15.0, 25.0, 10, 16);

	idx = coarseCoding.GetFeatureVectorAt(0,0);
	assert(idx == 0);

	idx = coarseCoding.GetFeatureVectorAt(1.7, 0);
	assert(idx == 16);

	idx = coarseCoding.GetFeatureVectorAt(-1.7, 0);
	assert(idx == 24);

	idx = coarseCoding.GetFeatureVectorAt(0.0, 5.2);
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

void Tester::test_Logger(){

	Logger logger;

	logger.AddNewData(3.4, 4.6);
	logger.AddNewData(4.5, 6.5);
	logger.Save2File("test.csv");

	test_result_printer(true, "Logger");
}
