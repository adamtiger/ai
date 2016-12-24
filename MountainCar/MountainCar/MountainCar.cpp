
#include "stdafx.h"
#include "Tester.h"

#include "Logger.h"
#include "LearningCoordinator.h"

int main()
{
	//Tester::RunAllTests();

	LearningCoordinator lc(10, 16, 0.2, 0.02, 0.98);

	lc.DoLearning(1000);

	lc.TestAgent();

	Logger* log = lc.GetLogger();

	log->Save2File("result.txt");

    return 0;
}

