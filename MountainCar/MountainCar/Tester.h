#ifndef __TESTER__
#define __TESTER__

#include <string>
#include <iostream>

class Tester {
	
public:

	static void RunAllTests();

private:

	static void test_result_printer(bool success, std::string name);

	// Tests:

	static void test_UpdateFeatureVectorAt();

	static void test_ExecutingAction();

	static void test_Agent();
};

#endif // __TESTSER__