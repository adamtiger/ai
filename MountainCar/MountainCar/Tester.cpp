#include "Tester.h"

void Tester::RunAllTests(){

	test_UpdateFeatureVectorAt();
}

void Tester::test_result_printer(bool success, std::string name){

	std::cout << "Test of " << name << " was:" << (success ? " Successful. :)" : "Failed!!! :(")  << std::endl;
}

void Tester::test_UpdateFeatureVectorAt(){

	bool success = true;



	test_result_printer(success, "UpdateFeatureVectorAt");
}
