#ifndef __LOGGER__
#define __LOGGER__

#include <string>
#include <vector>

/*
	This class is responsible for 
	store and save all of the positions of the car during
	a test run of the agent.
*/

class Logger {

public:

	void AddNewData(double x, double y);
	void Save2File(std::string name);

private:

	std::vector<double> xs_;
	std::vector<double> ys_;
};

#endif // __LOGGER__
