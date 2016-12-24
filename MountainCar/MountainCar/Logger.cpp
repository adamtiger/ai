#include "Logger.h"
#include <fstream>
#include <iostream>

using namespace std;

void Logger::AddNewData(double x, double y){

	xs_.push_back(x);
	ys_.push_back(y);

	cout << "x -> " << x << " ; y -> " << y << endl;
}

void Logger::Save2File(std::string name){

	ofstream file;
	file.open(name);
	
	for (int idx = 0; idx < xs_.size(); ++idx) {

		file << idx << ": x -> " << xs_[idx] << " ; y -> " << ys_[idx] << endl;
	}

	file.close();
}
