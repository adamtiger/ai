// NArmedBandit.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include "Stationary.h"
#include "NonStationary.h"

void StationaryAnalysis1() {

	bandit::stationary::Stationary stn;

	for (int i = 0; i < 10000; ++i)
		stn.DoTrial();

	for (int i = 0; i < 10; ++i) {

		float val = stn.GetValue(i);

		std::cout << "Action-value: " << i + 1 << " -> " << val << std::endl;
	}

	std::cout << "Finsihed" << std::endl;
}

void NonStationaryAnalysis1() {

	bandit::nonstationary::NonStationary nstn;

	for (int i = 0; i < 10000; ++i) {

		nstn.DoTrial();

		if (i % 500 == 0)
			nstn.ChangeEnv();
	}

	for (int i = 0; i < 10; ++i) {

		float val = nstn.GetValue(i);

		std::cout << "Action-value: " << i + 1 << " -> " << val << " : " << nstn.GetMu(i) << std::endl;
	}

	std::cout << "Finsihed" << std::endl;
}

int main()
{

	//StationaryAnalysis1();

	NonStationaryAnalysis1();

    return 0;
}

