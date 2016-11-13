// NArmedBandit.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include "Stationary.h"

int main()
{

	bandit::stationary::Stationary stn;

	for (int i = 0; i < 10000; ++i)
		stn.DoTrial();

	for (int i = 0; i < 10; ++i) {

		float val = stn.GetValue(i);

		std::cout << "Action-value: " << i + 1 << " -> " << val << std::endl;
	}

	std::cout << "Finsihed" << std::endl;

    return 0;
}

