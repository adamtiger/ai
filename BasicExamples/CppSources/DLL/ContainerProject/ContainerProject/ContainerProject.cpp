#include "stdafx.h"

#include "ContainerProject.h"

// NArmedBandit functions:

#include "Stationary.h"
#include "NonStationary.h"

float* StationaryAnalysis()
{
	bandit::stationary::Stationary stn;

	for (int i = 0; i < 10000; ++i)
		stn.DoTrial();

	float* values = new float[10];
	for (int i = 0; i < 10; ++i) {

		values[i] = stn.GetValue(i);
	}

	return values;
}

float* NonStationaryAnalysis()
{
	bandit::nonstationary::NonStationary nstn;

	for (int i = 0; i < 10000; ++i) {

		nstn.DoTrial();

		if (i % 500 == 0)
			nstn.ChangeEnv();
	}


	float* values = new float[10];
	for (int i = 0; i < 10; ++i) {

		values[i] = nstn.GetValue(i);
	}

	return values;
}


